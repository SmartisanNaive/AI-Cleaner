#!/usr/bin/env python3
"""
Chinese AI Text Humanizer v2.0
Transforms AI-generated Chinese text to sound more natural
Features: sentence restructuring, rhythm variation, context-aware replacement, multi-pass
"""

import sys
import re
import random
import json
import os
import argparse
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level flag: whether to apply noise strategies (strategies 2 & 3)
_USE_NOISE = True

# Module-level flag: whether to expand candidates with CiLin synonyms dict
# (~40K words, offline). Off by default for deterministic-ish behavior; opt-in
# via --cilin CLI flag.
_USE_CILIN = False

_ACADEMIC_LR_MARKERS = (
    '本研究',
    '研究表明',
    '理论意义',
    '实践价值',
    '研究发现',
    '研究结果',
    '研究对象',
    '研究方法',
    '文献综述',
    '实证分析',
    '理论框架',
    '学术价值',
    '现实意义',
    '实践意义',
    '变量',
    '样本',
    '模型',
    '假设',
)

_LONGFORM_LR_CN_CHAR_THRESHOLD = 1500


def _count_chinese_chars(text):
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

# Import n-gram statistical model for perplexity feedback
try:
    from ngram_model import analyze_text as ngram_analyze
except ImportError:
    try:
        from scripts.ngram_model import analyze_text as ngram_analyze
    except ImportError:
        ngram_analyze = None

try:
    from _text_utils import join_paragraphs, split_paragraphs
except ImportError:
    from scripts._text_utils import join_paragraphs, split_paragraphs

# Module-level flag: whether to use stats optimization (can be toggled by CLI)
_USE_STATS = True
PATTERNS_FILE = os.path.join(SCRIPT_DIR, 'patterns_cn.json')

def load_config():
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

CONFIG = load_config()

# ─── Replacement Mappings ───

PHRASE_REPLACEMENTS = CONFIG['replacements'] if CONFIG else {
    '值得注意的是': ['注意', '要提醒的是', '特别说一下'],
    '综上所述': ['总之', '说到底', '简单讲'],
    '不难发现': ['可以看到', '很明显'],
    '总而言之': ['总之', '总的来说'],
    '与此同时': ['同时', '这时候'],
    '赋能': ['帮助', '提升', '支持'],
    '闭环': ['完整流程', '全链路'],
    '助力': ['帮助', '支持'],
}

# Regex-based replacements (key is regex pattern)
_REGEX_REPLACEMENTS = {}
PLAIN_REPLACEMENTS = {}

for key, val in PHRASE_REPLACEMENTS.items():
    # Check if key contains regex special chars suggesting it's a pattern
    if any(c in key for c in ['.*', '.+', '[', '(', '|', '\\']):
        _REGEX_REPLACEMENTS[key] = val
    else:
        PLAIN_REPLACEMENTS[key] = val

# Sort regex replacements by key length descending (longer patterns first)
REGEX_REPLACEMENTS = dict(sorted(_REGEX_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True))

# ─── Scene Configurations ───

SCENES = {
    'general': {
        'casualness': 0.3,
        'merge_short': True,
        'split_long': True,
        'rhythm_variation': True,
    },
    'social': {
        'casualness': 0.7,
        'merge_short': True,
        'split_long': True,
        'shorten_paragraphs': True,
        'add_casual': True,
        'rhythm_variation': True,
    },
    'tech': {
        'casualness': 0.3,
        'merge_short': True,
        'split_long': True,
        'keep_technical': True,
        'rhythm_variation': True,
    },
    'formal': {
        'casualness': 0.1,
        'merge_short': True,
        'split_long': True,
        'reduce_rhetoric': True,
        'rhythm_variation': True,
    },
    'chat': {
        'casualness': 0.8,
        'merge_short': True,
        'split_long': True,
        'shorten_paragraphs': True,
        'add_casual': True,
        'rhythm_variation': True,
    },
}

# ─── Stats-Optimized Selection ───

def pick_best_replacement(sentence, old, candidates):
    """从多个候选替换中挑选。

    Only perplexity needed for ranking — skip full analyze_text for perf.
    """
    if not _USE_STATS or not candidates or len(candidates) <= 1:
        return random.choice(candidates) if candidates else ''

    try:
        from ngram_model import compute_perplexity
    except ImportError:
        from scripts.ngram_model import compute_perplexity

    scored = []
    for candidate in candidates:
        new_sentence = sentence.replace(old, candidate, 1)
        ppl_result = compute_perplexity(new_sentence, window_size=0)
        scored.append((candidate, ppl_result.get('perplexity', 0)))

    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n <= 2:
        return scored[-1][0]
    return scored[n - 2][0]


def _compute_burstiness(text):
    """计算文本的 burstiness（困惑度变异系数），用于句式重组判断。"""
    if not _USE_STATS or not ngram_analyze:
        return None
    stats = ngram_analyze(text)
    return stats.get('burstiness', None)


# ═══════════════════════════════════════════════════════════════════
#  Strategy 1: Low-frequency bigram injection — WORD_SYNONYMS table
# ═══════════════════════════════════════════════════════════════════

WORD_SYNONYMS = {
    # ── 逻辑连接 / 转折 ──
    # Cycle 95: dropped '所以' (logic_connectors w=7 self-defeat).
    '因此': ['因而', '为此', '故而'],
    # Cycle 97: dropped '不过' from both — logic_connectors w=7 self-defeat.
    '然而': ['但', '可是', '只是'],
    # Cycle 98: dropped '然而' (logic_connectors w=7 self-defeat — replacing
    # 但是 with 然而 just trades one detected connector for another).
    '但是': ['可是', '只是'],
    '虽然': ['尽管', '即便', '就算', '纵然'],
    # Cycle 96: dropped '因此' (logic_connectors w=7 self-defeat — replacing
    # 所以 with 因此 just trades one detected connector for another).
    '所以': ['因而', '故而', '于是'],
    '而且': ['并且', '况且', '何况', '再说'],
    '或者': ['要么', '抑或', '或是', '还是'],
    '如果': ['倘若', '假如', '若是', '要是'],
    '因为': ['由于', '缘于', '出于', '鉴于'],
    '尽管': ['虽然', '即便', '纵使', '就算'],
    # ── 动词 / 行为 ──
    '能够': ['可以', '得以', '足以', '有能力'],
    '进行': ['开展', '实施', '做', '搞'],
    '实现': ['达成', '做到', '完成', '办到'],
    '提高': ['提升', '增强', '改善', '拉高'],
    # cycle 160: dropped 演进 — fixed term '发展中国家' becomes
    # '演进中国家' which reads broken (developing country, not
    # evolving). Other 发展 contexts can substitute via 推进/进展/推动.
    '发展': ['推进', '进展', '推动'],
    # '影响' removed: the idiom slot 「在 X 影响下」 is high-frequency in
    # both academic and 玄幻 register, and every candidate breaks it —
    # '波及'/'左右' are verb-only ('在...左右下' / '在...波及下' are
    # ungrammatical), '触动' is instantaneous-emotional ('在...触动下'
    # reads as 在...刺激下 but awkward), only '冲击' fits the slot. Same
    # ambiguity as the historical removals of '存在' / '有效' / '发现'.
    # cycle 160: dropped 考察 — '研究表明' commonly substituted to
    # '考察表明', which reads off-register (考察 = inspection visit).
    # Audit found in deepseek academic sample.
    # cycle 205: dropped '审视' — "本研究" → "本审视" broken
    # (审视 = critically examine, "本审视" reads as "this examination")
    '研究': ['探究', '钻研'],
    '表明': ['显示', '说明', '反映', '揭示'],
    '认为': ['觉得', '以为', '判断', '主张'],
    '需要': ['有必要', '须', '要', '得'],
    '使用': ['运用', '采用', '用', '动用'],
    '具有': ['带有', '拥有', '含有', '具备'],
    '导致': ['引发', '造成', '招致', '引起'],
    # Cycle 63: dropped '拿出' (physical/colloquial register).
    # Cycle 65: dropped '供给' — '供给' carries an economics-supply sense
    # (goods/resources), not the conceptual '提供 解释/思路/借鉴' sense.
    # Audit on 170 samples found 76 humanize-introduced 供给 cases across
    # all genres ("无法供给清晰的推理路径" / "供给代码示例" / "供给精神
    # 食粮" / "供给一面思考的镜子"). Added '给予' (grant/give) which works
    # in abstract conceptual contexts.
    # cycle 214: dropped 呈上 — overly formal "respectfully present",
    # in tech/business context "呈上聊天功能" / "呈上食物" reads off.
    # Already in _CILIN_BLACKLIST but WORD_SYNONYMS path bypassed.
    '提供': ['给出', '给予'],
    '分析': ['剖析', '解读'],  # cycle 205: drop 审视; heartbeat: drop 拆解
    '促进': ['推动', '助推', '带动', '催动'],
    '利用': ['借用', '运用', '动用', '凭借'],
    '建立': ['搭建', '构筑', '组建', '创设'],
    '引起': ['招来', '激起', '触发', '挑起'],
    '采取': ['采用', '动用', '使出', '施行'],
    '包括': ['涵盖', '囊括', '含', '包含'],
    '产生': ['催生', '引出', '萌生', '冒出'],
    '增加': ['添加', '追加', '扩充', '加大'],
    '减少': ['缩减', '削减', '降低', '裁减'],
    '保持': ['维持', '持续'],
    # cycle 229: dropped '破解' — fits "破解难题/谜团" but reads aggressive on
    # generic "解决具体问题" ("什么破解具体的问题" landed in long_blog audit).
    '解决': ['化解', '处置', '攻克'],
    '改变': ['改动', '扭转', '调整', '变化'],
    '选择': ['挑选', '选定', '选用'],
    '支持': ['撑持', '扶持', '支撑'],
    '组成': ['构成', '拼成', '组合', '凑成'],
    '形成': ['催生', '铸成', '生成', '酿成'],
    '获得': ['取得', '赢得', '得到', '揽获'],
    # cycle 164: dropped '确定' — substring matches inside 确定性 (37 hits)
    # and 不确定性 (30 hits) which are technical noun terms; substitution
    # produces broken '锁定性' / '明确性' / '不敲定性' etc. Same family of
    # bug as the historical removals of '发现' / '存在' / '有效'.
    # '发现' removed: substring inside the 4-char idiom 案发现场 gets
    # corrupted into '案察觉场'/'案觉察场'/'案识破场' when the word-level
    # substitution crosses the idiom boundary. Same family of bug as '存在'
    # / '有效' below — without proper word-boundary tagging the safe move
    # is to drop the entry. Lost LR delta is small ('发现' is mostly used
    # as a finite verb where surrounding 2-char windows already vary).
    '推动': ['驱动', '助推', '催动', '拉动'],
    '加强': ['强化', '增强', '夯实', '巩固'],
    # Cycle 78: dropped '彰显' / '凸显' — both are in detect_cn's
    # ai_high_freq_words pattern (weight 6), so injecting them as humanize
    # alts for '体现' raises the AI score (self-defeating, same family
    # as cycles 76/77). Added '反映' which is a synonym in the same
    # semantic neighborhood without being an AI-flagged term.
    '体现': ['映射', '折射', '反映'],
    '满足': ['达到', '契合', '符合', '迎合'],
    # '存在' removed: substring matches across word boundaries like 留存+在
    # → 留存有 which breaks the 留存 compound. Too error-prone without
    # word-boundary awareness.
    '属于': ['归属', '算是', '属', '归入'],
    '考虑': ['斟酌', '权衡', '琢磨', '思量'],
    # cycle 164: dropped '处理' — substring matches inside 处理器 (12 hits
    # in longform corpus, technical noun); substitution produces broken
    # '处置器' / '打理器' / '应对器'. Same as 确定/发现/存在 above.
    '参与': ['加入', '介入', '参加', '投身'],
    '创造': ['缔造', '开创', '营造', '打造'],
    '描述': ['刻画', '勾勒', '叙述', '描绘'],
    '强调': ['着重', '突出', '力陈', '重申'],
    '反映': ['映射', '折射', '体现', '呈现'],
    '应用': ['运用', '采用', '使用', '施用'],
    '结合': ['融合', '配合', '糅合', '衔接'],
    '关注': ['留意', '聚焦', '在意', '着眼'],
    '涉及': ['牵涉', '关乎', '触及', '波及'],
    '依据': ['按照', '参照', '凭', '根据'],
    # Cycle 61: dropped '取用' (informal/archaic 'fetch and use').
    # Cycle 62: dropped '引用' too — '引用' means 'cite/quote/reference',
    # not 'adopt/employ'. Same audit found 27 hits where '采用' was
    # substituted with '引用' in formal contexts ("引用对抗学习" / "引用
    # 先进的5纳米制程" / "引用复式教学法") — clear semantic error: a method
    # is adopted, not cited.
    '采用': ['选用', '沿用'],
    # ── 副词 / 程度 ──
    '目前': ['眼下', '当前', '现阶段', '如今'],
    # Cycle 80: dropped '与此同时' — it is in detect_cn's mechanical_connectors
    # pattern (weight 10), so substituting '同时' with '与此同时' raises the
    # AI score (self-defeating). Pool 4→3.
    # Cycle 80 dropped '与此同时'. Cycle 94 swap '此外'/'另外'
    # (logic_connectors w=7 self-defeat) for '同样' / '一并' (clean).
    '同时': ['并且', '同样', '一并'],
    '通过': ['借助', '凭借', '经由', '依靠'],
    '根据': ['按照', '依据', '参照', '依照'],
    # '有效' removed: word is often adjectival (有效证件/有效身份/有效期),
    # and every alternative (管用/奏效/见效/起作用) is a verb/predicate that
    # breaks attributive usage (奏效身份证件). Would need word-level POS
    # tagging to handle safely.
    '基于': ['立足于', '依托', '以…为基础', '仰赖'],
    '对于': ['针对', '就', '关于', '面对'],
    '非常': ['极其', '十分', '很', '格外'],
    # cycle 214: dropped 业已 — archaic ("已经" classical alt). In modern
    # informational text "工作功底业已落后" reads stilted. Kept for novel
    # via NOVEL_BLACKLIST_CANDIDATES exclusion (but no longer in default).
    '已经': ['早已', '已', '早就'],
    '完全': ['彻底', '全然', '纯粹', '压根'],
    '不断': ['持续', '始终', '一再', '反复'],
    '逐渐': ['渐渐', '慢慢', '一步步', '日渐'],
    # '最要紧' alt removed: when source is '最主要', substitution gives
    # '最最要紧' (doubled-最 across word boundary).
    '主要': ['核心', '关键', '首要'],
    '一般': ['通常', '往常', '照例', '大抵'],
    '大量': ['海量', '大批', '众多', '成堆的'],
    # cycle 203: dropped '更', '再' — "更进一步" → "更更" / "更再" broken;
    # "再X" reads as repetition (wrong meaning, 进一步 implies progression).
    # cycle 252: dropped '深入' — when source has "进一步深入" adjacency, sub
    # gives "深入深入" (lf:42 academic audit). 继续 is safe single alt.
    '进一步': ['继续'],
    '充分': ['尽情', '透彻', '淋漓', '饱满'],
    '直接': ['径直', '当面', '立刻', '干脆'],
    # cycle 164: '特别' alts trimmed to '尤其' only — '格外'/'极'/'分外'
    # all break inside 特别是 (56 hits in longform corpus, common
    # transition) producing '格外是'/'极是'/'分外是' which read as
    # ungrammatical. '尤其' is the one alt that survives the substring
    # collision: '特别是' → '尤其是' is a valid rewrite.
    '特别': ['尤其'],
    '一定': ['某种', '相当', '一些', '多少'],
    '必须': ['得', '务必', '非得', '须'],
    # cycle 214: dropped 兴许 — archaic dialect ("perhaps" 北方话/古风).
    # "事情兴许并不如表面所示" 读起来古旧。
    '可能': ['也许', '或许', '大概'],
    # ── 名词 / 概念 ──
    # cycle 164: dropped '重要' — substring matches inside 重要性 (28 hits)
    # and 至关重要 (16 hits) and 重要性 → 核心性 / 要紧性 / 紧要性 is
    # broken (none of those are standard Chinese nouns), 至关重要 → 至关
    # 核心 also breaks the fixed idiom. The earlier '关键' alt was already
    # dropped here (cycle ~57) for doubled-关; the remaining alts have the
    # same compound-breakage bug just less visibly.
    # Cycle 60: dropped '醒目' (visually striking, not degree adverb).
    # Cycle 66: dropped '突出' too — 突出 is verb/adjective ('stick out /
    # prominent') and doesn't work as a degree adverb. Audit found 19
    # adverb-position substitutions where it produced register/semantic
    # mismatch ('突出下降' / '突出高于' / '突出提升'). Replaced with '大幅',
    # which works as adverb of degree (118 hits in human news corpus).
    # '突出' is kept in '强调' alts where it functions as V (突出重要性).
    # cycle 202: dropped '大幅' — adverb-only, "显著进展" → "大幅进展"
    # awkward (大幅 only modifies verbs of change like 提升/下降, not nouns).
    # B-3 long_blog audit: "版本可观/明显，提升了..." shows this slot is
    # too brittle after sentence restructuring.
    # '显著': [],
    # cycle 214: dropped 症结 — too narrow (medical "crux/critical
    # bottleneck"), 破解症结 doesn't compose grammatically (症结 needs
    # 解决 / 找到, not 破解). 难题 / 麻烦 cover most contexts.
    '问题': ['难题', '麻烦'],
    # cycle 203: dropped '层面' — "多方面" → "多层面" sub broken.
    # Also dropping 维度: "多方面" → "多维度" lands in detect_cn's
    # empty_grand_words list (self-defeat). 领域 alone doesn't carry
    # the adverbial sense of 多方面, so the whole entry retires.
    # '方面': ['维度', '领域'],
    '情况': ['状况', '形势', '境况', '局面'],
    '特点': ['特征', '属性', '标志', '特色'],
    # Cycle 71: dropped '招数' — colloquial 'trick / move' (martial-arts
    # connotation), wrong register for '方法' (systematic approach). Audit
    # found 16 humanize-introduced 招数 in news/blog ("教学招数" / "学习
    # 招数" / "教育招数论" / "工作招数" / "冲洗招数"). 招数 was already
    # blacklisted for academic, so this drop only affects general/social/
    # novel where it was firing inappropriately.
    '方法': ['办法', '手段', '途径'],
    '过程': ['历程', '进程', '流程', '经过'],
    '结果': ['成果', '产物', '结局'],
    '条件': ['前提', '条件', '要件', '门槛'],
    '作用': ['功用', '效用', '效能', '功能'],
    '内容': ['要素', '成分', '要点', '素材'],
    '程度': ['幅度', '力度', '地步', '深浅'],
    '原因': ['缘由', '根源', '起因', '来由'],
    '目标': ['目的', '指向', '靶心', '方向'],
    # cycle 214: dropped 层次 — organizational/hierarchical sense, not
    # numerical. "雌激素水平 → 雌激素层次" semantically broken (hormone
    # has level/水平/水准, not hierarchy/层次).
    '水平': ['档次', '高度', '水准'],
    '范围': ['领域', '地带', '区间', '覆盖面'],
    '趋势': ['走向', '苗头', '势头', '倾向'],
    # cycle 208: dropped '实力' — "沟通能力" → "沟通实力" wrong (cycle 205
    # blocked from cilin but WORD_SYNONYMS path was missed). 实力 = strength,
    # 能力 = capability — different concepts.
    # cycle 214: dropped 功底 — too narrow (skill foundation in art/craft).
    # B-3 long_blog audit: 才干 also reads off in product prose
    # ("沟通才干"). No safe broad synonym remains.
    # '能力': [],
    '优势': ['长处', '强项', '亮点', '好处'],
    '资源': ['物资', '储备', '要素'],
    # '场景' alt removed: when source is '市场环境', substitution gives
    # '市场场景' (doubled-场 across word boundary).
    # Cycle 79: dropped '生态' — it is in detect_cn's empty_grand_words
    # pattern (weight 12, the highest). Substituting '环境' with '生态'
    # produces AI-buzzword uses ('AI生态' / '教育生态') that the detector
    # immediately flags. Added '局面' / '情境' as clean alts in the same
    # semantic neighborhood without doubled-char boundary issues.
    '环境': ['氛围', '背景', '局面', '情境'],
    '系统': ['体系', '架构', '框架'],
    '策略': ['路线', '方案', '对策', '路子'],
}

# ═══════════════════════════════════════════════════════════════════
#  Academic-scene filters for WORD_SYNONYMS
# ═══════════════════════════════════════════════════════════════════

# Global blacklist: candidates that are themselves detected as AI patterns by
# detect_cn. Substituting INTO these is self-defeating ("环境"→"生态" triggers
# empty_grand_words; "作用"→"彰显" triggers ai_high_freq_words). Applies to all scenes.
# Kept in sync with detect_cn.py CRITICAL_PHRASES + HIGH_SIGNAL_PHRASES.
_AI_PATTERN_BLACKLIST = {
    # empty_grand_words
    '赋能', '闭环', '智慧时代', '数字化转型', '生态', '愿景', '顶层设计',
    '协同增效', '降本增效', '打通壁垒', '深度融合', '创新驱动', '全方位',
    '多维度', '系统性',
    # ai_high_freq_words
    '助力', '彰显', '凸显', '焕发', '深度剖析', '加持', '赛道', '破圈',
    '出圈', '颠覆', '革新', '底层逻辑', '抓手', '链路', '触达', '心智',
    '沉淀', '对齐', '拉通', '复盘', '迭代',
}


# Words that should NOT be substituted at all in academic context.
# These are core academic vocabulary; mechanical substitution ("研究"→"探究" etc.)
# degrades readability without reducing AIGC detection score.
ACADEMIC_PRESERVE_WORDS = {
    '研究', '分析', '发现', '指出', '表明', '认为', '显示', '揭示',
    '系统', '方法', '结果', '数据', '效果', '作用', '问题', '目标',
    '应用', '提高', '能力', '影响', '过程', '条件',
}

# Candidates that are too colloquial / archaic / informal for academic writing.
# When scene='academic', these will be filtered out of the synonym candidate pool
# before picking. If only a blacklisted candidate remains, the original word is kept.
ACADEMIC_BLACKLIST_CANDIDATES = {
    # 动词 - 过于口语或古语
    '施用', '拉高', '搞', '弄', '整', '做', '做过', '搞定', '摆平',
    '挑', '琢磨', '思量', '打理', '料理', '撑持', '揽获', '敲定',
    '识破', '觉察', '察觉', '看出', '拆解', '宛若',
    # 名词/形容词 - 口语化
    '本事', '家底', '本钱', '档次', '段位', '地带', '招数', '打法',
    '麻烦', '症结', '亮点', '好处', '苗头', '势头', '门槛',
    '成堆的', '最要紧的', '海量',
    # 程度词 - 口语
    '压根', '干脆', '径直', '当面', '兴许', '估摸着', '约莫', '大抵',
    '早就', '业已',
    # 架构/框架 对 "系统" - 过于泛化
    '架构', '框架',
    # 探究/剖析/审视 对 "研究/分析" - 虽然偶尔可用但大规模替换破坏学术调性
    '探究', '剖析',
    # 连接词口语化
    '缘于', '缘由', '来由',
    # 因果/序列连词 - 在 academic 里 '于是' 倾向 sequential temporal sense
    # ('then …'), 不像 '因此 / 因而' 那样表示 logical inference. Cycle 64
    # audit found 12 academic samples with '于是 解释 / 于是 削弱 / 于是
    # 及时干预' 等 logical 上下文里被误用. 保留给 general/novel scene.
    '于是',
}


# Novel/fiction register: a subset of ACADEMIC_BLACKLIST_CANDIDATES still
# applies to 3rd-person 玄幻/武侠/小说 prose, but several entries are
# narrative-friendly verbs ('察觉'/'识破') that academic writing rejects yet
# read naturally in fiction. Carve those out so novel mode keeps useful
# perplexity-boosting substitutes while still stripping colloquial ones
# ('搞'/'拉高'/'业已') that break narrative register.
NOVEL_BLACKLIST_CANDIDATES = ACADEMIC_BLACKLIST_CANDIDATES - {
    # Action/perception verbs that fiction uses freely
    '觉察', '察觉', '识破', '看出', '拆解',
    # 海量/眼下 are 武侠/玄幻 idioms ("海量灵气" / "眼下危机")
    '海量', '眼下',
    # 古风 register friendly
    '宛若',
    # Investigation verbs OK in narrative ("探究秘境奥秘")
    '探究', '剖析',
}


def _filter_candidates_for_scene(word, candidates, scene):
    """过滤不适合场景的候选词。返回过滤后的列表，若全被过滤则返回原列表。

    Always filters _AI_PATTERN_BLACKLIST (candidates that trigger detect_cn itself).
    Additionally filters ACADEMIC_BLACKLIST_CANDIDATES when scene='academic',
    or NOVEL_BLACKLIST_CANDIDATES when scene='novel'.
    """
    filtered = [
        c for c in candidates
        if c not in _AI_PATTERN_BLACKLIST and c not in _CILIN_BLACKLIST
    ]
    if scene == 'academic':
        filtered = [c for c in filtered if c not in ACADEMIC_BLACKLIST_CANDIDATES]
    elif scene == 'novel':
        filtered = [c for c in filtered if c not in NOVEL_BLACKLIST_CANDIDATES]
    return filtered if filtered else candidates


# ═══════════════════════════════════════════════════════════════════
#  CiLin (哈工大同义词词林扩展版) - optional expansion
# ═══════════════════════════════════════════════════════════════════

_CILIN_CACHE = None
_CILIN_FILE = os.path.join(SCRIPT_DIR, 'cilin_synonyms.json')

# Curated blacklist of CiLin candidates that are archaic, domain-mismatched,
# or POS-mismatched for common Chinese words. CiLin's "synonym" relation is
# taxonomic (not substitutable), so these slip through — manually filtered
# from spot-checks on 应用/发展/重要/系统/分析/提高/使用.
_CILIN_BLACKLIST = {
    # Archaic / 文言 — "conscript/order-around" tone for 使用/应用
    '使唤', '使役', '役使', '差遣', '驱使',
    # Mismatched POS (noun / noun-phrase for adjective 重要)
    '严重性', '要紧性', '关键性', '基本点', '国本',
    # Domain-mismatched (upward-numerical for 发展)
    '上扬', '上移', '上进', '升华',
    # 发展 alts: 前行/前进 = literal motion, "X的发展前景" → "X的前行前景" broken
    '前行', '前进',
    # Archaic / classical for 系统
    '板眼', '伦次', '条贯', '战线',
    # Overly colloquial / butcher-y for 分析
    '剖解', '解构',
    # Redundant / unnatural
    '显要', '要害', '紧要',
    # cycle 150 quality audit additions —— cilin synonyms that broke
    # semantics in real bn=10 humanize output across academic / novel
    # / review samples. each entry: source word → bad synonym observed.
    # Poetic / descriptive for 最高 ("highest" — quantitative)
    '万丈', '亭亭', '凌云', '参天', '摩天', '高高的',
    # Wrong scale / register for 团队 ("team")
    '团伙', '集团',
    # Technical / wrong-POS for 核心 ("core")
    '主从', '为主',
    # Assembly / event mismatch for 会议 ("meeting")
    '集会',
    # Specific-context for 完成 ("complete")
    '交卷', '到位', '姣好', '完了', '完事',
    # Wrong meaning for 问题 ("problem/issue")
    '主焦点', '事端', '关节', '关子',
    # Wrong meaning for 进行 ("conduct")
    '前进', '行进',
    # POS / meaning mismatch found in cycle 150 quality audit
    '容许',  # replaces 可能 — verb instead of modal
    '呈上',  # replaces 提供 — overly formal "submit upward"
    # cycle 186: cilin 领域 alts that mean physical land, wrong for
    # abstract domain — 教育领域 → 教育土地/园地/国土/圈子/天地 broken
    '土地', '园地', '国土', '圈子', '天地',
    # cycle 195: broken alts surfaced in README humanize 输出 audit
    '念书',  # 学习 alt — "深度学习" → "深度念书" semantically wrong
    '攻读',  # 学习 alt — only "study academically", off in "深度学习"
    '学学',  # 学习 alt — broken (just repeated char)
    '修业',  # 学习 alt — archaic ("study at school")
    '上学',  # 学习 alt — only "go to school", off in tech contexts
    '就学',  # 学习 alt — same as 上学
    '肥力',  # 精力 alt — 肥力 means soil fertility (土壤肥力)
    '个私',  # 个人 alt — regional/dialect, off in formal text
    '人家',  # 个人 alt — pronoun "she/he/they", semantic shift
    '匹夫',  # 个人 alt — archaic "common person"
    '一发',  # 更加 alt — archaic, "一发充实" reads broken
    '事体',  # 工作/事情 alt — regional dialect, off in formal text
    '本性',  # 个性 alt — "个性化" → "本性化" broken (本性 ≈ nature)
    '天性',  # 个性 alt — "个性化" → "天性化" broken
    '生性',  # 个性 alt — "个性化" → "生性化" broken
    '秉性',  # 个性 alt — same broken pattern
    '赋性',  # 个性 alt — same broken pattern
    '擘画',  # 规划/计划 alt — archaic, off in modern Chinese
    '宏图',  # 规划/计划 alt — "任务规划" → "任务宏图" wrong (宏图 = grand vision)
    '圈圈',  # 层面/局面/范畴 alt — wrong meaning ("circle")
    '框框',  # 层面/范畴 alt — wrong meaning ("frame")
    '局面',  # 层面 alt — "各个层面" → "各个局面" awkward
    '对头',  # 正确/科学 alt — colloquial "correct/foe", semantic shift
    '不利',  # 正确/科学 alt — opposite meaning ("unfavorable")!
    '不易',  # 正确/科学 alt — unrelated ("not easy")
    '得法',  # 正确/科学 alt — narrow ("appropriate method")
    '上头',  # 方面 alt — body part ("top of head")
    '恰切',  # 适应 alt — "自适应" → "自恰切" broken
    '出发点',  # 角度 alt — "从角度" → "从出发点" register-narrow
    '动用',  # 应用/使用 alt — "应用" → "动用" implies mobilizing resources
    '深浅',  # 深度 alt — "深度学习" → "深浅学习" broken
    '纵深',  # 深度 alt — military register, off
    '穿越',  # 通过 alt — "通过" → "穿越" wrong (穿越 = traverse)
    '穿过',  # 通过 alt — same wrong meaning
    '越过',  # 通过 alt — same wrong meaning
    '适于',  # 适应 alt — "自适应" → "自适于" broken
    '升任',  # 提升 alt — only "promote in rank"
    '升官',  # 提升 alt — same job-promotion narrow
    '升迁',  # 提升 alt — same job-promotion narrow
    '提干',  # 提升 alt — same, military/cadre register
    '咱家',  # 个人 alt — colloquial regional ("us/me"), wrong meaning
    '助长',  # 推动 alt — implies negative ("AI 推动教育" → "AI 助长教育" wrong, 助长 = abet/encourage-bad)
    '事理',  # 道理 alt — archaic register, off in modern Chinese
    '理路',  # 道理 alt — same archaic
    '所以然',  # 道理 alt — too philosophical, off in modern Chinese
    '技巧',  # 技术 alt — narrow "skill", off in tech contexts
    '招术',  # 技术 alt — wuxia register, very off
    '规模',  # 层面/范畴 alt — wrong dimension ("scale" not "aspect")
    '升格',  # 提升 alt — "upgrade to higher class", off in skill/effort contexts
    '升级',  # 提升 alt — software/version register, off in many contexts
    '数目字',  # 数字 alt — "数字化" → "数目字化" broken (数目字 = numerical figure)
    # cycle 203 (sway 语句通顺优先 directive): more broken alts surfaced
    '兼具',  # 具有 alt — narrow "include both", "兼具广阔前景" broken
    '由此',  # 通过 alt — connector word, "由此各方合力" broken (loses 通过 means "via")
    '稿子',  # 规划/计划 alt — colloquial "draft", off in formal "任务稿子"
    '不错',  # 科学 alt — informal compliment, "践行不错的时间管理" broken
    '正值',  # 正在 alt — only with time periods (正值春季), broken in "正值推动"
    '条理',  # 系统 alt — "智能评估系统" → "智能评估条理" broken (条理 = orderliness)
    '功用',  # 意义/作用 alt — narrow "function", "意义" → "功用" register-mismatched
    # cycle 205 (sway 语义不通畅 directive 续):
    '世界',  # 领域 alt — "教育领域" → "教育世界" semantic shift (世界 = world)
    '实力',  # 能力 alt — "沟通能力" → "沟通实力" wrong (能力 = capability, 实力 = strength)
    '体系',  # 系统 alt — "智能评估系统" → "智能评估体系" register-mismatched
    '审美',  # 审视 alt — "审视" → "审美" totally wrong meaning (aesthetic judgment)
    '琢磨',  # 研究 alt — informal "ponder", off in formal contexts
    '作用',  # 意义 alt — "真正意义上" → "真正作用上" broken (作用 = function, 意义 = meaning/significance)
    '力量',  # 意义/能力 alt — "真正意义上" → "真正力量上" broken
    '功力',  # 意义 alt — "真正意义上" → "真正功力上" broken (功力 = 内力 wuxia)
    '功效',  # 意义 alt — "真正意义上" → "真正功效上" broken
    '功能',  # 意义 alt — "真正意义上" → "真正功能上" broken (function not meaning)
    '今朝',  # 现在 alt — archaic poetic register ("今朝有酒今朝醉"), off in modern prose
    '目下',  # 目前 alt — archaic ("at present" classical Chinese), sway flagged msg 2198
    '手上',  # 目前 alt — colloquial "in hand", off in formal/academic
    '时下',  # 目前 alt — narrow ("nowadays" trend-context), off in research register
    # cycle 208 (sway 整理 README sweep):
    '于今',  # 现在 alt — archaic, "于今" 不像现代汉语
    '今日',  # 现在 alt — slightly poetic, off in modern prose ("今日X" 报纸 register)
    '今昔',  # 现在 alt — comparative "now and then", different meaning
    '参酌',  # 研究 alt — archaic "consult and consider", off in modern formal
    '掂量',  # 研究 alt — colloquial "weigh up"
    '揣摩',  # 研究 alt — narrow "ponder/figure out"
    '斟酌',  # 研究 alt — narrow "deliberate carefully", off in technical research
    '切磋',  # 研究 alt — narrow "exchange skills" (martial arts/scholarly)
    '技艺',  # 技术 alt — narrow "art/craft", off in tech contexts
    '技能',  # 技术 alt — narrow "skill", off when 技术 means "technology"
    '反过来看',  # noise/transition alt — odd opener mid-essay
    '说到这里',  # noise/transition alt — narrative voice, off in essay
    '人为',  # 人工 alt — "人工智能" → "人为智能" broken (人为=man-made, conceptually different)
    '人造',  # 人工 alt — same; "人造智能" reads as "fake AI"
    '力士',  # 人工 alt — totally different ("strongman")
    '人力',  # 人工 alt — "人工智能" → "人力智能" broken (人力 = manpower)
    '教养',  # 教育 alt — "教育教学" → "教养教学" broken (教养=upbringing/manners)
    '教化',  # 教育 alt — moralistic tone, off in modern AI/tech context
    '感化',  # 教育 alt — moralistic, off
    '启蒙',  # 教育 alt — narrow ("enlighten" beginner level)
    '教诲',  # 教育 alt — moralistic ("teaching/admonition"), off
    '教导',  # 教育 alt — narrow ("guide/instruct"), off in 教育领域
    '力促',  # 推动 alt — archaic ("forcefully promote")
    '方略',  # 规划/计划 alt — military/strategic, off in "任务规划" → "任务方略"
    '透过',  # 通过 alt — physical "penetrate through", off in 通过合力 context
    '末了',  # 最后 alt — colloquial dialect
    '末后',  # 最后 alt — archaic
    '末尾',  # 最后 alt — physical position, off in temporal context
    '尾子',  # 最后 alt — colloquial
    '尾声',  # 最后 alt — narrow ("finale" of event/work)
    '鹏程',  # 前景 alt — mythological "Peng's flight", way too poetic
    '奔头儿',  # 前景 alt — colloquial dialect ("something to look forward to")
    '乌纱',  # 前程 alt — archaic "official's hat", career-narrow
    '乌纱帽',  # 前程 alt — same
    '功名',  # 前程 alt — imperial-exam era register
    '前程',  # 前景 alt — career-path slot, breaks "广阔的发展前景" idiom
    '前途',  # 前景 alt — career-track slot, "广阔的发展前途" reads off in 现代 prose
    '兼备',  # 具有 alt — "具有X" → "兼备X" requires plural object
    '议会',  # 会议 alt — "parliament", totally different ("此次议会" 错)
    '集会',  # 会议 alt — narrow ("rally"), off in business meeting context
    '治理',  # 管理 alt — "governance", domain-shift from "manage"
    '治本',  # 管理 alt — narrow medical idiom ("treat root cause")
    '治治',  # 管理 alt — colloquial duplicate ("punish/teach a lesson")
    '管事',  # 管理 alt — narrow ("be in charge of trifles"), colloquial
    '贯彻',  # 实现 alt — narrow ("carry through policy"), "技术实现" → "技术贯彻" 错
    '落实',  # 实现 alt — narrow ("implement policy"), "技术实现" → "技术落实" 错
    '装具',  # 设备 alt — military equipment, "厨房设备" → "厨房装具" 错
    '两样',  # 不同 alt — colloquial, "上百种不同" → "上百两样" 病句
    '释疑',  # 解释 alt — classical, "希望找到一种解释" → "找到一种释疑" 古文
    '训诂',  # 解释 alt — narrow (textual exegesis), 古文 register
    '层系',  # 层次 alt — geological layer, semantic shift
    '兴许',  # 可能 alt — archaic dialect ("perhaps" 北方话), reads stilted
    '业经',  # 已经 alt — formal/legal classical
    '著录',  # 记录 alt — narrow (catalog/formally record), "聊天记录" → "聊天著录" 错
    '笔录',  # 记录 alt — narrow (deposition), domain-shift from generic record
    '记要',  # 记录 alt — minute-taking, narrow
    '主焦点',  # 症结 alt — non-word
    '关子',  # 症结 alt — narrow ("punchline of joke")
    '各别',  # 不同 alt — colloquial "individually different", "上百各别" 错
    '唯恐',  # 可能 alt — archaic "for fear that", "事情可能" → "事情唯恐" 错
    '例外',  # 不同 alt — "exception", "上百种不同" → "上百例外" 病句
    '下狠心',  # 决定 alt — colloquial idiom "make up mind", "决定您" → "下狠心您" 错
    '主宰',  # 决定 alt — too strong ("rule over")
    '了得',  # 决定 alt — different meaning ("remarkable")
    '仲裁',  # 决定 alt — legal/formal arbitration
    '公决',  # 决定 alt — legal/formal public decision
    '公断',  # 决定 alt — legal/formal public arbitration
    '品目',  # 种类 alt — narrow ("article entries" in catalog)
    '档级',  # 种类 alt — narrow ("rank/grade")
    # cycle 216 longform audit additions:
    '惨遭',  # 面临 alt — "suffer tragically" wrong tone for "面临挑战"
    '屡遭',  # 面临 alt — narrow (repeatedly suffer)
    '倍受',  # 面临 alt — only fits 关注/重视 (positive), "倍受挑战" 错
    '备受',  # 面临 alt — same constraint
    '未遭',  # 面临 alt — archaic
    # cycle 218 longform inject_noise + cilin audit additions:
    '筋肉',  # 肌肉 alt — Japanese-derived, off in modern Chinese
    '年华',  # 时间 alt — poetic ("youth"), "恢复年华" 错
    '光阴',  # 时间 alt — poetic, off in tech/business
    '岁时',  # 时间 alt — archaic
    '年光',  # 时间 alt — poetic
    '上马',  # 开始 alt — colloquial "start project (大跃进 era)"
    '伊始',  # 开始 alt — formal "at the start", off in casual
    '先声',  # 开始 alt — narrow ("first signs/prelude")
    '城池',  # 城市 alt — ancient walled city
    '城邑',  # 城市 alt — archaic
    '地市',  # 城市 alt — gov-policy "city-prefecture"
    '大哥大',  # 手机 alt — 90s mobile phone
    '无绳机',  # 手机 alt — cordless landline phone
    '固化',  # 稳定 alt — "solidify" wrong meaning
    '安乐',  # 稳定 alt — "peaceful/comfortable"
    '原则性',  # 稳定 alt — "principled" wrong slot
    # cycle 220 quality cleanup:
    '不无',  # 具有/具备/拥有 alt — literary double-negative, "不无广阔" 错
    '万顷',  # 广阔 alt — ancient land measure (万顷土地)
    '周边',  # 广阔 alt — "peripheral", wrong slot for 广阔
    '周遍',  # 广阔 alt — archaic
    '宏阔',  # 广阔 alt — formal/literary
    '能事',  # 能力 alt — narrow ("things one can do well")
    '能耐',  # 能力 alt — colloquial "skill/ability"
    '身手',  # 能力 alt — narrow ("agility/martial skill")
    '意志',  # 旨在 alt — "willpower" not "purpose", "意志提高" 错
    '心意',  # 旨在 alt — "intention" but not "aim", same slot mismatch
    '意旨',  # 旨在 alt — archaic "imperial decree"
    '旨意',  # 旨在 alt — same archaic
    '法旨',  # 旨在 alt — Buddhist/imperial decree
    '拍卖',  # 处理 alt — "auction" totally different domain
    '处分',  # 处理 alt — narrow disciplinary action
    '上座',  # 首席 alt — "seat of honor", wrong for executive title
    '上位',  # 首席 alt — narrow ("upper position")
    '剖示',  # 展示 alt — non-word/very rare
    '兆示',  # 展示 alt — narrow archaic ("portend")
    '呈示',  # 展示 alt — formal/legal narrow
    '试点县',  # 县城 alt — "pilot county" gov-policy specific
    '版纳',  # 县城 alt — actual place name (西双版纳), nonsense as alt
    '京都',  # 北京 alt — Kyoto / archaic capital
    '上京',  # 北京 alt — archaic
    '京华',  # 北京 alt — poetic
    '京城',  # 北京 alt — slightly archaic, ok in some contexts but mostly off
    '京师',  # 北京 alt — imperial-era term
    '凤城',  # 北京 alt — poetic name for capitals
    '中标',  # 成功 alt — "win bid" commercial
    '交卷',  # 成功 alt — "submit exam paper"
    '姣好',  # 成功 alt — "beautiful" not "successful"
    '完了',  # 成功 alt — "finished" not "succeeded"
    '到位',  # 成功 alt — narrow ("in place"), often wrong slot
    '作派',  # 主义 alt — "mannerism/style" wrong, "存在主义" → "存在作派" 错
    '官气',  # 主义 alt — "bureaucratic air"
    '架子',  # 主义 alt — "framework/airs"
    '作风',  # 主义 alt — "style" sometimes ok but breaks 主义 idioms
    '犯得上',  # 值得 alt — colloquial "worth doing" 北方话
    '犯得着',  # 值得 alt — same
    '其时',  # 当时 alt — archaic "at that time"
    '讲堂',  # 教室 alt — formal/grand "lecture hall"
    '归于',  # 归属 alt — preposition rather than noun, "归于的" 怪
    # cycle 221 academic 5-sample audit additions:
    '胁从',  # 威胁 alt — "be coerced into" (legal term)
    '威慑',  # 威胁 alt — narrow ("intimidate")
    '威逼',  # 威胁 alt — colloquial coercion
    '胁迫',  # 威胁 alt — legal coercion
    '体贴',  # 关注 alt — personal "considerate"
    '关怀',  # 关注 alt — soft "show concern (caring)"
    '关爱',  # 关注 alt — "love & care"
    '眷顾',  # 关注 alt — formal/literary "favor"
    '求战',  # 挑战 alt — military "ask for battle"
    '包罗',  # 包括 alt — formal "encompass all"
    '席卷',  # 包括 alt — "sweep over"
    '揽括',  # 包括 alt — "monopolize/include all"
    '强攻',  # 攻击 alt — military
    '抢攻',  # 攻击 alt — sports/military
    '挨斗',  # 攻击 alt — political-era "denounced"
    '掊击',  # 攻击 alt — archaic
    '反响',  # 影响 alt — "echo/response", narrow
    '反射',  # 影响 alt — physics
    # 反应 NOT blocked: high-frequency word, blocking costs HC3 avg -0.5
    # without clear fluency win in informational text. Keep alt available.
    '反馈',  # 影响 alt — narrow technical
    '想当然',  # 影响 alt — different ("take for granted")
    '感应',  # 影响 alt — physics/spiritual
    '安好',  # 安全 alt — 古文 "well-being"
    '安康',  # 安全 alt — 古文 "health"
    '安然',  # 安全 alt — adverb-like "safely"
    '康宁',  # 安全 alt — 古文 "peace"
    '无恙',  # 安全 alt — 古文 "no harm"
    '音信',  # 信息 alt — 古文 "tidings"
    '音尘',  # 信息 alt — 古文
    '音息',  # 信息 alt — 古文
    '音讯',  # 信息 alt — 古文
    '音问',  # 信息 alt — 古文
    '中坚',  # 核心 alt — narrow ("backbone force")
    '争辩',  # 理论 alt — different ("argue")
    '争鸣',  # 理论 alt — narrow ("contend")
    '反驳',  # 理论 alt — different ("rebut")
    '回驳',  # 理论 alt — same
    '声辩',  # 理论 alt — narrow ("plead")
    '仰承',  # 利用/凭借 → ??? chain — deferential "accept respectfully"
    '拼杀',  # 攻击 alt — "fight to death", off in academic
    # cycle 222 news/blog/review audit additions:
    '构造',  # 布局 alt — "structure" wrong slot, "战略布局" → "战略构造" 错
    '师资',  # 老师 alt — collective noun, "一名师资" 错
    '园丁',  # 老师 alt — metaphor, off in factual text
    '产物',  # 结果 alt — "product", semantic shift
    '下场',  # 结果 alt — narrow ("downfall")
    '下文',  # 结果 alt — narrow ("subsequent passage")
    '了局',  # 结果 alt — archaic
    '分晓',  # 结果 alt — narrow ("decisive outcome")
    '名堂',  # 结果 alt — colloquial ("trick/explanation")
    '部署',  # 配置 alt — "deploy" military/IT slot
    '嬉水',  # 游戏 alt — "play in water" totally wrong
    '嬉戏',  # 游戏 alt — narrow "play"
    '一日游',  # 游戏 alt — "one-day tour"
    '休闲游',  # 游戏 alt — "leisure tour"
    '好耍',  # 游戏 alt — colloquial "fun"
    '差一点',  # 几乎 alt — "almost (didn't)" different meaning
    '差点儿',  # 几乎 alt — colloquial 北方话
    '常备',  # 日常 alt — "regular/ready" wrong slot
    '万般',  # 日常 alt — "all kinds" wrong slot
    '一般性',  # 日常 alt — "general" wrong slot
    '处事',  # 处理 alt — substring collision with 处理器, "处理器" → "处事器" 错
    '措置',  # 处理 alt — formal/archaic
    '凌厉',  # 强烈 alt — narrow ("sharp/fierce")
    '凶猛',  # 强烈 alt — narrow ("fierce")
    '利害',  # 强烈 alt — narrow ("intense/clever")
    '剧烈',  # 强烈 alt — narrow ("violent")
    '凭依',  # 利用 alt — formal/literary
    '从新',  # 重新 alt — colloquial 北方话
    '再也',  # 重新 alt — adverb-only
    '再行',  # 重新 alt — formal
    '双重',  # 重新 alt — wrong meaning ("double")
    '允当',  # 适合 alt — formal/archaic
    '切合',  # 适合 alt — narrow ("fit closely")
    '切当',  # 适合 alt — formal/archaic
    '合乎',  # 适合 alt — formal pre-noun
    '合宜',  # 适合 alt — formal/archaic
    '搭架子',  # 布局 alt — colloquial "set up framework"
    '份额',  # 重量 alt — "share/portion" wrong
    '净重',  # 重量 alt — narrow ("net weight")
    '千粒重',  # 重量 alt — agricultural specific
    '毛重',  # 重量 alt — narrow ("gross weight")
    '咋舌',  # 惊叹 alt — "tongue-tied" wrong
    '奇异',  # 惊叹 alt — "strange/peculiar" semantic shift
    '奇怪',  # 惊叹 alt — "strange"
    '希罕',  # 惊叹 alt — "rare/curious"
    '惊呆',  # 惊叹 alt — narrow "stunned"
    # cycle 230 long_blog audit additions:
    '上心',  # 专注 alt — verb "take to heart"; "上心于X" 不通 (上心 doesn't take 于)
    '寻常',  # 通常 alt — literary "ordinary"; "数据寻常用于" reads off-register
    # cycle 231 general/workplace audit additions:
    '推向',  # 推动 alt — needs directional object; "推向教育的大趋势" 不通
             # (推向 expects target/level: "推向更高水平"; not abstract "X的大趋势")
    # cycle 232 long_blog seed=1 audit additions — bad alts in 变化 family:
    '变卦',  # 变化 alt — narrow ("renege on agreement"); "动态变卦" 不通
    '事变',  # 变化 alt — historical event ("七七事变"); wrong slot for generic 变化
    '变故',  # 变化 alt — "mishap/misfortune" too negative for generic context
    '变型',  # 变化 alt — likely typo of 变形, narrow material-science slot
    '切变',  # 变化 alt — physics term ("shear"), wrong slot for generic 变化
    # cycle 234 long_blog seed=7 audit additions — narrow 实现/创造 cilin alts:
    '促成',  # 实现 alt — "facilitate/bring about", off for "技术实现" → "技术促成"
    '兑现',  # 实现 alt — "fulfill (a promise)", narrow; "技术兑现" 不通
    '创设',  # 创造 alt — "set up institution"; "创设产品" off (institutions/laws fit, products don't)
    # cycle 235 general seed=7 audit additions — narrow 学生 + modal-mismatched alts:
    '学员',  # 学生 alt — "trainee" narrow, off for generic "学生" in education context
    '学童',  # 学生 alt — "young pupil" narrow age slot
    '桃李',  # 学生 alt — metaphor "students/disciples" idiom-only
    '可知',  # 能够/亦可 alt — modal mismatch ("it can be known"), not "can do"
    # cycle 236 long_blog seed=42 audit additions:
    '保险',  # 确保/保证 alt — "insurance" connotation; "保险产品能够" reads "insurance product"
    '一对一', # 一定/相当 alt — completely wrong meaning ("one-on-one"); "一对一要具备" 不通
    # cycle 237 longform sample audit — 首先/其次 family bad alts:
    '处女',  # 首先/状元/首度 alt — "maiden/virgin"; off in any modern context
    '头条',  # 首先/状元/首度 alt — "headline"; wrong slot
    '头版',  # 首先/首度 alt — "front page"; wrong slot
    '排头',  # 首先/首度 alt — "front of line"; wrong slot
    '伯仲',  # 其次 alt — "brothers/peers"; wrong slot
    '老二',  # 其次 alt — colloquial "second son"; vulgar register
    '第二性', # 其次 alt — Beauvoir's book "The Second Sex"; specific cultural reference
    # cycle 238 systematic word-family scan — 形成/帮助/需要/降低/考虑 wrong-slot alts:
    # 形成 family (all alts mean "achieve/in-place/mutate"; none fits "form/take shape"):
    '做到',  # 形成 alt — "achieve"; "形成共识" → "做到共识" 不通 (also in WORD_SYNONYMS 实现 alts, that path unaffected)
    '变异',  # 形成 alt — biology "mutation"; wrong slot
    '善变',  # 形成 alt — character trait "fickle"; wrong POS
    '多变',  # 形成 alt — adjective "changeable"; wrong POS
    # 帮助 family wrong-meaning alts:
    '佑助',  # 帮助 alt — literary "bless and help"; archaic
    '匡助',  # 帮助 alt — formal "assist (the upright)"; archaic register
    '匡扶',  # 帮助 alt — "support (the righteous)"; archaic political register
    '受助',  # 帮助 alt — OPPOSITE direction! "receive help"
    '增援',  # 帮助 alt — military "reinforce"; wrong slot
    # 需要 family wrong-slot alts:
    '内需',  # 需要 alt — "domestic demand" (economic noun); wrong slot
    '特需',  # 需要 alt — "special needs" (medical/service noun); wrong slot
    '索要',  # 需要 alt — "demand insistently"; wrong tone
    # 降低 family financial-narrow alts:
    '下挫',  # 降低 alt — financial "drop sharply"; narrow
    '下滑',  # 降低 alt — financial/sports "decline"; narrow
    '下跌',  # 降低 alt — financial "fall"; narrow
    '低落',  # 降低 alt — emotional "in low spirits"; wrong slot
    # 考虑 family wrong-meaning alts:
    '合计',  # 考虑 alt — "calculate/sum up"; wrong meaning
    # cycle 239 systematic word-family scan continued — 收集/工作/问题/方法/关系/特点:
    '募集',  # 收集 alt — "raise funds"; narrow financial
    '筹募',  # 收集 alt — "fundraise"; narrow financial
    '收载',  # 收集 alt — "include/contain (in book)"; wrong slot
    '综采',  # 收集 alt — "synthesize and extract"; archaic
    '专职',  # 工作 alt — "professional/full-time" noun; wrong slot
    '作业',  # 工作 alt — "homework/operations"; different concept
    '事故',  # 问题 alt — "accident/incident"; wrong slot for generic 问题
    '抓挠',  # 方法 alt — colloquial "scratch/lousy way"; wrong meaning
    '具结',  # 关系 alt — legal "guarantee in writing"; narrow
    '风味',  # 特点 alt — "flavor"; food-only
    # cycle 240 systematic adjective family scan — wrong-meaning/colloquial alts:
    # 深入 family:
    '中肯',  # 深入 alt — "to the point"; different concept
    '刻骨',  # 深入 alt — "deeply engraved"; emotional only
    # 明显 family:
    '扎眼',  # 明显 alt — "eye-catching" colloquial
    '明摆着', # 明显 alt — colloquial register
    # 清晰 family:
    '丁是丁', # 清晰 alt — "clear-cut" colloquial idiom
    '清丽',  # 清晰 alt — "clear and beautiful"; wrong slot
    # 复杂 family:
    '繁体',  # 复杂 alt — "traditional Chinese characters"; wrong slot
    # 稳定 family:
    '一贯',  # 稳定 alt — adverb "consistently"; wrong POS
    '原则性', # 稳定 alt — formal noun "principled"; wrong POS
    # 快速 family:
    '不会儿', # 快速 alt — colloquial 北方话 "in a moment"
    '劈手',  # 快速 alt — archaic "swiftly with hand"
    '快当',  # 快速 alt — colloquial "quick/efficient"
    # 准确 family:
    '准儿',  # 准确 alt — colloquial "for sure"
    '纯正',  # 准确 alt — "pure/authentic"; wrong slot
    # 有效 family:
    '中用',  # 有效 alt — colloquial "useful"
    '合用',  # 有效 alt — narrow "fit for use"
    # 严格 family:
    '严词',  # 严格 alt — narrow noun "stern words"
    # 强大 family:
    '强压',  # 强大 alt — "suppress strongly"; wrong slot
    # cycle 241 systematic domain noun scan — wrong-slot/narrow alts:
    # 经济 family (cost-effective ≠ economy):
    '上算',  # 经济 alt — "cost-effective" colloquial
    '划得来', # 经济 alt — "worthwhile" colloquial
    '占便宜', # 经济 alt — "take advantage" colloquial
    # 文化 family:
    '双文明', # 文化 alt — specific term ("two civilizations")
    # 产业 family (whole family means "family property", wrong slot for industry):
    '家业',  # 产业 alt — "family business/property"
    '家事',  # 产业 alt — "family affairs"
    '家产',  # 产业 alt — "family assets"
    '家底',  # 产业 alt — "family savings"
    '家当',  # 产业 alt — "household belongings"
    '家私',  # 产业 alt — "household goods"
    # 行业 family:
    '本行',  # 行业 alt — narrow "one's own line of work"
    '正业',  # 行业 alt — idiom "proper occupation"
    # 企业 family (specific store types ≠ enterprise):
    '代销店', # 企业 alt — "consignment store"
    '供销社', # 企业 alt — "supply and marketing cooperative"
    '信用社', # 企业 alt — "credit union"
    '合作社', # 企业 alt — "cooperative"
    '商厦',  # 企业 alt — "commercial building"
    # 政府 family:
    '内阁',  # 政府 alt — narrow "cabinet"
    # 国家 family (archaic poetic):
    '江山',  # 国家 alt — "rivers and mountains"; poetic
    '社稷',  # 国家 alt — "altars to gods of soil/grain"; archaic
    # 世界 family:
    '世道',  # 世界 alt — "ways of the world"; archaic
    '世风',  # 世界 alt — "social mores"; archaic
    '中外',  # 世界 alt — "Chinese and foreign"; wrong slot
    '五洲',  # 世界 alt — "five continents"; archaic
    # cycle 242: 频率 fixes "时间管理频率" → "时间管理效率"; cascade
    # protected via 轻重 _CILIN_SOURCE_BLACKLIST (idiom 轻重缓急) and
    # 生气 blacklist (精力 alt — primarily means "anger", off in vitality slot).
    '频率',  # 效率 alt — different meaning ("frequency")
    '生气',  # 精力 alt — primarily means "anger"; "时间和生气" 不通
    # cycle 243 longform blog audit additions:
    '上上',  # 可以 alt — colloquial "tops/very good"; modal mismatch ("上上在最大限度地" 不通)
    '分红',  # 分配 alt — financial "dividend"; "资源分红" 不通
    '不等',  # 不同 alt — "unequal" different concept; "不等的服务节点" 不通
    # cycle 244 longform blog audit additions:
    '放眼',  # 纵目/极目 alt — verb form; "在放眼全球" 不通 (放眼 needs object)
    '之际',  # 关键/关头 alt — archaic "at the moment of"; "之际介于我们" 不通
    '介于',  # 在于 alt — "between"; wrong meaning, "之际介于我们" wrong concept
    # cycle 245 long_blog seed=7 audit additions:
    '历历',  # 清晰 alt — archaic "vividly" (历历在目 idiom-only); "历历的产品构想" 不通
    # 创-family narrow alts (after cycle 234 blocked 创设, fallback to these still narrow):
    '创始',  # 创造 alt — "found/initiate"; narrow ("创始更具价值的产品" wrong)
    '创办',  # 创造 alt — "start (business)"; narrow
    '创立',  # 创造 alt — "establish"; narrow
    '创导',  # 创造 alt — formal "advocate"; rare
    # cycle 246 academic carbon trading audit additions:
    '企图',  # 作用 alt — "attempt/scheme"; "发挥重要企图" 不通
    # 手段 family — 一手/伎俩/心数/心眼/手眼 all colloquial/derogatory/idiom:
    '一手',  # 手段 alt — colloquial "trick/one hand"
    '伎俩',  # 手段 alt — negative "trick/scheme"
    '心数',  # 手段 alt — wrong slot ("mind/wits")
    '心眼',  # 手段 alt — wrong slot ("mind/heart")
    '手眼',  # 手段 alt — idiom-only ("手眼通天")
    # cycle 247 不-family + 反而 family scan:
    '是的',  # 不易/正确 alt — particle "yes"; wrong slot
    '倒转',  # 反而 alt — "reverse direction"; off-meaning
    '反是',  # 反而 alt — archaic
    # long_blog audit (post-d3dc2ea):
    # 检点 (alt of 上心/专注/只顾/在意/查点 etc) means "examine/scrutinize",
    # not "focus on" — "我常常只顾于" → "我常常检点于" misreads.
    '检点',
    # 使得 (alt of 驱动/有效/管用/可行/...) means "cause/make happen", not
    # "drive" — "数据驱动的决策" → "数据使得的决策" 不通.
    '使得',
    # 何等 (alt of 如何/什么/怎样/...) is exclamatory ("how/what a"), wrong
    # POS for interrogative slot — "如何攻克" → "何等攻克" 不通.
    '何等',
    # social/general 病句 audit (post-codex review):
    # 个人 cilin alts include 斯人 (literary "this person" — modern 不通);
    # 我们 cilin alts include 咱俩 (dual "the two of us" — wrong number),
    # 吾侪/吾辈/我辈 (all 文言 plurals), 俺们 (regional). Block them.
    '斯人',  # 个人 alt — 文言 "斯人" never modern Chinese
    '咱俩',  # 我们 alt — dual ("the two of us"), wrong for plural 我们
    '吾侪',  # 我们 alt — 文言
    '吾辈',  # 我们 alt — 文言
    '我辈',  # 我们 alt — 文言
    '俺们',  # 我们 alt — 北方方言, register mismatch in standard prose
    # 科学 cilin alts include 无误 ("error-free") — different concept;
    # 不易 family already source-blocked but 无误 leaks via 科学 path.
    '无误',  # 科学 alt — "error-free", different from "scientific/systematic"
    # 真正 cilin alts include 实际 ("actually" adv) — POS clash with attributive
    # use ("真正重要" → "实际重要" 不通). Other alts (一是一/委实/实在/实打实) keep.
    '实际',  # 真正 alt — POS clash in attributive contexts ("真正重要")
    # 毫无疑问 cilin alts include 大势所趋 ("inevitable trend") — totally
    # different meaning. 毫无疑问 = "no doubt"; 大势所趋 = "general/inevitable".
    '大势所趋',  # 毫无疑问 alt — meaning mismatch
    # general audit (post-d3dc2ea + e94e7b7):
    # 能够 cilin alts (亦可/可知) are both 文言. "能够 X" → "亦可 X" / "可知 X"
    # leaks 文言 register into general/academic prose. Block both alts; source
    # is also added to _CILIN_SOURCE_BLACKLIST below.
    '亦可',  # 能够 / 可知 alt — 文言, register slip in modern prose
    '可知',  # 能够 / 亦可 alt — 文言 ("可知道")
    # b4 hero candidate audit — substitutions appearing in dramatic-drop AI samples:
    # 可巧 (alt of 适时/及时/刚巧) means "happens to coincide", not "in a timely
    # manner" — "可巧调整" 不通.
    '可巧',
    # 接轨 (alt of 继续/延续/接续) means "integrate/connect with"; "接轨坚持"
    # / "接轨定投" misreads as integration.
    '接轨',
    # 报恩 (alt of 回报) means "repay kindness", not "return/feedback" —
    # "给我们最好的报恩" wrong concept.
    '报恩',
    # B-3 long_blog mutation audit: CiLin alternates that read broken in
    # modern product/blog prose ("在意于什么样攻克", "这一涉世",
    # "这一年不休是").
    '在意',
    '涉世',
    '不休',
    # "利用数据" is fine, but "利用频率" is a common false slot when replacing
    # 使用 inside product-metrics prose.
    '利用',
    # "顺着这个构思" is an off-slot replacement for 思路 in discourse markers.
    '构思',
    '小心',
    '不住',
    '笔触',
    '应用',
    '掌管',
    '关键',
    '可观',
    '保管',
    '才干',
    '什么样',
    '创制',
    '此即',
    '打算',
    '只顾',
    '调动',
    '在心',
    '何以',
    '什么',
    '当真',
    # heartbeat audit: 尤其 (alt of 更加) needs comparison context. "尤其充实"
    # / "迎来尤其充实和有意义的人生" (social hero) reads off; should be
    # intensifier "更加充实".
    '尤其',
    # 何如 (alt of 如何/什么/怎样) is 文言 "how about" — wrong slot for direct
    # interrogative; "何如解决" 不通.
    '何如',
    # 主体 (alt of 核心/主脑) means "main body/subject" not "essence" —
    # "产品决策的主体" misreads (long_blog audit).
    '主体',
    # 为重 (also alt of 核心) — adverbial "considered-important" doesn't fit
    # noun slot "决策的核心"; cilin fallback after 主体 still wrong.
    '为重',
    # 为主 (also alt of 核心) — adverbial "principally"; same slot mismatch.
    '为主',
    # 主从 (also alt of 核心) — relational "principal vs subordinate";
    # not a noun substitute for 核心 essence slot.
    '主从',
    # cycle 251: 一时 (alt of 时代/时期/时日/一代/一世) means "momentarily/for
    # a moment" — semantically opposite to era/period/lifetime. "技术快速
    # 推进、全球化深入推进的一时" (lf:80 audit, source had 时代) is broken;
    # should stay "时代". Multi-source mistake in cilin grouping.
    '一时',
}


# Source-side blacklist: 2-char cilin keys whose substitution produces
# broken Chinese — either because they're almost always part of longer
# compounds (substring-collision) or because their cilin alts shift
# meaning even in standalone position. Block at the source (skip these
# as replacement targets in reduce_cross_para_3gram_repeat).
#
# cycle 191: '不了' — X不了 negative-potential compound (受不了/少不了/
# 免不了/做不了…), 不了 → 不息/不停 breaks compound (少不息).
# cycle 192: empirical audit of 10 high-freq function words. Each line
# below = source word + the broken sample observed in test:
_CILIN_SOURCE_BLACKLIST = {
    '不了',  # 少不了 → 少不息
    '不是',  # 不是教师 → 纰缪教师 (alts: 不对/偏向/纰缪 — meaning shift)
    '一下',  # 想一下 → 想一瞬 (alts 一刹那/一瞬 too dramatic)
    '一些',  # 带一些礼物 → 带好几礼物 (好几 needs 个 measure word)
    '不要',  # 不要担心 → 并非担心 (并非 is statement-of-fact, not directive)
    '就是',  # 就是这样 → 即使这样 (即使 = "even if", needs main clause)
    '不能',  # 不能解决 → 未能解决 ("can't" → "didn't succeed", semantic shift)
    '什么',  # 什么东西 → 咋样东西 (咋样 colloquial + register-mismatch)
    '只是',  # 只是开始 → 单单开始 (单单 modifies things, not actions)
    # Idiom-anchor nouns: substituting these breaks fixed compounds even when
    # the alt is grammatically valid. "发展前景"→"发展未来" reads off.
    '前景',  # 发展前景 / 应用前景 / 推进前景 — idiom-fixed
    '前途',  # 发展前途 / 学术前途 — idiom-fixed
    # Adverbial-compound anchors: 方面 cilin alts (上头/上面/地方/方位/方向)
    # all break "多方面" / "各方面" idiomatic compounds.
    '方面',  # 多方面/各方面 → 多地方/各上面 — broken
    # Educational vocabulary anchors: 教学 cilin alts (上书/任课/执教/主讲)
    # are role-specific or archaic, all break the generic noun slot.
    '教学',  # 教育教学 → 教育上书 — archaic ("submit memorial")
    # Substring-collision anchors: 2-char keys frequently embedded in
    # larger compounds where substitution corrupts the parent. cycle 214
    # audit found "说不定" → "说不安" via 不定 → 不安 substring sub.
    '不定',  # inside 说不定 / 拿不定 / 一定不定; alts (不安/动乱) corrupt parent
    '末日',  # 后期/晚期/期末 alts shift "doomsday" → "later period"
    '后期',  # 后期/晚期/期末 cluster — same shift
    # cycle 235: 不容 cilin alts (不肯/回绝/拒绝/推却/推辞/闭门羹) all mean
    # "refuse/decline" — wrong for 不容忽视/不容置疑 (which means "doesn't
    # tolerate/permit"). Source blacklist since ALL alts are wrong-meaning.
    '不容',
    # cycle 242: 轻重 cilin alts (份量/份额/净重/分寸/分量/千粒重) all are
    # weight/measure terms; none fit the idiom 轻重缓急 (priority/important).
    # Source blacklist since substitution always breaks the idiom.
    '轻重',
    # cycle 242: 精力 cilin alts (元气/活力/生命力/生机/生气/肥力) all in
    # vitality family; "时间和精力" is set idiom, any substitution lands
    # on awkward "时间和生气/生机/活力". Block source.
    '精力',
    # cycle 243: 可以 cilin alts (上上/上佳/上好/不含糊/不离儿/不赖) all are
    # colloquial "good/fine"; none works as modal "可以" (can/may). Block source.
    '可以',
    # cycle 246: 温室 cilin alts (花房/保暖棚/大棚/暖房/暖棚/温棚) all are
    # physical greenhouse types; "温室气体" is fixed scientific term, ANY
    # substitution breaks it ("花房气体" → "flower-house gas" 错). Block source.
    '温室',
    # cycle 247: 不易 cilin alts (不利/不错/对头/得法/无误/是的) — 不利 is
    # OPPOSITE meaning, others wrong slot. Most already individually
    # blacklisted; source block to be exhaustive.
    '不易',
    # cycle 247: 不堪 alts (受不了/吃不住/吃不消/哪堪/架不住/禁不住) all
    # colloquial; 不堪 fixed term ("不堪重负/不堪入目/不堪一击") — substitution
    # breaks 4-char idioms.
    '不堪',
    # cycle 247: 进而 alts (一发/尤为/尤其/愈加/愈发/愈来愈) all degree adverbs;
    # 进而 is sequential connector ("furthermore/then"), different concept.
    '进而',
    # long_blog audit: substring collision. cilin '品蓝' → '藏蓝' fires inside
    # "产品蓝图" → "产藏蓝图" because regex matches '品蓝' substring across
    # word boundary (产品|蓝图). Block source — color noun never wanted in
    # AIGC humanize anyway.
    '品蓝',
    # general audit: 能够 cilin alts (亦可/可知) are both 文言. Source blacklist
    # since neither alt ever fits modern modal "能够 X" slot.
    '能够',
}


def _load_cilin():
    """Lazy-load filtered CiLin synonyms. Returns dict[word] -> list[candidate] or empty dict."""
    global _CILIN_CACHE
    if _CILIN_CACHE is not None:
        return _CILIN_CACHE
    if not os.path.exists(_CILIN_FILE):
        _CILIN_CACHE = {}
        return _CILIN_CACHE
    try:
        with open(_CILIN_FILE, 'r', encoding='utf-8') as f:
            _CILIN_CACHE = json.load(f)
    except (json.JSONDecodeError, OSError):
        _CILIN_CACHE = {}
    return _CILIN_CACHE


def expand_with_cilin(word, candidates, scene='general'):
    """Expand a candidate list with CiLin synonyms (filtered through blacklists).

    Only used when enabled via --cilin CLI flag. CiLin has ~40K words vs the
    hand-curated ~200 in WORD_SYNONYMS, so expansion gives much more variety —
    but CiLin's "synonym" relation is loose (taxonomic, not strictly substitutable)
    and contains archaic/idiomatic candidates. Always filter through scene blacklist.
    """
    cilin = _load_cilin()
    extras = cilin.get(word, [])
    if not extras:
        return candidates
    existing = set(candidates)
    filtered = []
    for c in extras:
        if c in existing:
            continue
        if c in _AI_PATTERN_BLACKLIST:
            continue
        if c in _CILIN_BLACKLIST:
            continue  # semantic/POS/register mismatch, curated
        if scene == 'academic' and c in ACADEMIC_BLACKLIST_CANDIDATES:
            continue
        if scene == 'novel' and c in NOVEL_BLACKLIST_CANDIDATES:
            continue
        filtered.append(c)
        existing.add(c)
    return list(candidates) + filtered


# ═══════════════════════════════════════════════════════════════════
#  Strategy 3: Noise expression injection — expression table
# ═══════════════════════════════════════════════════════════════════

NOISE_EXPRESSIONS = {
    'hedging': ['说实话', '坦白讲', '客观地说', '实事求是地讲', '平心而论',
                '老实说', '不夸张地说', '公正地看'],
    'self_correction': ['或者说', '准确地讲', '换个角度看', '严格来说',
                        '更确切地说', '往深了讲', '细想一下'],
    # cycle 183 dropped '或许' — in detect_cn HEDGING_PHRASES, injection
    # increases hedging_language count, self-defeat (cycle 77 family).
    'uncertainty': ['大概', '差不多', '似乎', '多少有些',
                    '约莫', '估摸着', '八成'],
    # Cycle 77: dropped '换句话说' — it is in detect_cn's ai_high_freq_words
    # pattern, so injecting it raises the AI score (self-defeating).
    # cycle 208: trimmed — '话说回来'/'反过来看'/'说到这里'/'回过头看' all
    # narrative-voice openers that read as off-register in essay/factual text.
    # Kept '再往下想'/'顺着这个思路' which work in analytical contexts.
    'transition_casual': ['再往下想', '顺着这个思路'],
    # cycle 195: trimmed 8 → 3 — removed register-mismatched fillers
    # (怎么说呢/不瞒你说/你别说/讲真/这么说吧) that read very colloquial /
    # internet-slangy. They land in formal/business/academic text and
    # break fluency. Kept '其实/说到底/当然了' which fit most registers.
    'filler': ['当然了', '其实', '说到底'],
    # Cycle 55: dropped 5 entries that appear 0 times in 2.5M chars of
    # human Chinese (news + novel corpora) — '依我之见 / 以我的经验 /
    # 在我的理解里 / 就我所知 / 我个人倾向于'. These read as AI-style
    # stilted hedges in any register (academic / general / social), not
    # just academic. '我觉得' and '在我看来' kept (105 + 4 hits in human
    # corpus, idiomatic).
    'personal': ['我觉得', '在我看来'],
}

# Academic-safe categories (no oral fillers or personal opinions)
NOISE_ACADEMIC_CATEGORIES = ['hedging', 'self_correction', 'uncertainty']
# Academic-specific hedging (more formal)
NOISE_ACADEMIC_EXPRESSIONS = {
    # cycle 157: pool expanded from 4 → 7 each. Cycle 154 bn=10 academic
    # dropped from +15 (with casual-filler injection) to +10.5 (with this
    # formal-only pool). More formal candidates give random.choice more
    # variety, raising the chance of hitting LR-favorable phrasing.
    'hedging': ['客观地说', '实事求是地讲', '平心而论', '公正地看',
                '从客观角度看', '理性而言', '客观看待'],
    'self_correction': ['准确地讲', '严格来说', '更确切地说', '往深了讲',
                        '细究而论', '准确而言', '严谨地说'],
    # Cycle 77: dropped '在一定程度上' from this academic uncertainty pool too
    # (sister fix to cycle 76 in academic_cn). It is in detect_cn's hedging_
    # language and ai_high_freq_words patterns; injecting it raises the AI
    # score. Pool 5→4.
    # cycle 183 dropped '或许' from academic uncertainty too — sister
    # fix to general pool. Same detect_cn HEDGING_PHRASES self-defeat.
    'uncertainty': ['大致', '似乎', '多少',
                    '大体', '约莫', '大体上'],
}

def _load_bigram_freq():
    """Load bigram frequencies from the n-gram frequency table."""
    try:
        from ngram_model import _load_freq
    except ImportError:
        try:
            from scripts.ngram_model import _load_freq
        except ImportError:
            return {}
    freq = _load_freq()
    return freq.get('bigrams', {})


def reduce_high_freq_bigrams(text, strength=0.3, scene='general'):
    """
    策略1: 扫描文本中的高频 bigram，尝试用低频同义替换降低可预测性。
    strength: 0-1，控制替换比例。
    scene: 'general' / 'academic' / 'social' —
      - academic: 跳过 ACADEMIC_PRESERVE_WORDS，候选过 ACADEMIC_BLACKLIST_CANDIDATES

    使用基于词的替换（非位置），避免长度变化导致的错位问题。
    """
    bigram_freq = _load_bigram_freq()
    if not bigram_freq:
        return _simple_synonym_pass(text, strength, scene=scene)

    chars = re.findall(r'[\u4e00-\u9fff]', text)
    if len(chars) < 4:
        return text

    preserve = ACADEMIC_PRESERVE_WORDS if scene == 'academic' else set()

    # Step 1: Score each WORD_SYNONYMS word by its surrounding bigram frequency
    word_scores = []  # (word, total_bigram_freq, count_in_text)
    for word in WORD_SYNONYMS:
        if word in preserve:
            continue
        count = text.count(word)
        if count == 0:
            continue
        # Compute bigram frequency of this word's characters
        word_chars = re.findall(r'[\u4e00-\u9fff]', word)
        total_freq = 0
        for i in range(len(word_chars) - 1):
            bg = word_chars[i] + word_chars[i + 1]
            total_freq += bigram_freq.get(bg, 0)
        word_scores.append((word, total_freq, count))

    if not word_scores:
        return text

    # Step 2: Sort by bigram frequency (highest first)
    word_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Replace top N unique words (controlled by strength)
    n_replace = max(1, int(len(word_scores) * strength))
    replaced_words = set()

    for word, freq_score, count in word_scores[:n_replace]:
        if word in replaced_words:
            continue

        candidates = _filter_candidates_for_scene(word, WORD_SYNONYMS[word], scene)
        if _USE_CILIN:
            candidates = expand_with_cilin(word, candidates, scene)

        # Rank candidates by bigram frequency ascending (rarest first)
        ranked = []
        for candidate in candidates:
            cand_chars = re.findall(r'[\u4e00-\u9fff]', candidate)
            if not cand_chars:
                continue
            total_f = 0
            for i in range(len(cand_chars) - 1):
                total_f += bigram_freq.get(cand_chars[i] + cand_chars[i + 1], 0)
            ranked.append((candidate, total_f))
        if not ranked:
            continue
        ranked.sort(key=lambda x: x[1])

        # Pick strategy: NOT the rarest (too weird, e.g. 施用/拉高/本事),
        # but moderately rare — lower third by bigram frequency when possible.
        n_cand = len(ranked)
        if n_cand == 1:
            primary = ranked[0][0]
        elif n_cand == 2:
            primary = ranked[0][0]
        else:
            idx = min(max(1, n_cand // 3), n_cand - 2)
            primary = ranked[idx][0]

        # Partial replacement: don't replace EVERY occurrence of the word.
        # Replacing all creates NEW AI-pattern repetition (e.g. "系统"×6 → "架构"×6).
        # Keep some original occurrences + mix in alternative candidates for variation.
        SENTINEL = '\x00'

        def _protect(w):
            return SENTINEL.join(w) if len(w) > 1 else w

        occurrences = [m.start() for m in re.finditer(re.escape(word), text)]
        if not occurrences:
            continue
        # Replace ~60% of occurrences (min 1, always at least the first)
        n_replace_occ = max(1, int(len(occurrences) * 0.6))
        # Randomly select which occurrences to replace (deterministic via current seed)
        to_replace = set(random.sample(range(len(occurrences)), n_replace_occ))

        # Pick alternative candidates for variety when multiple occurrences replaced
        # (avoid monotone repetition of single replacement)
        alt_candidates = [c for c, _ in ranked if c != primary] or [primary]

        # Capture original text for next-char lookups (text mutates inside loop)
        original_text = text
        ranked_alts = [c for c, _ in ranked]

        def _pick_safe(default, next_ch):
            """Avoid alts whose last char equals next_ch (would double).
            Falls back to default if no safe alt exists."""
            if not next_ch or default[-1:] != next_ch:
                return default
            for cand in ranked_alts:
                if cand and cand[-1] != next_ch:
                    return cand
            return default

        # Rebuild text by iterating occurrences back-to-front (avoid shifting positions)
        for k in reversed(range(len(occurrences))):
            pos = occurrences[k]
            if k not in to_replace:
                continue
            # Word-boundary doubling guard: check next char in source after the
            # word being replaced. If alt ends with that char, swap to a
            # non-doubling alt. Catches '能够以X' → '可以以X' / '系统的研究'
            # → '架构的的' family of bugs without removing the entry entirely.
            next_ch = original_text[pos + len(word):pos + len(word) + 1]
            # Cycle 54: left-context cross-boundary guard. '解决' inside
            # '了解决策' actually spans 了解|决策 (two distinct words);
            # replacing 解决 with 攻克 corrupts to '了攻克策'. Skip when
            # the word's leading char + prev char form a known 2-char word
            # AND the word's trailing char + next char also form a 2-char
            # word — that's the cross-boundary signature.
            prev_ch = original_text[pos - 1:pos] if pos > 0 else ''
            if word == '解决' and prev_ch == '了' and next_ch in '策心议定断':
                continue
            if word == '解决' and next_ch == '方':
                continue
            if word == '研究' and prev_ch == '本':
                continue
            # Compound-noun guard: '发展' acts as N1 in 'X的发展前景/态势/...'
            # — substituting to verb-form alts (推进/进展/推动) breaks the
            # NP. Skip when followed by a known compound noun suffix.
            if word == '发展':
                next_two = original_text[pos + len(word):pos + len(word) + 2]
                _np_suffixes = (
                    '前景', '前途', '态势', '趋势', '历程', '规律',
                    '方向', '格局', '局面', '动力', '空间', '潜力',
                    '阶段', '路径', '路线', '方式', '模式',
                )
                if next_two in _np_suffixes:
                    continue
                # Same-sentence repetition guard: '推动X长效发展' → '推动X长效推进'
                # gives 推动+推进 redundancy. Skip if 推 appears in prior 6
                # chars within same sentence.
                left_ctx = original_text[max(0, pos - 6):pos]
                if '推' in left_ctx and not any(c in '。！？' for c in left_ctx):
                    continue
            # 分析 in noun-modifier slot: '分析师' / '分析员' should not
            # become '解读师' / '剖析员' (not real words).
            if word == '分析':
                next_ch = original_text[pos + len(word):pos + len(word) + 1]
                if next_ch in '师员家者':
                    continue
            # Pick primary for first replaced occurrence, alternate for others
            if k == min(to_replace):
                replacement = _pick_safe(primary, next_ch)
            else:
                pick = random.choice([primary] + alt_candidates)
                replacement = _pick_safe(pick, next_ch)
            protected = _protect(replacement)
            text = text[:pos] + protected + text[pos + len(word):]

        replaced_words.add(word)

        # Also mark synonyms of the same word to avoid replacing the replacement
        for syn in candidates:
            if syn != primary and syn in WORD_SYNONYMS:
                replaced_words.add(syn)

    # Strip sentinels
    text = text.replace('\x00', '')

    return text


def _simple_synonym_pass(text, strength=0.3, scene='general'):
    """Fallback: replace a fraction of WORD_SYNONYMS matches randomly.

    scene: 'academic' filters PRESERVE words and BLACKLIST candidates.
    """
    preserve = ACADEMIC_PRESERVE_WORDS if scene == 'academic' else set()
    found = []
    for word in WORD_SYNONYMS:
        if word in preserve:
            continue
        start = 0
        while True:
            pos = text.find(word, start)
            if pos < 0:
                break
            found.append((word, pos))
            start = pos + len(word)
    if not found:
        return text
    n_replace = max(1, int(len(found) * strength))
    random.shuffle(found)
    replaced_positions = set()
    for word, pos in found[:n_replace]:
        if any(p in replaced_positions for p in range(pos, pos + len(word))):
            continue
        candidates = _filter_candidates_for_scene(word, WORD_SYNONYMS[word], scene)
        if not candidates:
            continue
        candidate = random.choice(candidates)
        text = text[:pos] + candidate + text[pos + len(word):]
        for p in range(pos, pos + len(candidate)):
            replaced_positions.add(p)
    return text


# ═══════════════════════════════════════════════════════════════════
#  Strategy 2: Sentence length randomization
# ═══════════════════════════════════════════════════════════════════

_PARA_BOOST_ATTRIBUTION = (
    '指出', '表明', '认为', '揭示', '发现', '显示', '提出',
    '说', '称', '讲', '强调', '主张', '断言',
)
_PARA_BOOST_SUBORDINATE = (
    '随着', '鉴于', '为了', '由于', '尽管', '虽然',
    '如果', '假如', '若是', '倘若', '要是', '即便', '纵然',
    '除了', '除非', '只要', '只有', '无论', '不管',
    '当', '每当', '一旦',
)
_PARA_BOOST_BARE_CONTINUATOR = (
    '使得', '使', '导致', '引起', '造成', '致使',
)


def _boost_one_paragraph_cv(para, target_cv):
    """Truncate the longest sentence at first comma if paragraph-internal
    sentence-length CV is below target. Reuses guards from
    randomize_sentence_lengths Strategy B."""
    cn_count = len(re.findall(r'[一-鿿]', para))
    if cn_count < 60:
        return para

    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            pairs.append([s, p])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])

    if len(pairs) < 3:
        return para

    lens = [len(re.findall(r'[一-鿿]', s)) for s, _ in pairs]
    valid = [(i, l) for i, l in enumerate(lens) if l >= 5]
    if len(valid) < 3:
        return para
    vl = [l for _, l in valid]
    m = sum(vl) / len(vl)
    if m == 0:
        return para
    var = sum((l - m) ** 2 for l in vl) / len(vl)
    cv = (var ** 0.5) / m

    if cv >= target_cv:
        return para

    long_idx = max(range(len(pairs)), key=lambda i: lens[i])
    long_s, long_p = pairs[long_idx]
    if lens[long_idx] < 18:
        return para

    comma_pos = long_s.find('，')
    if comma_pos < 0:
        return para
    first_part = long_s[:comma_pos]
    rest_part = long_s[comma_pos + 1:]
    if (len(re.findall(r'[一-鿿]', first_part)) < 8 or
            len(re.findall(r'[一-鿿]', rest_part)) < 8):
        return para

    first_stripped = first_part.lstrip()
    last_nl = first_part.rfind('\n')
    if last_nl >= 0:
        tail_cn = len(re.findall(r'[一-鿿]',
                                 first_part[last_nl + 1:]))
        if tail_cn < 3:
            return para

    if first_part.endswith(_PARA_BOOST_ATTRIBUTION):
        return para
    if first_stripped.startswith(_PARA_BOOST_SUBORDINATE):
        return para
    if rest_part.lstrip().startswith(_PARA_BOOST_BARE_CONTINUATOR):
        return para

    pairs[long_idx] = [first_part, '。']
    pairs.insert(long_idx + 1, [rest_part, long_p or '。'])
    return ''.join(s + p for s, p in pairs)


_PARA_BOOST_REACTIONS = (
    '的确', '确实如此', '颇有道理', '不无道理',
    '有一定道理', '各有道理', '各有说法', '值得深思',
)


def _boost_one_para_via_merge(para, target_cv):
    """Merge a single pair of adjacent short-medium sentences with a comma
    to lift a uniform paragraph's internal sentence-length CV. Reuses the
    Strategy-A merge guards from randomize_sentence_lengths (reactions,
    paragraph-break boundary, total length cap)."""
    cn_count = len(re.findall(r'[一-鿿]', para))
    if cn_count < 60:
        return para

    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            pairs.append([s, p])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])

    if len(pairs) < 4:
        return para

    lens = [len(re.findall(r'[一-鿿]', s)) for s, _ in pairs]
    valid = [l for l in lens if l >= 5]
    if len(valid) < 3:
        return para
    m = sum(valid) / len(valid)
    if m == 0:
        return para
    var = sum((l - m) ** 2 for l in valid) / len(valid)
    cv = (var ** 0.5) / m

    if cv >= target_cv:
        return para

    # Find an adjacent pair both 5..25 chars whose merged length is <=60
    # (so we cross the medium→long boundary and lift CV without making
    # the merged sentence unwieldy).
    for i in range(len(pairs) - 1):
        l1, l2 = lens[i], lens[i + 1]
        if not (5 <= l1 <= 25 and 5 <= l2 <= 25):
            continue
        if l1 + l2 > 60:
            continue
        s1, _ = pairs[i]
        s2, p2 = pairs[i + 1]
        if (s1.strip() in _PARA_BOOST_REACTIONS or
                s2.strip() in _PARA_BOOST_REACTIONS):
            continue
        if '\n' in s2:
            continue
        merged = s1.rstrip() + '，' + s2.lstrip()
        pairs[i] = [merged, p2]
        pairs.pop(i + 1)
        break

    return ''.join(s + p for s, p in pairs)


def reduce_cross_para_3gram_repeat(text, max_replacements=4, scene='general',
                                   style=None, seed=None):
    """v5 P1.3 humanize counter-measure for cross_para_3gram_repeat
    (LR coef +2.24 on longform).

    Walks paragraphs, identifies 2-char words (CiLin keys) that appear
    in 2+ paragraphs, and replaces ONE occurrence in a later paragraph
    with a CiLin synonym. Replacing a 2-char word breaks two
    overlapping 3-grams, so even a handful of substitutions measurably
    drops the cross-paragraph trigram repetition ratio.

    Scene-aware via the same blacklists as expand_with_cilin
    (_AI_PATTERN_BLACKLIST / _CILIN_BLACKLIST / ACADEMIC_BLACKLIST_CANDIDATES
    / NOVEL_BLACKLIST_CANDIDATES). Skips when the scene/style filters
    yield no usable synonym.

    Prefers words in exactly 2 paragraphs (each replacement directly
    drops a repeat — words spanning 3+ paragraphs need more sub work
    to clear).
    """
    if seed is not None:
        random.seed(seed)

    cilin = _load_cilin()
    if not cilin:
        return text

    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return text

    cilin_keys = set(cilin.keys()) - _CILIN_SOURCE_BLACKLIST
    para_words = []
    for p in paragraphs:
        chars = re.findall(r'[一-鿿]', p)
        words_in_p = set()
        for i in range(len(chars) - 1):
            w = chars[i] + chars[i + 1]
            if w in cilin_keys:
                words_in_p.add(w)
        para_words.append(words_in_p)

    word_paras = {}
    for i, words in enumerate(para_words):
        for w in words:
            word_paras.setdefault(w, []).append(i)

    candidates = [(w, ps) for w, ps in word_paras.items() if len(ps) >= 2]
    if not candidates:
        return text

    # Prefer words appearing in fewer paragraphs (each replacement
    # there directly clears the repeat). Then random within tier.
    candidates.sort(key=lambda x: len(x[1]))
    # Shuffle within each tier of equal paragraph-count
    tier_buckets = {}
    for w, ps in candidates:
        tier_buckets.setdefault(len(ps), []).append((w, ps))
    for k in tier_buckets:
        random.shuffle(tier_buckets[k])
    ordered = []
    for k in sorted(tier_buckets):
        ordered.extend(tier_buckets[k])

    new_paragraphs = list(paragraphs)
    replaced = 0

    for word, para_indices in ordered:
        if replaced >= max_replacements:
            break
        synonyms = cilin.get(word, [])
        if not synonyms:
            continue
        filtered = []
        for c in synonyms:
            if c in _AI_PATTERN_BLACKLIST:
                continue
            if c in _CILIN_BLACKLIST:
                continue
            if scene == 'academic' and c in ACADEMIC_BLACKLIST_CANDIDATES:
                continue
            if (scene == 'novel' or style == 'novel') and \
                    c in NOVEL_BLACKLIST_CANDIDATES:
                continue
            filtered.append(c)
        if not filtered:
            continue
        synonym = random.choice(filtered)
        # Replace in the LAST occurrence paragraph (so the established
        # term lands in earlier paragraphs and the variation shows up
        # later — closer to how humans drift).
        last_idx = para_indices[-1]
        new_para = new_paragraphs[last_idx].replace(word, synonym, 1)
        if new_para != new_paragraphs[last_idx]:
            new_paragraphs[last_idx] = new_para
            replaced += 1

    return join_paragraphs(new_paragraphs)


_LONGFORM_PARA_HEAD_MARKERS = (
    '首先', '其次', '再次', '最后', '然后', '接下来', '与此同时',
    '此外', '另外', '除此之外', '具体而言', '具体来说', '具体地说',
    '一方面', '另一方面', '总的来说', '总而言之', '综上所述',
    '因此', '所以', '由此', '进而', '从而', '基于此',
    '然而', '不过', '事实上', '实际上',
)

_LONGFORM_STARTER_MARKERS = (
    '同时', '此外', '另外', '因此', '所以', '然而', '不过',
    '事实上', '实际上', '具体来说', '具体而言', '总的来说',
    '换言之', '简而言之', '需要注意的是', '值得注意的是',
)


def _strip_leading_marker_once(fragment, markers):
    s = fragment.lstrip()
    prefix = fragment[:len(fragment) - len(s)]
    for marker in sorted(markers, key=len, reverse=True):
        if s.startswith(marker):
            rest = s[len(marker):]
            if rest.startswith(('，', ',', '、', '：', ':')):
                rest = rest[1:]
            if len(re.findall(r'[一-鿿]', rest)) >= 12:
                return prefix + rest.lstrip()
    return fragment


def _longform_discourse_marker_diversity(text, seed=None):
    """Remove repeated paragraph-head discourse markers on long candidates."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return text

    seen = set()
    changed = 0
    result = []
    for p in paragraphs:
        stripped = p.lstrip()
        marker = None
        for m in sorted(_LONGFORM_PARA_HEAD_MARKERS, key=len, reverse=True):
            if stripped.startswith(m):
                marker = m
                break
        if marker and marker in seen and changed < 3 and random.random() < 0.8:
            new_p = _strip_leading_marker_once(p, (marker,))
            if new_p != p and new_p.strip():
                p = new_p
                changed += 1
        if marker:
            seen.add(marker)
        result.append(p)

    return join_paragraphs(result)


def _longform_merge_one_sentence_pair(para):
    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip():
            pairs.append([parts[i], parts[i + 1]])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])
    if len(pairs) < 3:
        return para

    for i in range(len(pairs) - 1):
        s1, _ = pairs[i]
        s2, p2 = pairs[i + 1]
        l1 = len(re.findall(r'[一-鿿]', s1))
        l2 = len(re.findall(r'[一-鿿]', s2))
        if not (8 <= l1 <= 28 and 8 <= l2 <= 32 and l1 + l2 <= 62):
            continue
        if s2.lstrip().startswith(_PARA_BOOST_BARE_CONTINUATOR):
            continue
        pairs[i] = [s1.rstrip() + '，' + s2.lstrip(), p2]
        pairs.pop(i + 1)
        return ''.join(s + p for s, p in pairs)
    return para


def _longform_split_one_comma_clause(para):
    parts = re.split(r'([。！？])', para)
    pairs = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip():
            pairs.append([parts[i], parts[i + 1]])
    if len(parts) % 2 == 1 and parts[-1].strip():
        pairs.append([parts[-1], ''])
    if len(pairs) < 2:
        return para

    for i, (sent, punct) in enumerate(pairs):
        if len(re.findall(r'[一-鿿]', sent)) < 34:
            continue
        for m in re.finditer(r'[，,]', sent):
            left = sent[:m.start()]
            right = sent[m.end():]
            if (len(re.findall(r'[一-鿿]', left)) >= 12 and
                    len(re.findall(r'[一-鿿]', right)) >= 14 and
                    not right.lstrip().startswith(_PARA_BOOST_BARE_CONTINUATOR)):
                pairs[i] = [left.rstrip() + '。' + right.lstrip(), punct]
                return ''.join(s + p for s, p in pairs)
    return para


def _longform_paragraph_punct_drift(text, seed=None):
    """Create mild paragraph-to-paragraph punctuation rhythm drift."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return text

    result = []
    changed = 0
    start = random.randrange(2)
    for idx, p in enumerate(paragraphs):
        new_p = p
        if changed < 3 and len(re.findall(r'[一-鿿]', p)) >= 70:
            if (idx + start) % 2 == 0:
                new_p = _longform_split_one_comma_clause(p)
            else:
                new_p = _longform_merge_one_sentence_pair(p)
            if new_p != p and new_p.strip():
                changed += 1
        result.append(new_p)
    return join_paragraphs(result)


def _longform_paragraph_length_cv_micro_adjust(text, seed=None):
    """Single guarded merge/split pass when paragraph lengths are too uniform."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 5:
        return text
    cv = _para_cv(paragraphs)
    if cv is not None and cv >= 0.48:
        return text
    adjusted = vary_paragraph_rhythm(text)
    if len(split_paragraphs(adjusted)) < len(paragraphs) - 1:
        return text
    return adjusted


def _longform_starter_entropy_boost(text, seed=None):
    """Reduce repeated safe transition starters without inventing new wording."""
    if seed is not None:
        random.seed(seed)
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return text

    starter_counts = {}
    for sent in re.split(r'[。！？!?；;\n]+', text):
        chars = re.findall(r'[一-鿿]', sent)
        if len(chars) >= 2:
            key = ''.join(chars[:2])
            starter_counts[key] = starter_counts.get(key, 0) + 1
    repeated = {k for k, v in starter_counts.items() if v >= 2}
    if not repeated:
        return text

    changed = 0

    def strip_sentence(m):
        nonlocal changed
        boundary, body = m.group(1), m.group(2)
        chars = re.findall(r'[一-鿿]', body)
        key = ''.join(chars[:2]) if len(chars) >= 2 else ''
        if key not in repeated or changed >= 3 or random.random() >= 0.7:
            return m.group(0)
        new_body = _strip_leading_marker_once(body, _LONGFORM_STARTER_MARKERS)
        if new_body != body and new_body.strip():
            changed += 1
            return boundary + new_body
        return m.group(0)

    result = []
    pattern = re.compile(r'(^|[。！？!?；;\n])([^。！？!?；;\n]+)')
    for p in paragraphs:
        result.append(pattern.sub(strip_sentence, p))
    return join_paragraphs(result)


def _apply_longform_mutation_profile(text, mutation_seed=None, scene='general',
                                     style=None):
    """Candidate-only longform mutations for best-of-n exploration."""
    before_paras = len(split_paragraphs(text))
    if before_paras < 3:
        return text

    try:
        from ngram_model import compute_lr_score
    except ImportError:
        try:
            from scripts.ngram_model import compute_lr_score
        except ImportError:
            compute_lr_score = None

    def lr_score(candidate):
        if compute_lr_score is None:
            return None
        lr = compute_lr_score(candidate, scene='longform')
        return lr['score'] if lr else None

    def structurally_safe(candidate):
        after_paras = split_paragraphs(candidate)
        if any(not p.strip() for p in after_paras):
            return False
        return len(after_paras) >= before_paras - 2

    current = text
    current_score = lr_score(current)
    steps = (
        lambda t: _longform_discourse_marker_diversity(
            t, seed=None if mutation_seed is None else mutation_seed + 23),
        lambda t: _longform_paragraph_punct_drift(
            t, seed=None if mutation_seed is None else mutation_seed + 37),
        lambda t: _longform_paragraph_length_cv_micro_adjust(
            t, seed=None if mutation_seed is None else mutation_seed + 41),
        lambda t: _longform_starter_entropy_boost(
            t, seed=None if mutation_seed is None else mutation_seed + 53),
    )

    for step in steps:
        candidate = step(current)
        if candidate == current or not structurally_safe(candidate):
            continue
        candidate_score = lr_score(candidate)
        if (current_score is not None and candidate_score is not None and
                candidate_score > current_score):
            continue
        current = candidate
        if candidate_score is not None:
            current_score = candidate_score

    return current if structurally_safe(current) else text


_PARA_INTERJECTION_NEUTRAL = (
    # cycle 195: trimmed 8 → 3 — removed 5 academic-only interjections
    # (此点尚需 / 此种情形 / 相关因素 / 若进一步 / 仔细推敲) that read
    # contemplative-academic when injected mid-text in informational /
    # workplace / general samples. Kept 3 entries that fit informational
    # registers (common-saying or "另一种角度" framing). Loses some pool
    # variety; bn=10 still has 3 distinct picks per pass.
    '事情可能并不如表面所示那般简单，需要更细致地审视。',
    '若从更多角度去考虑，结论恐怕会有不少不同之处。',
    '换个角度去看也成立，问题的另一面同样不容忽视。',
)


# Narrative-voice variants for novel style — character-internal / group
# beats only. Setting-specific lines (time-of-day, indoor / outdoor,
# weather) are deliberately excluded so the inserted paragraph doesn't
# contradict the surrounding scene state. Each is >=20 cn chars to pass
# the >=20 paragraph filter used by compute_paragraph_length_cv.
_PARA_INTERJECTION_NOVEL = (
    '众人都不约而同地陷入了一阵短暂的压抑沉默。',
    '他抬起头来，目光缓缓扫过众人脸上的神色一遍。',
    '他转过头去，目光在某处停留了片刻又缓缓移开。',
    '时间仿佛在这一刻悄然凝固住了，没有人开口说话。',
    '他心中暗暗思量了一阵子，事情似乎并不那么简单。',
    '气氛变得有些紧张了起来，众人之间默然不语好一会。',
    '他皱了皱眉头，似乎在心里反复斟酌着什么内容不解。',
    '他眯起了眼，神色之中流露出一种难以言喻的情绪。',
)


def insert_short_interjection_paragraph(text, target_cv=0.50, style=None,
                                        seed=None):
    """v5 P1.2 humanize counter-measure for paragraph_length_cv (d=-1.49).

    For multi-paragraph text whose paragraph-length CV is below target,
    insert a single short interjection paragraph (~20-22 cn chars) AFTER
    one of the longer existing paragraphs (top quartile by length).
    The interjection sharply lifts paragraph-length variance without
    restructuring existing paragraphs (cycle 28 lesson: split/merge of
    existing paragraphs has persistently negative ROI; this function
    only adds, never restructures).

    Two pools, picked by style:
      - novel  : narrative beats (atmosphere / action / dialogue gap)
      - other  : reflective neutral-formal sentences

    Skips:
      - Single-paragraph text
      - Text already varied (CV >= target)
      - When adjacent paragraph is a markdown header / list / bold
        subheader (would split a structural pair)
    """
    if seed is not None:
        random.seed(seed)

    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 4:
        return text

    lens = [len(re.findall(r'[一-鿿]', p)) for p in paragraphs]
    valid_pairs = [(i, l) for i, l in enumerate(lens) if l >= 20]
    if len(valid_pairs) < 3:
        return text
    valid_lens = [l for _, l in valid_pairs]
    m = sum(valid_lens) / len(valid_lens)
    if m == 0:
        return text
    var = sum((l - m) ** 2 for l in valid_lens) / len(valid_lens)
    cv = (var ** 0.5) / m

    if cv >= target_cv:
        return text

    sorted_pairs = sorted(valid_pairs, key=lambda x: -x[1])
    top_count = max(2, len(valid_pairs) // 4)
    top_indices = [i for i, _ in sorted_pairs[:top_count]]

    insert_after = random.choice(top_indices)
    next_idx = insert_after + 1
    if next_idx < len(paragraphs):
        next_lstrip = paragraphs[next_idx].lstrip()
        if (next_lstrip.startswith('#') or next_lstrip.startswith('- ') or
                next_lstrip.startswith('* ') or
                (next_lstrip.startswith('**') and
                 next_lstrip.rstrip().endswith('**'))):
            return text

    pool = _PARA_INTERJECTION_NOVEL if style == 'novel' \
        else _PARA_INTERJECTION_NEUTRAL
    interjection = random.choice(pool)

    new_paragraphs = list(paragraphs)
    new_paragraphs.insert(next_idx, interjection)
    return join_paragraphs(new_paragraphs)


def boost_para_cv_via_merge(text, target_cv=0.40):
    """v5 P1 humanize counter-measure (merge variant).

    Walks paragraphs and, for any whose internal sentence-length CV is
    below target, merges a single pair of adjacent short-medium
    sentences with a comma. This removes one period (counter to the
    truncation variant in boost_para_sent_len_cv that adds one) so
    the punct_density LR contribution doesn't cancel the para-CV
    contribution, and the merged sentence typically clears the
    medium→long threshold (sent_len_long_frac coef in the longform LR
    is -0.44, so producing more longs helps).
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        return text
    return join_paragraphs(_boost_one_para_via_merge(p, target_cv)
                           for p in paragraphs)


def boost_para_sent_len_cv(text, target_cv=0.40):
    """v5 P1 humanize counter-measure for stat_low_para_sent_len_cv (d=-2.08).

    For each paragraph (>=60 cn chars, >=3 sentences) where internal
    sentence-length CV is below target, truncate the longest sentence
    at its first comma so the paragraph contains at least one short
    sentence among its mediums. Single pass — does not iterate.

    Skips short paragraphs and applies the same guards as
    randomize_sentence_lengths Strategy B (attribution verbs, subordinate
    clause heads, bare causative continuators, paragraph-break tail).
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        # Single-paragraph text — signal doesn't apply.
        return text
    return join_paragraphs(_boost_one_paragraph_cv(p, target_cv)
                           for p in paragraphs)


def randomize_sentence_lengths(text, aggressive=False, seed=None):
    """
    策略2: 刻意制造不均匀的句子长度分布。
    - 随机选 20% 的短句保持极短
    - 随机选 10% 的句子通过合并拉长
    - 制造"短-长-短-长-特长-短"的节奏
    """
    if seed is not None:
        random.seed(seed)

    # Split into sentences preserving punctuation
    parts = re.split(r'([。！？])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            sentences.append((s, p))
    # Handle trailing text
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append((parts[-1], ''))

    if len(sentences) < 4:
        return text

    merge_rate = 0.15 if not aggressive else 0.25
    truncate_rate = 0.15 if not aggressive else 0.25

    result = []
    i = 0
    while i < len(sentences):
        s, p = sentences[i]
        cn_len = len(re.findall(r'[\u4e00-\u9fff]', s))

        # Strategy A: merge short adjacent sentences into a long one
        if (i + 1 < len(sentences) and random.random() < merge_rate):
            s2, p2 = sentences[i + 1]
            cn_len2 = len(re.findall(r'[\u4e00-\u9fff]', s2))
            # Don't merge if adjacent sentence is a known reaction phrase (cycle 22
            # bug fix — short reactions inserted by `insert_short_reactions` were
            # being silently merged back, collapsing the short_frac signal).
            _reactions = (
                '的确', '确实如此', '颇有道理', '不无道理', '事出有因',
                '耐人寻味', '值得深思', '让人深思', '可见一斑', '有一定道理',
                '各有道理', '各有说法', '难以一概', '难以断言', '说来话长',
                '一言难尽',
            )
            s_stripped = s.strip()
            s2_stripped = s2.strip()
            # Paragraph boundary: split by [。！？] preserves \n\n as leading
            # whitespace on the next sentence. Merging would .lstrip() the
            # \n\n away and collapse two paragraphs into one — discourse
            # structure loss (Petalses issue #5).
            if '\n' in s2 or s_stripped in _reactions or s2_stripped in _reactions:
                pass
            elif cn_len + cn_len2 < 100:
                merged = s.rstrip() + '，' + s2.lstrip()
                result.append(merged + p2)
                i += 2
                continue

        # Strategy B: truncate longer sentences to their first clause (creates short punchy sentences)
        if cn_len > 20 and cn_len < 50 and random.random() < truncate_rate:
            # Truncate to first clause (split at first comma), keep rest as next sentence
            comma_pos = s.find('，')
            if comma_pos > 5 and comma_pos < len(s) - 5:
                first_part = s[:comma_pos]
                first_stripped = first_part.lstrip()
                # Guard 0: don't truncate when the first_part fragment after the
                # last paragraph break is too short. The [。！？] split doesn't
                # respect \n\n, so a segment can span "## header\n\n现在，X..."
                # — truncating yields "## header\n\n现在。X..." stranding a
                # 2-char fragment after the section header.
                last_nl = first_part.rfind('\n')
                if last_nl >= 0:
                    tail_cn = len(re.findall(r'[一-鿿]',
                                             first_part[last_nl + 1:]))
                    if tail_cn < 3:
                        result.append(s + p)
                        i += 1
                        continue
                # Guard 1: skip if first part ends in an attribution/reporting verb.
                # Otherwise "X 指出，" becomes "X 指出。" + bare clause — broken grammar.
                _attribution_suffixes = (
                    '指出', '表明', '认为', '揭示', '发现', '显示', '提出',
                    '说', '称', '讲', '强调', '主张', '断言',
                )
                if first_part.endswith(_attribution_suffixes):
                    result.append(s + p)
                    i += 1
                    continue
                # Guard 2: skip if first part is a subordinate clause (starts with
                # 随着/鉴于/为了/由于/尽管/虽然/如果 etc.). Splitting at comma would
                # leave a fragment that can't stand alone: "随着X的发展。Y" is broken.
                _subordinate_prefixes = (
                    '随着', '鉴于', '为了', '由于', '尽管', '虽然',
                    '如果', '假如', '若是', '倘若', '要是', '即便', '纵然',
                    '除了', '除非', '只要', '只有', '无论', '不管',
                    '当', '每当', '一旦',
                    # cycle 201: 面对X / 处在X = context introducer that needs
                    # a main clause. Splitting at comma leaves a fragment.
                    # ('在' kept out — too broad; handled by suffix guard below)
                    '面对', '处在',
                )
                if first_stripped.startswith(_subordinate_prefixes):
                    result.append(s + p)
                    i += 1
                    continue
                # cycle 201: context-introducer SUFFIXES that need a main
                # clause (covers "在X的背景下" cycle-190 alts: "...这种局面，"
                # "...这个情境里，" "...之中，"). Catches the "在" case
                # without blocking all "在..." sentences.
                _context_suffixes = (
                    '这种局面', '这种情况', '这个情境里', '这种背景下',
                    '之中', '的背景下',
                )
                if first_part.endswith(_context_suffixes):
                    result.append(s + p)
                    i += 1
                    continue
                # Guard 3: skip if next clause starts with a bare causative
                # verb (使/使得/导致/造成 etc.) OR a continuation marker
                # (同时/此外/另外/更/不仅/而且/进而/继而/充分/进一步/同样).
                # These all assume the prior clause's subject/context — splitting
                # creates fragment "X。同时Y。" which reads as orphaned.
                # cycle 206 (sway 标点符号奇怪): added 同时/充分/进一步 etc.
                # Audit on workplace example showed pattern "工作效率，同时也Y，
                # 充分体现Z" splitting into 3 short sentences with multiple
                # paragraph-end periods — sway flagged as awkward.
                _bare_continuators = (
                    '使得', '使', '导致', '引起', '造成', '致使',
                    '同时', '同样', '此外', '另外', '更', '不仅', '而且',
                    '进而', '继而', '充分', '进一步', '同时也',
                )
                # Modal/aux continuators: only block when first_part is a
                # short bare NP (no main verb). Long first_parts with their
                # own verb can stand alone, so allow truncation there.
                _modal_continuators = (
                    '能够', '能', '可以', '可', '将会', '将',
                    '亦可', '亦', '也将', '也能', '也可',
                )
                rest_after_comma = s[comma_pos + 1:].lstrip()
                if rest_after_comma.startswith(_bare_continuators):
                    result.append(s + p)
                    i += 1
                    continue
                if rest_after_comma.startswith(_modal_continuators):
                    # Heuristic: if first_part has a main verb, allow split.
                    # Bare NP (subject-only) creates a fragment.
                    first_cn = len(re.findall(r'[一-鿿]', first_part))
                    _verb_markers = (
                        '是', '有', '做', '用', '把', '让', '使', '给',
                        '提', '推', '完', '实', '达', '形', '构', '反',
                        '显', '表', '维', '保', '改', '优', '调', '处',
                        '通过', '运用', '采用', '成为', '需要', '获得',
                    )
                    if first_cn < 12 and not any(m in first_part for m in _verb_markers):
                        result.append(s + p)
                        i += 1
                        continue
                rest_part = s[comma_pos + 1:]
                result.append(first_part + p)
                # Push the rest as a new "sentence" to be processed
                if rest_part.strip():
                    result.append(rest_part + '。')
                i += 1
                continue

        result.append(s + p)
        i += 1

    return ''.join(result)


# ═══════════════════════════════════════════════════════════════════
#  Strategy 3: Noise expression injection
# ═══════════════════════════════════════════════════════════════════

def _dialogue_density_local(text):
    """Fraction of chars inside Chinese dialogue quotes. AI novels use a
    mix of curly U+201C/D (“”), corner U+300C/D (「」), and ASCII pairs
    (which some models output instead). Threshold 0.08 flags narrative."""
    n = 0
    for pat in (r'“[^“”]{3,}?”', r'「[^「」]{3,}?」'):
        for m in re.findall(pat, text):
            n += len(m)
    # ASCII " pairs: split on ", odd-indexed segments are inside quotes
    parts = text.split('"')
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            if len(parts[i]) >= 3:
                n += len(parts[i])
    return n / max(1, len(text))


# Narrative-safe subset of NOISE_EXPRESSIONS categories. filler/personal/
# transition_casual inject 1st-person author voice or oral fillers that
# break 3rd-person fiction register.
_NARRATIVE_SAFE_CATEGORIES = ['hedging', 'uncertainty', 'self_correction']


def inject_noise_expressions(text, density=0.15, style='general'):
    """
    策略3: 在句子间或句中适当位置插入噪声表达。
    density: 大约每多少句插入一个（0.15 ≈ 每 6-7 句一个）
    style: general / academic
    """
    # cycle 152: when style='general' but the text has 2+ markdown
    # headers (academic survey / technical article), the 'filler' /
    # 'transition_casual' / 'personal' categories from NOISE_EXPRESSIONS
    # ('当然了' / '坦白讲' / '不瞒你说' etc.) read off-register inside
    # formal prose. Promote to the academic noise subset, which keeps
    # only hedging / self_correction / uncertainty.
    if style == 'general':
        n_md_headers = sum(1 for line in text.split('\n')
                           if re.match(r'^\s*#{1,6}\s', line))
        if n_md_headers >= 2:
            style = 'academic'

    if style == 'academic':
        categories = NOISE_ACADEMIC_CATEGORIES
        expressions = NOISE_ACADEMIC_EXPRESSIONS
    else:
        categories = list(NOISE_EXPRESSIONS.keys())
        expressions = NOISE_EXPRESSIONS
        # Narrative guard: if text is dialogue-heavy, drop categories that
        # break 3rd-person voice (filler/personal/transition_casual).
        if _dialogue_density_local(text) >= 0.08:
            categories = [c for c in categories if c in _NARRATIVE_SAFE_CATEGORIES]
            if not categories:
                return text

    # Split into sentences
    parts = re.split(r'([。！？])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ''
        if s.strip():
            sentences.append([s, p])
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append([parts[-1], ''])

    if len(sentences) < 3:
        return text

    # Track expressions already injected in this run. Re-injecting the same
    # phrase ("\u5f80\u6df1\u4e86\u8bb2" / "\u5e73\u5fc3\u800c\u8bba") three times in one sample reads as a
    # tic, which detect_cn flags as repetitive and a human reviewer flags as
    # robot-style.
    used = set()

    # cycle 203 (sway directive \u8bed\u53e5\u901a\u987a\u4f18\u5148): track which paragraphs already
    # had a noise injection. Multiple injections per paragraph create
    # "\u5728\u6211\u770b\u6765\uff0cX\u3002\u6ce8\u610f\uff0cY\u3002\u8bf4\u5230\u5e95\uff0cZ" robotic chains. Hard cap = 1
    # injection per paragraph. Identifies paragraph by the cumulative \n\n
    # count in text up to the sentence position.
    para_injected = {}

    injected = 0
    cum_text = ''
    for i in range(len(sentences)):
        s_text = sentences[i][0]
        s_punct = sentences[i][1] or ''
        # cycle 203: track cumulative text to identify current paragraph
        # (paragraph = chunk between \n\n breaks). Update at top so all
        # `continue` branches keep para_idx in sync.
        para_idx = cum_text.count('\n\n')
        cum_text += s_text + s_punct
        # Skip the last sentence (avoid orphaned expressions)
        if i >= len(sentences) - 1:
            continue
        # Skip very short sentences
        if len(re.findall(r'[\u4e00-\u9fff]', s_text)) < 8:
            continue
        # Skip sentences that contain dialogue quotes. Injecting a noise
        # expression into a quoted line puts narrator filler inside a
        # character's mouth \u2014 awkward and breaks dialogue flow.
        if '"' in s_text or '\u201c' in s_text or '\u201d' in s_text or '\u300c' in s_text or '\u300d' in s_text:
            continue
        # Cycle 57/58: skip sentences that start with markdown structural
        # markers (# heading / - * bullet / **bold** subheader / 1. 2.
        # numbered list). Injecting '\u4e0d\u7792\u4f60\u8bf4\uff0c' before '#### 2.2 ...' or
        # '\u5728\u6211\u770b\u6765\uff0c**3. \u54c1\u724c\u5efa\u8bbe\uff1a\u6587\u5316\u2026**' corrupts the structural marker.
        # Cycle 58 widens the **-prefix check from "starts AND ends with **"
        # (pure bold subheader) to just "starts with **" \u2014 covers hybrid
        # forms like '**1. \u8d44\u6e90\u74f6\u9888\uff1a** \u9ad8\u5e76\u53d1\u610f\u5473\u7740\u2026' that the cycle 57
        # check missed (audit found 34 longform samples with this pattern).
        s_lstripped = s_text.lstrip()
        # cycle 203 (sway directive 语句通顺优先): skip if sentence already
        # starts with a SHORT transition marker. These come from
        # patterns_cn.json replacements (值得注意的是→注意, 综上所述→总之,
        # 其次→另外/此外, etc.). Stacking noise on top reads as
        # "在我看来，注意，X..." — multiple transitions piled up, robotic.
        # Trade: drops some LR-favorable noise, accepted per sway directive.
        _existing_transitions = (
            '注意，', '特别说一下，', '要提醒的是，', '总之，', '说到底，',
            '简单讲，', '归结起来，', '另外，', '此外，', '还有，',
            '可以看到，', '很明显，', '你会发现，',
            '一开始，', '最初，', '起头，', '先说，',
            '接着，', '然后，', '再就是，', '最后说一点，',
            # Standard discourse connectors: stacking noise before these
            # creates "顺着这个思路，然而，X" double-connector reads.
            '然而，', '但是，', '不过，', '可是，', '因此，', '所以，',
            '因而，', '而且，', '同时，', '不仅，', '相反，', '反之，',
        )
        if s_lstripped.startswith(_existing_transitions):
            continue
        # cycle 203: per-paragraph injection cap = 1 (sway 语句通顺优先).
        # Skip if this paragraph already had an injection — prevents
        # "在我看来，X。注意，Y" cross-sentence stacking.
        if para_injected.get(para_idx, 0) >= 1:
            continue
        # cycle 203 sub: also skip if the same paragraph (the one we're
        # in, or that the current sentence will land in) already contains
        # any of the existing-transition markers (from replacements). This
        # catches "注意，X。" + "在我看来，Y" same-paragraph stacking
        # where 注意 came from values_注意的是 replacement, not noise.
        # Build the paragraph slice: all sentences sharing this paragraph.
        para_slice = ''
        running_para = 0
        for j in range(len(sentences)):
            if running_para == para_idx:
                para_slice += sentences[j][0] + (sentences[j][1] or '')
            running_para += (sentences[j][0] + (sentences[j][1] or '')).count('\n\n')
            if running_para > para_idx:
                break
        if any(t in para_slice for t in _existing_transitions):
            continue
        if s_lstripped.startswith('#') or s_lstripped.startswith('- ') or s_lstripped.startswith('* '):
            continue
        if s_lstripped.startswith('**'):
            continue
        if re.match(r'^\d+[.\u3002\uff0e)\uff09]', s_lstripped):
            continue
        # Chinese numbered section headers: "\u4e00\u3001X" / "\uff08\u4e00\uff09X" \u2014 common
        # in long-form Chinese essays. Don't inject noise before them.
        if re.match(r'^[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001,\uff0c]', s_lstripped):
            continue
        if re.match(r'^[\uff08(][\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\uff09)]', s_lstripped):
            continue
        # Title/heading guard: standalone line without terminal punctuation
        # (ends in non-\u3002\uff01\uff1f and is followed by \n\n) \u2014 usually a title.
        # "\u4ece\u7a0b\u5e8f\u5458\u8f6c\u4ea7\u54c1\u7ecf\u7406\uff0c\u7b2c\u4e00\u5e74\u5b66\u5230\u7684\u4e09\u4ef6\u4e8b" \u2192 skip noise injection.
        s_trimmed = s_text.rstrip()
        if s_trimmed and s_trimmed[-1] not in '\u3002\uff01\uff1f.!?':
            next_text = sentences[i + 1][0] if i + 1 < len(sentences) else ''
            if next_text.startswith('\n\n') or next_text.startswith('\n'):
                continue
        if random.random() > density:
            continue

        cat = random.choice(categories)
        expr_list = expressions.get(cat, [])
        if not expr_list:
            continue
        avail = [e for e in expr_list if e not in used]
        if not avail:
            avail = expr_list  # fallback when category exhausted
        expr = random.choice(avail)
        used.add(expr)

        s, p = sentences[i]

        # Preserve leading whitespace (\n\n paragraph breaks) — sentences
        # that start a new paragraph have \n\n at their head (artifact of
        # the [。！？] split). .lstrip() would eat those and collapse
        # paragraph structure.
        leading_ws_len = len(s) - len(s.lstrip())
        leading = s[:leading_ws_len]
        s_body = s[leading_ws_len:]

        # Decide insertion position
        if cat in ('hedging', 'filler', 'personal', 'transition_casual'):
            # Insert at sentence beginning (after any paragraph break)
            s = leading + expr + '，' + s_body
        elif cat in ('self_correction', 'uncertainty'):
            # Insert mid-sentence at a comma
            comma_pos = s_body.find('，')
            if comma_pos > 3:
                s = leading + s_body[:comma_pos + 1] + expr + '，' + s_body[comma_pos + 1:]
            else:
                s = leading + expr + '，' + s_body

        sentences[i] = [s, p]
        injected += 1
        # cycle 203: bump per-paragraph counter for cap enforcement
        para_injected[para_idx] = para_injected.get(para_idx, 0) + 1

    return ''.join(s + p for s, p in sentences)


# ─── Core Transforms ───

def remove_three_part_structure(text):
    """Remove 首先/其次/最后, 第一/第二/第三 patterns"""
    # Don't just delete — replace with natural transitions
    replacements = [
        (r'首先[，,]\s*', ''),
        (r'其次[，,]\s*', lambda m: random.choice(['另外，', '此外，', ''])),
        (r'最后[，,]\s*', lambda m: random.choice(['还有，', ''])),  # cycle 208: drop 最后说一点 (awkward in essays)
        (r'第一[，,、]\s*', ''),
        (r'第二[，,、]\s*', lambda m: random.choice(['接着，', '然后，', ''])),
        (r'第三[，,、]\s*', lambda m: random.choice(['还有，', '再就是，', ''])),
        (r'第[四五六七八九][，,、]\s*', lambda m: random.choice(['另外，', ''])),
        (r'其一[，,、]\s*', ''),
        (r'其二[，,、]\s*', lambda m: random.choice(['另外，', ''])),
        (r'其三[，,、]\s*', lambda m: random.choice(['还有，', ''])),
    ]
    
    for pattern, repl in replacements:
        if callable(repl):
            text = re.sub(pattern, repl, text)
        else:
            text = re.sub(pattern, repl, text)
    
    return text

def replace_phrases(text, casualness=0.3):
    """Replace AI phrases with natural alternatives (context-aware)"""
    # Apply regex replacements FIRST (per-sentence, max 1 regex replacement per sentence)
    # Split by sentence-ending punctuation to handle multiple templates in same line
    parts = re.split(r'([。！？\n])', text)
    rebuilt = []
    for part in parts:
        replaced = False
        for pattern, alternatives in REGEX_REPLACEMENTS.items():
            if replaced:
                break
            if isinstance(alternatives, str):
                alternatives = [alternatives]
            try:
                match = re.search(pattern, part)
                if match:
                    replacement = random.choice(alternatives)
                    expanded = match.expand(replacement)
                    part = part[:match.start()] + expanded + part[match.end():]
                    replaced = True
            except re.error:
                pass
        rebuilt.append(part)
    text = ''.join(rebuilt)
    
    # Then plain replacements, sorted by length (longest first) to avoid partial matches
    sorted_phrases = sorted(PLAIN_REPLACEMENTS.keys(), key=len, reverse=True)
    
    for phrase in sorted_phrases:
        alternatives = PLAIN_REPLACEMENTS[phrase]
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        
        if phrase in text:
            # Filter out alternatives that contain the phrase as a substring —
            # those cause infinite re-match loops (e.g. 相反 -> 相反地 reinserts
            # 相反). Without this, slow-path bug: cycle 2 HC3 500 hang, cycle 13
            # longform benchmark kill on samples 85/86/133/144 (all had 相反).
            safe_alts = [alt for alt in alternatives if phrase not in alt]
            if not safe_alts:
                continue
            # Dedupe replacement choices for this phrase. pick_best_replacement
            # is deterministic on perplexity, so when the same phrase occurs
            # multiple times in a long sample it gets rewritten to the same
            # alternative every iteration ('可能引起' x4-5 in audit). Track
            # which alts have been used and prefer unused ones; fall back to
            # the full safe list once exhausted.
            used = set()
            replacement = pick_best_replacement(text, phrase, safe_alts)
            text = text.replace(phrase, replacement, 1)
            used.add(replacement)
            while phrase in text:
                avail = [a for a in safe_alts if a not in used]
                if not avail:
                    # Cycle exhausted — clear `used` so the next round
                    # rotates through the alts again instead of falling
                    # back to a single deterministic pick. Without this
                    # reset, sample 38 of the longform corpus rewrites
                    # 9 occurrences of '然后' as 6×'随后' + '接着' + '之后'
                    # + '随后' instead of an even distribution.
                    used.clear()
                    avail = safe_alts
                replacement = pick_best_replacement(text, phrase, avail)
                text = text.replace(phrase, replacement, 1)
                used.add(replacement)

    return text

def merge_short_sentences(text, min_len=8):
    """Merge overly short consecutive sentences, with burstiness guard."""
    # Measure burstiness before restructuring
    burst_before = _compute_burstiness(text)

    sentences = re.split(r'([。！？])', text)
    if len(sentences) < 4:
        return text
    
    result = []
    i = 0
    while i < len(sentences) - 1:
        sent = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        # Check if this and next sentence are both short
        next_sent = sentences[i + 2] if i + 2 < len(sentences) else ''
        
        if len(sent.strip()) < min_len and len(next_sent.strip()) < min_len and next_sent.strip():
            # Don't merge across paragraph boundaries — \n\n leading
            # next_sent would be stripped by .strip(), collapsing paragraphs.
            if '\n' in sent or '\n' in next_sent:
                result.append(sent + punct)
                i += 2
            else:
                # Merge with comma
                merged = sent.strip() + '，' + next_sent.strip()
                next_punct = sentences[i + 3] if i + 3 < len(sentences) else '。'
                result.append(merged + next_punct)
                i += 4
        else:
            result.append(sent + punct)
            i += 2
    
    # Handle remaining
    while i < len(sentences):
        result.append(sentences[i])
        i += 1
    
    new_text = ''.join(result)

    # Burstiness guard: if merging made text more uniform, revert
    if burst_before is not None:
        burst_after = _compute_burstiness(new_text)
        if burst_after is not None and burst_after < burst_before * 0.8:
            return text  # revert — merging reduced burstiness too much

    return new_text

def split_long_sentences(text, max_len=80):
    """Split overly long sentences at natural breakpoints, with burstiness guard."""
    burst_before = _compute_burstiness(text)

    sentences = re.split(r'([。！？])', text)
    result = []
    
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        chinese_len = len(re.findall(r'[\u4e00-\u9fff]', sent))
        
        if chinese_len > max_len:
            # Find natural split points: 但是/不过/然而/同时/而且
            split_points = [
                (m.start(), m.group()) for m in
                re.finditer(r'[，,](但是|不过|然而|同时|而且|所以|因此|另外)', sent)
            ]

            def _tail_too_short(part):
                # Skip splits that would strand a tiny fragment after the most
                # recent paragraph/line break. Sentences split by [。！？] can
                # span "## header\n\nX，Y" so a comma-split produces broken
                # "## header\n\nX。Y" output.
                last_nl = part.rfind('\n')
                if last_nl < 0:
                    return False
                tail_cn = len(re.findall(r'[一-鿿]', part[last_nl + 1:]))
                return tail_cn < 3

            if split_points:
                # Split at the most central point
                mid = len(sent) // 2
                best = min(split_points, key=lambda x: abs(x[0] - mid))
                part1 = sent[:best[0]]
                part2 = sent[best[0]+1:]  # Skip the comma
                if _tail_too_short(part1):
                    result.append(sent + punct)
                else:
                    result.append(part1 + '。' + part2 + punct)
            else:
                # Split at a comma near the middle. Filter commas whose
                # following clause starts with a bare causative verb
                # (使得/导致/etc.) — splitting there yields "X。使得Y" which
                # strands a subject-less verb.
                _bare_continuators = (
                    '使得', '使', '导致', '引起', '造成', '致使',
                )
                def _safe_comma(idx):
                    rest = sent[idx + 1:].lstrip()
                    return not rest.startswith(_bare_continuators)
                commas = [m.start() for m in re.finditer(r'[，,]', sent)
                          if _safe_comma(m.start())]
                if commas:
                    mid = len(sent) // 2
                    best_comma = min(commas, key=lambda x: abs(x - mid))
                    part1 = sent[:best_comma]
                    part2 = sent[best_comma+1:]
                    if _tail_too_short(part1):
                        result.append(sent + punct)
                    else:
                        result.append(part1 + '。' + part2 + punct)
                else:
                    result.append(sent + punct)
        else:
            result.append(sent + punct)
    
    # Handle remaining
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1])
    
    new_text = ''.join(result)

    # Burstiness guard: if splitting made text more uniform, revert
    if burst_before is not None:
        burst_after = _compute_burstiness(new_text)
        if burst_after is not None and burst_after < burst_before * 0.8:
            return text

    return new_text

def _para_cv(paragraphs):
    """Helper: compute paragraph-length CV over valid (>=20 cn) paragraphs."""
    cn_lens = [len(re.findall(r'[一-鿿]', p)) for p in paragraphs]
    valid_lens = [l for l in cn_lens if l >= 20]
    if len(valid_lens) < 3:
        return None
    m = sum(valid_lens) / len(valid_lens)
    if m == 0:
        return None
    var = sum((l - m) ** 2 for l in valid_lens) / len(valid_lens)
    return (var ** 0.5) / m


def vary_paragraph_rhythm(text):
    """Break uniform paragraph lengths by merging or splitting"""
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 3:
        return text

    # v5 P1.2 guard (cycle 143): if paragraph-length CV is already
    # adequate (>=0.40, near human distribution), skip merge/split.
    # cycle 142 found that further structural tweaks on already-varied
    # paragraphs push the distribution back toward uniform — a stuck
    # academic sample went from CV 0.405 to 0.320 after the full
    # pipeline because a long paragraph got split, averaging the
    # distribution down.
    cv_initial = _para_cv(paragraphs)
    if cv_initial is not None and cv_initial >= 0.40:
        return text

    lengths = [len(p) for p in paragraphs]
    avg_len = sum(lengths) / len(lengths) if lengths else 100

    def _is_md_header(p):
        # Markdown headers ('# ', '## ', '### ' …), bullets, bold section
        # subheaders, numbered list items, and dialogue lines are
        # deliberately short structural paragraphs; merging them collapses
        # document structure (sample 63 of longform corpus: ## headers
        # lost; cycle-44 audit: bold subheaders + numbered list items;
        # cycle-46 audit: novel sample 1323 had two dialogue paragraphs
        # like '"嗯，我很喜欢。"' merged into one block, losing the
        # turn-by-turn formatting).
        s = p.lstrip()
        if s.startswith('#') or s.startswith('- ') or s.startswith('* '):
            return True
        if s.startswith('**') and s.rstrip().endswith('**'):
            return True
        if re.match(r'^\d+[.。．)）]', s):
            return True
        # Chinese numbered section headers: "一、X" / "（一）X" / "(一)X"
        # Common in long-form Chinese essays (long_blog 一、 二、 三、)
        if re.match(r'^[一二三四五六七八九十]+[、,，]', s):
            return True
        if re.match(r'^[（(][一二三四五六七八九十]+[）)]', s):
            return True
        # Dialogue line (Chinese / Western quotes / Japanese 「」)
        if s and s[0] in '"“「':
            return True
        return False

    # cycle 226 N-2d: enumeration markers AS THEY SURVIVE the upstream
    # remove_three_part_structure (strips 首先/其次/最后; replaces 其次→
    # 另外/此外/'') and replace_phrases (此外→还有/再说/加之, 综上所述→
    # 总之/说到底/简单讲). The AI long-form pattern is "each enumerator
    # gets its own paragraph" — so the surviving paragraph heads are what
    # we want to merge into the previous block to break that rhythm.
    _ENUM_PARA_HEADS = (
        '综上', '此外', '另外', '总之', '总的来说', '总而言之', '再者',
        '还有', '接着', '然后', '再就是', '加之', '再说', '说到底', '简单讲',
    )

    def _starts_enum(p):
        s = p.lstrip()
        for m in _ENUM_PARA_HEADS:
            if s.startswith(m + '，') or s.startswith(m + ',') or s.startswith(m + '、'):
                return True
        return False

    result = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]

        # Enum-marker preferential merge (N-2d): if the NEXT paragraph starts
        # with a surviving enumeration head, merge into current with combined
        # length cap to avoid mega-paragraphs. Length cap 2.5x avg keeps the
        # merged block within human plausible range.
        if (i + 1 < len(paragraphs) and
            _starts_enum(paragraphs[i + 1]) and
            not _is_md_header(para) and
            not _is_md_header(paragraphs[i + 1]) and
            len(para) + len(paragraphs[i + 1]) < avg_len * 2.5 and
            random.random() < 0.45):
            merged = para + '\n' + paragraphs[i + 1]
            result.append(merged)
            i += 2
            continue

        # Randomly merge short adjacent paragraphs (skip markdown headers /
        # bullet items — those are deliberately short structural markers).
        if (i + 1 < len(paragraphs) and
            len(para) < avg_len * 0.6 and
            len(paragraphs[i + 1]) < avg_len * 0.6 and
            not _is_md_header(para) and
            not _is_md_header(paragraphs[i + 1]) and
            random.random() < 0.4):
            merged = para + '\n' + paragraphs[i + 1]
            result.append(merged)
            i += 2
            continue
        
        # Split long paragraphs
        if len(para) > avg_len * 1.5:
            sentences = re.split(r'([。！？])', para)
            mid = len(sentences) // 2
            # Ensure we split at a sentence boundary (every other element is punctuation)
            if mid % 2 == 1:
                mid -= 1
            part1 = ''.join(sentences[:mid])
            part2 = ''.join(sentences[mid:])
            if part1.strip() and part2.strip():
                result.append(part1.strip())
                result.append(part2.strip())
                i += 1
                continue
        
        result.append(para)
        i += 1

    # cycle 219 N-2b: fallback to push paragraph CV up for AI uniform
    # texts. When the main loop's 0.6/1.5 thresholds didn't fire (all
    # paragraphs in narrow band), CV stays low. To INCREASE variance,
    # we merge two adjacent paragraphs whose combined length would be
    # > 1.5x current average — this creates a long outlier on the
    # right tail. Only fires on real long text (>= 8 paragraphs) to
    # avoid distorting medium-length 5-paragraph compositions where
    # merging 2/5 over-shifts CV past human distribution.
    cv_after = _para_cv(result)
    if cv_after is not None and cv_after < 0.35 and len(result) >= 8:
        cn_lens = [len(re.findall(r'[一-鿿]', p)) for p in result]
        valid_lens = [l for l in cn_lens if l >= 20]
        avg_cn = sum(valid_lens) / len(valid_lens) if valid_lens else 100
        # Find adjacent pair whose combined length is the most above
        # average (creates a long outlier when merged).
        best = None
        for k in range(len(result) - 1):
            if _is_md_header(result[k]) or _is_md_header(result[k + 1]):
                continue
            if cn_lens[k] < 20 or cn_lens[k + 1] < 20:
                continue
            combined = cn_lens[k] + cn_lens[k + 1]
            # Want combined > 1.5x avg to push variance.
            if combined > avg_cn * 1.5:
                excess = combined - avg_cn
                if best is None or excess > best[0]:
                    best = (excess, k)
        if best is not None and random.random() < 0.6:
            k = best[1]
            merged = result[k] + '\n' + result[k + 1]
            result = result[:k] + [merged] + result[k + 2:]

    return join_paragraphs(result)

def reduce_punctuation(text):
    """Reduce excessive punctuation intelligently"""
    # Replace some semicolons with commas or periods
    parts = text.split('；')
    if len(parts) > 3:
        result_parts = [parts[0]]
        for i, part in enumerate(parts[1:], 1):
            # Alternate between comma and period
            if i % 2 == 0:
                result_parts.append('。' + part.lstrip())
            else:
                result_parts.append('，' + part)
        text = ''.join(result_parts)
    
    # Limit consecutive em dashes
    text = re.sub(r'——', '—', text)
    
    return text

def cap_transition_density(text, target=6.0):
    """Drop clause-initial transition phrases until density <= target.

    Runs AFTER all other humanize passes. Keeps transitions that are
    low-density already; removes excess probabilistically. Detect threshold
    fires at density > 8 per 1000 chars, so target 6 gives margin.
    """
    try:
        from ngram_model import _TRANSITION_PHRASES
    except ImportError:
        from scripts.ngram_model import _TRANSITION_PHRASES

    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cn_chars < 100:
        return text

    hits = sum(text.count(p) for p in _TRANSITION_PHRASES)
    density = hits / cn_chars * 1000
    if density <= target:
        return text

    remove_prob = min(0.9, (density - target) / density)

    for phrase in sorted(_TRANSITION_PHRASES, key=len, reverse=True):
        esc = re.escape(phrase)
        pattern = re.compile(r'(^|[。！？\n])(' + esc + r')([，,、])?')

        def sub(m):
            if random.random() < remove_prob:
                return m.group(1)
            return m.group(0)

        text = pattern.sub(sub, text)

    return text


def inject_sentence_particles(text, rate=0.15):
    """Append casual sentence-ending particles (吧/嘛/呗) to random statements.

    Intended for casual/social/chat scenes only. Skips questions/exclamations
    (already tonal) and sentences already ending in a particle. Short sentences
    skipped (too brittle), very long ones skipped (feels forced).
    """
    parts = re.split(r'([。！？])', text)
    particles = ['吧', '嘛', '呗']
    for i in range(0, len(parts) - 1, 2):
        sent = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ''
        if punct in '！？':
            continue
        cn = sum(1 for c in sent if '\u4e00' <= c <= '\u9fff')
        if cn < 6 or cn > 40:
            continue
        rstripped = sent.rstrip()
        if rstripped and rstripped[-1] in '吧嘛呗呢啊哦嗯哈的了':
            continue
        if random.random() < rate:
            parts[i] = rstripped + random.choice(particles)
    return ''.join(parts)


def add_casual_expressions(text, casualness=0.3):
    """Inject casual/human expressions"""
    if casualness < 0.2:
        return text
    
    casual_openers = ['说实话', '其实', '确实', '讲真', '坦白说']
    casual_transitions = ['话说回来', '说到这个', '不过呢', '但是吧']
    casual_endings = ['就是这么回事', '差不多就这样', '大概就这些']
    
    sentences = re.split(r'([。！？])', text)
    result = []
    added = 0
    total = len(sentences) // 2
    max_additions = max(1, int(total * casualness * 0.3))
    
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        if added < max_additions and random.random() < casualness * 0.2:
            if i == 0:
                opener = random.choice(casual_openers)
                sent = opener + '，' + sent
            elif i > total:
                transition = random.choice(casual_transitions)
                sent = transition + '，' + sent
            added += 1
        
        result.append(sent + punct)
    
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1])
    
    return ''.join(result)

def shorten_paragraphs(text, max_length=150):
    """Break long paragraphs for social/chat scenes"""
    paragraphs = split_paragraphs(text)
    result = []
    
    for para in paragraphs:
        if len(para) > max_length:
            sentences = re.split(r'([。！？])', para)
            chunks = []
            current = ''
            
            for i in range(0, len(sentences) - 1, 2):
                sent = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
                if len(current) + len(sent) > max_length and current:
                    chunks.append(current.strip())
                    current = sent
                else:
                    current += sent
            
            if current.strip():
                chunks.append(current.strip())
            
            result.extend(chunks)
        else:
            result.append(para)
    
    return join_paragraphs(result)

def diversify_vocabulary(text):
    """Reduce word repetition by using synonyms"""
    # Common overused words and their alternatives
    diversity_map = {
        '进行': ['做', '开展', '实施', '推进'],
        '实现': ['达到', '做到', '完成'],
        '提供': ['给出', '带来'],  # Cycle 63: dropped 拿出 (see WORD_SYNONYMS comment)
        '具有': ['有', '拥有', '带有'],
        # cycle 252: dropped '深入' — adjacency cascade "进一步深入" → "深入深入"
        # leaves no alt; effectively skip 进一步 in diversify_vocabulary path.
        # '进一步': drop entry (was ['深入'])
        '不断': ['持续', '一直', '始终'],
        # '有效' skipped: attributive/adj usage (有效证件) breaks with verb substitutes
        '积极': ['主动', '热心'],
        '促进': ['推动', '带动'],
        '加强': ['强化', '增强'],
        '提高': ['提升', '增加'],
        # cycle 164: dropped '重要' — same compound-breakage as
        # WORD_SYNONYMS upstream (重要性 → 核心性, 至关重要 → 至关核心
        # both broken).
    }
    
    for word, alts in diversity_map.items():
        count = text.count(word)
        if count > 2:
            # Keep first occurrence, replace subsequent
            first = True
            parts = text.split(word)
            result = [parts[0]]
            for part in parts[1:]:
                if first:
                    result.append(word)
                    first = False
                else:
                    result.append(random.choice(alts))
                result.append(part)
            text = ''.join(result)
    
    return text

# ─── Main Humanize Pipeline ───

def _estimate_source_aiscore(text):
    """Quick pre-detect of how AI-like the input is. Returns 0-100 score or None."""
    try:
        from detect_cn import detect_patterns, calculate_score
    except ImportError:
        try:
            from scripts.detect_cn import detect_patterns, calculate_score
        except ImportError:
            return None
    try:
        issues, metrics = detect_patterns(text)
        return calculate_score(issues, metrics)
    except Exception:
        return None


DEFAULT_BEST_OF_N = 20
DEFAULT_SECONDARY_WEIGHT = 0.2


def _clamp_0_100(value):
    return max(0.0, min(100.0, float(value)))


def _norm_linear(value, low, high, invert=False):
    if value is None or high == low:
        return 0.0
    raw = (float(value) - low) / (high - low)
    if invert:
        raw = 1.0 - raw
    return _clamp_0_100(raw * 100.0)


def _starter_entropy(text, width=2):
    sentences = [s.strip() for s in re.split(r'[。！？!?；;\n]+', text) if s.strip()]
    starters = {}
    total = 0
    for sent in sentences:
        chars = re.findall(r'[\u4e00-\u9fff]', sent)
        if len(chars) < width:
            continue
        key = ''.join(chars[:width])
        starters[key] = starters.get(key, 0) + 1
        total += 1
    if total < 5:
        return 0.0
    entropy = 0.0
    for count in starters.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _secondary_signal_details(text):
    """Return auxiliary best-of-n AI-likeness score and raw/capped features.

    These are deliberately not LR calibration inputs. They reuse already
    implemented but capped/disabled signals to sway candidate ranking only.
    """
    if not text or ngram_analyze is None:
        return {
            'score': 0.0,
            'bino': 0.0,
            'curv': 0.0,
            'mattr': 0.0,
            'starter_h': 0.0,
            'bino_s': 0.0,
            'curv_s': 0.0,
            'mattr_s': 0.0,
            'starter_s': 0.0,
        }
    try:
        analysis = ngram_analyze(text)
    except Exception:
        return {
            'score': 0.0,
            'bino': 0.0,
            'curv': 0.0,
            'mattr': 0.0,
            'starter_h': 0.0,
            'bino_s': 0.0,
            'curv_s': 0.0,
            'mattr_s': 0.0,
            'starter_s': 0.0,
        }

    bino = (analysis.get('bino') or {}).get('mean_lp_diff') or 0.0
    curv = (analysis.get('curv') or {}).get('curvature_mean') or 0.0
    mattr = analysis.get('char_mattr') or 0.0
    starter_h = _starter_entropy(text, width=2)

    # Direction: higher score means more AI-like. Binoculars diff is less
    # negative on HC3 ChatGPT; curvature is higher; MATTR and starter entropy
    # are lower when wording/openers are more repetitive.
    bino_s = _norm_linear(bino, -4.6, -2.2)
    curv_s = _norm_linear(curv, 0.0, 1.2)
    mattr_s = _norm_linear(mattr, 0.50, 0.72, invert=True)
    starter_s = _norm_linear(starter_h, 1.2, 2.4, invert=True)

    score = (
        0.35 * bino_s +
        0.25 * curv_s +
        0.25 * mattr_s +
        0.15 * starter_s
    )
    return {
        'score': round(_clamp_0_100(score), 2),
        'bino': round(float(bino), 4),
        'curv': round(float(curv), 4),
        'mattr': round(float(mattr), 4),
        'starter_h': round(float(starter_h), 4),
        'bino_s': round(bino_s, 2),
        'curv_s': round(curv_s, 2),
        'mattr_s': round(mattr_s, 2),
        'starter_s': round(starter_s, 2),
    }


def _compute_secondary_signal(text):
    return _secondary_signal_details(text)['score']


def _pick_lr_scene(text):
    """Pick the LR scorer used to rank best-of-n candidates."""
    academic_hits = sum(1 for marker in _ACADEMIC_LR_MARKERS if marker in text)
    if _count_chinese_chars(text) >= _LONGFORM_LR_CN_CHAR_THRESHOLD:
        return 'longform'
    if academic_hits >= 2:
        return 'academic'
    return 'general'


def _format_best_of_debug(seed, scene_picked, lr_scores, secondary, rank_score,
                          fused_score, top_contribs):
    top = ', '.join(f'{name}={value:+.2f}' for name, value in top_contribs[:3])
    return (
        f'best_of_n seed={seed} scene_picked={scene_picked} '
        f'LR_general={lr_scores.get("general", "NA")} '
        f'LR_academic={lr_scores.get("academic", "NA")} '
        f'LR_longform={lr_scores.get("longform", "NA")} '
        f'secondary={secondary["score"]} '
        f'[bino={secondary["bino"]} curv={secondary["curv"]} '
        f'mattr={secondary["mattr"]} starter_h={secondary["starter_h"]}] '
        f'rank={rank_score:.2f} fused={fused_score} top_3_contributions=[{top}]'
    )


def humanize(text, scene='general', aggressive=False, seed=None, best_of_n=DEFAULT_BEST_OF_N,
             style=None, debug_best_of_n=False, score_mode='lr',
             secondary_weight=DEFAULT_SECONDARY_WEIGHT):
    """Apply all humanization transformations in order.

    Graduated intensity based on source AI-score (pre-detect):
      - score < 15 (conservative): only phrase replacement + punctuation cleanup
      - score 15-39 (moderate): + restructure + lighter bigram substitution
      - score >= 40 (full): entire pipeline including noise injection
    Aggressive flag forces 'full' tier.

    best_of_n: if set to an integer, runs humanize N times with different seeds
    and returns the output that scores lowest on the scene-aware LR ensemble
    (requires scripts/lr_coef_*.json). Useful when minimizing LR score matters
    more than latency.

    Rationale: HC3 benchmark showed that full pipeline on already-clean text
    (source score < 15) adds spurious AI patterns (段落均匀/熵低) via noise
    injection, sometimes INCREASING detected score. Tiered intensity avoids this.
    """
    if best_of_n and best_of_n > 1:
        try:
            from ngram_model import compute_lr_score
        except ImportError:
            from scripts.ngram_model import compute_lr_score
        if score_mode not in ('lr', 'fused', 'lr+rule'):
            raise ValueError('score_mode must be one of: lr, fused, lr+rule')
        detect_for_rule = None
        if score_mode in ('fused', 'lr+rule') or debug_best_of_n:
            try:
                from detect_cn import calculate_score, detect_patterns
            except ImportError:
                from scripts.detect_cn import calculate_score, detect_patterns
            detect_for_rule = (calculate_score, detect_patterns)
        base_seed = seed if seed is not None else 42
        candidates = []
        for i in range(best_of_n):
            s = base_seed + i
            out = humanize(text, scene=scene, aggressive=aggressive,
                           seed=s, best_of_n=None, style=style)
            lr_scene = _pick_lr_scene(out)
            if lr_scene == 'longform':
                out = _apply_longform_mutation_profile(
                    out, mutation_seed=s, scene=scene, style=style)
                lr_scene = _pick_lr_scene(out)
            lr = compute_lr_score(out, scene=lr_scene)
            score = lr['score'] if lr else 50
            rule_score = 0
            if detect_for_rule:
                calculate_score, detect_patterns = detect_for_rule
                issues, metrics = detect_patterns(out)
                rule_score = calculate_score(issues, metrics)
            fused = round(0.8 * score + 0.2 * rule_score)
            secondary = _secondary_signal_details(out)
            if score_mode == 'fused':
                rank_score = fused + secondary_weight * secondary['score']
                rank_tiebreak = score
            elif score_mode == 'lr+rule':
                rank_score = score + secondary_weight * secondary['score']
                rank_tiebreak = rule_score
            else:
                rank_score = score + secondary_weight * secondary['score']
                rank_tiebreak = 0
            if debug_best_of_n:
                lr_scores = {}
                for debug_scene in ('general', 'academic', 'longform'):
                    debug_lr = compute_lr_score(out, scene=debug_scene)
                    lr_scores[debug_scene] = debug_lr['score'] if debug_lr else 'NA'
                top_contribs = lr.get('top_contributions', []) if lr else []
                print(_format_best_of_debug(s, lr_scene, lr_scores, secondary,
                                            rank_score, fused, top_contribs),
                      file=sys.stderr)
            candidates.append((rank_score, rank_tiebreak, s, out))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return candidates[0][3]

    if seed is not None:
        random.seed(seed)

    config = SCENES.get(scene, SCENES['general'])
    casualness = config.get('casualness', 0.3)
    if aggressive:
        casualness = min(1.0, casualness + 0.3)

    source_score = _estimate_source_aiscore(text)
    # Tier thresholds calibrated on HC3-Chinese: most naturally-written ChatGPT
    # scores 5-25 on detect_cn. Full pipeline on very-clean input (< 5) adds
    # spurious noise. Moderate tier skips noise/sentence-randomization but keeps
    # everything else. Trade picks up most of the full-tier gains with fewer regressions.
    if aggressive or source_score is None or source_score >= 25:
        tier = 'full'
    elif source_score >= 5:
        tier = 'moderate'
    else:
        tier = 'conservative'

    # Pass 1: Structure cleanup — always run (safe, targeted)
    text = remove_three_part_structure(text)
    text = replace_phrases(text, casualness)

    # Pass 2: Deep sentence restructuring — all tiers (with moderate strength in conservative)
    try:
        from restructure_cn import deep_restructure
    except ImportError:
        try:
            from scripts.restructure_cn import deep_restructure
        except ImportError:
            deep_restructure = None
    if deep_restructure:
        # Conservative keeps restructure but with aggressive=False to be gentler
        text = deep_restructure(text, aggressive=aggressive, scene=scene)

    # Pass 2b: Sentence merge/split
    if config.get('merge_short', False):
        text = merge_short_sentences(text)
    if config.get('split_long', False):
        text = split_long_sentences(text)

    # Pass 3: Rhythm and variety — diversify all tiers, rhythm only moderate+
    text = reduce_punctuation(text)
    text = diversify_vocabulary(text)
    if tier != 'conservative' and config.get('rhythm_variation', False):
        text = vary_paragraph_rhythm(text)

    # Pass 4: Scene-specific — only at full tier
    if tier == 'full':
        if config.get('add_casual', False) or aggressive:
            text = add_casual_expressions(text, casualness)
            # Sentence-end particles (吧/嘛/呗) — cycle 14 tried but caused xhs regression
            # (seed=42: 53 → 59). Random state shift + downstream interaction. Parked.
        if config.get('shorten_paragraphs', False):
            text = shorten_paragraphs(text)

    # ── Perplexity-boosting strategies — tier-gated ──
    # Bigram substitution active in moderate+full (safe, targeted)
    if tier != 'conservative':
        bigram_strength = 0.5 if aggressive else 0.3
        if tier == 'moderate':
            bigram_strength *= 0.6
        # Route bigram substitution through the novel-register filter when
        # --style novel is active. NOVEL_BLACKLIST_CANDIDATES strips the
        # overtly colloquial / book-Chinese substitutes ('搞'/'拉高'/'业已'/
        # '早就') that break narrative register, while keeping
        # ('察觉'/'识破') that academic mode rejects.
        bigram_scene = 'novel' if style == 'novel' else scene
        text = reduce_high_freq_bigrams(text, strength=bigram_strength, scene=bigram_scene)

    # Noise + sentence randomization only at full tier — these are the operations
    # that on HC3 sometimes added spurious AI patterns to already-clean text.
    if tier == 'full' and _USE_NOISE:
        noise_density = 0.25 if aggressive else 0.15
        # Novel/fiction register: noise injection (regardless of expression
        # subset) frequently lands on prepositional or vocative sentence heads
        # ('作为...' / '人物名+verb') and reads as awkward. Lean on word
        # substitutions + transition cap + paraphrase replacement for delta
        # in novel mode instead.
        if style != 'novel':
            # Cycle 104: route academic scene through NOISE_ACADEMIC_EXPRESSIONS
            # subset (hedging / self_correction / uncertainty). Cycle 54 tried
            # this and lost -2 academic hero, but cycles 76-101 since cleaned
            # the pool of self-defeating entries — second attempt with the
            # tighter pool. Audit found 20+ filler / transition_casual /
            # personal injections in academic samples ('不瞒你说' / '说到底' /
            # '讲真' / '约莫' / '估摸着') that read off-register.
            noise_style = 'academic' if scene == 'academic' else 'general'
            text = inject_noise_expressions(text, density=noise_density, style=noise_style)
        text = randomize_sentence_lengths(text, aggressive=aggressive, seed=seed)

    # v5 P1 humanize counter-measure for stat_low_para_sent_len_cv. The
    # truncation variant (boost_para_sent_len_cv) was shelved because
    # adding a period bumps punct_density and cancels the para-CV win.
    # The merge variant lifts a uniform paragraph by combining two
    # adjacent short-medium sentences with a comma — removing one
    # period, often pushing the merged sentence over the long threshold,
    # both of which point LR away from AI. n=20 sweep at target=0.40
    # showed avg LR delta -0.95 with zero regressions.
    text = boost_para_cv_via_merge(text)

    # v5 P1.2 humanize counter-measure for paragraph_length_cv (LR coef
    # -1.99 on longform). For multi-paragraph text whose paragraph
    # length CV is below 0.60, insert a single 22-24 cn-char reflection
    # paragraph after one of the longer existing paragraphs. Skipped
    # for novel style (narrative paragraphs differ; reflective
    # interjections read off-register). n=30 by-genre sweep:
    #   novel    skipped 10/10 ✓
    #   academic fired 4/10, LR delta 0.00 (neutral)
    #   news     fired 10/10, LR avg -2.10 (3 down / 1 up / 6 same)
    text = insert_short_interjection_paragraph(text, target_cv=0.60,
                                               style=style, seed=seed)

    # v5 P1.3 humanize counter-measure for cross_para_3gram_repeat (LR
    # coef +2.24 on longform). Replaces a few CiLin-known 2-char words
    # that recur across paragraphs with scene-filtered synonyms,
    # breaking the cross-paragraph trigram repetition. n=20 sweep at
    # max_replacements=4: fired 20/20, LR delta avg -1.65, zero
    # regressions.
    text = reduce_cross_para_3gram_repeat(text, max_replacements=4,
                                          scene=scene, style=style,
                                          seed=seed)

    # Final transition cap — AI overuses 首先/然而/此外/因此 etc, detect fires
    # density > 8/1000 chars. Cap at 6 to leave margin. Preserves text that's
    # already under the threshold.
    # Long-form (novel/blog) humans use far fewer transitions (d=0.92 gap vs
    # AI). Drop cap target on long text so novel humanize approaches human 2.4
    # density instead of staying at AI's 4.4 baseline.
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    trans_target = 3.0 if cn_chars >= 1500 else 6.0
    text = cap_transition_density(text, target=trans_target)

    # Novel/fiction register: strip overused AI-style intensifiers.
    # Spot-check on 20 \u7384\u5e7b samples showed \u300c\u5341\u5206/\u975e\u5e38/\u6781\u5176/\u683c\u5916/\u6781\u4e3a/\u6781\u5ea6/
    # \u5c24\u4e3a/\u9887\u4e3a\u300d+ adj appears ~25-28 times per 20-sample batch as an AI
    # mannerism. Negative lookaheads exclude the two false positives we
    # observed: '\u5341\u5206\u949f' (time noun) and '\u975e\u5e38\u89c4' (adv prefix).
    # Skip '\u65e0\u6bd4' (\u53e5\u5c3e idiomatic, deletion would break clauses) and
    # '\u76f8\u5f53' (quantifier, '\u76f8\u5f53\u591a/\u76f8\u5f53\u957f' \u2260 intensifier).
    if style == 'novel':
        text = re.sub(r'\u5341\u5206(?![\u949f\u4e4b])', '', text)
        text = re.sub(r'\u975e\u5e38(?![\u89c4])', '', text)
        text = re.sub(r'\u6781\u5176', '', text)
        text = re.sub(r'\u683c\u5916', '', text)
        text = re.sub(r'\u6781\u4e3a', '', text)
        text = re.sub(r'\u6781\u5ea6', '', text)
        text = re.sub(r'\u5c24\u4e3a', '', text)
        text = re.sub(r'\u9887\u4e3a', '', text)

    # Clean up artifacts
    text = re.sub(r'[，,]{2,}', '，', text)  # Remove double commas
    text = re.sub(r'[。]{2,}', '。', text)    # Remove double periods
    text = re.sub(r'\n{3,}', '\n\n', text)    # Normalize newlines
    text = re.sub(r'，。', '。', text)          # Remove comma before period
    text = re.sub(r'。，', '。', text)          # Remove period before comma
    text = re.sub(r'(版本(?:显著|明显|可观))，(提升了)', r'\1\2', text)
    
    # ── Final verification loop (stats-optimized) ──
    # If perplexity is still too low, do a targeted second pass on worst sentences
    if _USE_STATS and ngram_analyze:
        stats = ngram_analyze(text)
        ppl = stats.get('perplexity', 0)
        # Threshold: if perplexity is in the "too smooth" zone, try to improve.
        # D-5 (cycle 31): raised 200 → 350 to cover the typical humanized-output
        # perplexity range (~250-300) where indicators still fire.
        if 0 < ppl < 350 and len(text) >= 100:
            sentences = re.split(r'([。！？])', text)
            # Score each sentence
            sent_scores = []
            for i in range(0, len(sentences) - 1, 2):
                s = sentences[i]
                if len(s.strip()) < 5:
                    continue
                s_stats = ngram_analyze(s)
                sent_scores.append((i, s_stats.get('perplexity', 0)))
            
            if sent_scores:
                # Sort by perplexity ascending (worst = most predictable first)
                sent_scores.sort(key=lambda x: x[1])
                # Try to improve the worst 20% (at most 5 sentences)
                n_fix = min(5, max(1, len(sent_scores) // 5))
                
                # Use a different random seed for the second pass
                if seed is not None:
                    random.seed(seed + 1)
                
                for idx, _ in sent_scores[:n_fix]:
                    sent = sentences[idx]
                    # Try each replacement on this sentence
                    sorted_phrases = sorted(PLAIN_REPLACEMENTS.keys(), key=len, reverse=True)
                    for phrase in sorted_phrases:
                        if phrase in sent:
                            alternatives = PLAIN_REPLACEMENTS[phrase]
                            if isinstance(alternatives, str):
                                alternatives = [alternatives]
                            best = pick_best_replacement(sent, phrase, alternatives)
                            sentences[idx] = sent.replace(phrase, best, 1)
                            break  # one fix per sentence to avoid over-rewriting
                
                text = ''.join(sentences)

    
    return text.strip()

# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description='中文 AI 文本人性化 v2.0')
    parser.add_argument('file', nargs='?', help='输入文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--scene', default='general',
                       choices=['general', 'social', 'tech', 'formal', 'chat'],
                       help='场景 (default: general)')
    parser.add_argument('--style', help='写作风格 (调用 style_cn.py)')
    parser.add_argument('-a', '--aggressive', action='store_true', help='激进模式')
    parser.add_argument('--seed', type=int, help='随机种子（可复现）')
    parser.add_argument('--best-of-n', type=int, default=DEFAULT_BEST_OF_N, metavar='N',
                        help=f'运行 N 次 humanize 取 LR 分数最低的那次（默认 {DEFAULT_BEST_OF_N}，N 倍延迟，0 关闭）')
    parser.add_argument('--debug-best-of-n', action='store_true',
                       help='打印 best-of-n 每个候选的 LR scene、分数和主要贡献（stderr）')
    parser.add_argument('--score-mode', default='lr', choices=['lr', 'fused', 'lr+rule'],
                       help='best-of-n 排序方式：lr=scene-aware LR；fused=0.8*LR+0.2*rule；lr+rule=LR 优先、rule 打破平局')
    parser.add_argument('--secondary-weight', type=float, default=DEFAULT_SECONDARY_WEIGHT,
                       help=f'best-of-n secondary signal 权重（默认 {DEFAULT_SECONDARY_WEIGHT}，0 关闭）')
    parser.add_argument('--no-stats', action='store_true',
                       help='跳过统计优化（困惑度反馈），回退到纯规则替换')
    parser.add_argument('--no-noise', action='store_true',
                       help='跳过噪声策略（句长随机化 + 噪声表达插入）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（= --no-stats --no-noise），只跑短语替换 + 结构清理')
    parser.add_argument('--cilin', action='store_true',
                       help='用 CiLin 同义词词林扩展候选（~40K 词 vs 手工 200 词）')

    args = parser.parse_args()

    # Toggle stats optimization
    global _USE_STATS
    _USE_STATS = not (args.no_stats or args.quick)

    # Toggle noise strategies
    global _USE_NOISE
    _USE_NOISE = not (args.no_noise or args.quick)

    # Toggle CiLin expansion
    global _USE_CILIN
    _USE_CILIN = args.cilin
    
    # Read input
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f'错误: 文件未找到 {args.file}', file=sys.stderr)
            sys.exit(1)
    else:
        text = sys.stdin.read()
    
    if not text.strip():
        print('错误: 输入为空', file=sys.stderr)
        sys.exit(1)
    
    # Humanize
    result = humanize(text, args.scene, args.aggressive, args.seed,
                       best_of_n=args.best_of_n, style=args.style,
                       debug_best_of_n=args.debug_best_of_n,
                       score_mode=args.score_mode,
                       secondary_weight=args.secondary_weight)
    
    # Apply style if specified
    if args.style:
        try:
            from style_cn import apply_style
        except ImportError:
            try:
                from scripts.style_cn import apply_style
            except ImportError:
                apply_style = None

        if apply_style:
            result = apply_style(result, args.style, humanize_first=False, seed=args.seed)
        else:
            print('警告: 未找到风格转换模块', file=sys.stderr)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        style_info = f' (风格: {args.style})' if args.style else ''
        scene_info = f' (场景: {args.scene})'
        print(f'✓ 已保存到 {args.output}{scene_info}{style_info}')
    else:
        print(result)

if __name__ == '__main__':
    main()
