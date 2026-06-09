import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Check,
  Copy,
  Eraser,
  FileClock,
  RotateCcw,
  Send,
  Settings,
  Trash2,
  WandSparkles,
} from 'lucide-react'
import logoUrl from './assets/logo.png'
import { DiffView } from './components/DiffView'
import { HistoryList } from './components/HistoryList'
import { SettingsPanel } from './components/SettingsPanel'
import { Tooltip } from './components/Tooltip'
import { TypewriterOutput } from './components/TypewriterOutput'
import { WindCanvas } from './components/WindCanvas'
import { api, streamNlpRewrite, streamRewrite } from './lib/api'
import type {
  DiffSpan,
  HistoryItem,
  NlpMode,
  NlpStyle,
  PlatformName,
  ProviderName,
  RewriteResponse,
  SettingsView,
} from './types'

type LastAction = 'agent' | 'nlp'

type UiPreferences = {
  lastAction: LastAction
  platform: PlatformName
  iterations: number
  nlpEnabled: boolean
  nlpMode: NlpMode
  nlpStyle: NlpStyle
  nlpBestOfN: number
  nlpSeed: string
  nlpAggressive: boolean
}

const UI_PREFERENCES_KEY = 'ai-cleaner-ui-preferences'

const defaultPreferences: UiPreferences = {
  lastAction: 'agent',
  platform: 'weipu',
  iterations: 2,
  nlpEnabled: false,
  nlpMode: 'manual',
  nlpStyle: 'academic',
  nlpBestOfN: 10,
  nlpSeed: '',
  nlpAggressive: false,
}

const initialText =
  '自然语言处理是一门融合了计算机科学、数学与语言学的综合性学科，而文本分类作为其重要的研究方向，在大数据时代具有显著意义——文本型数据凭借其存储轻便、描述力强的特点，成为最常见的电子数据类型之一。如何在海量文本中高效且准确地提取所需信息，已成为一个现实而迫切的问题。本文主要基于自然语言处理中的文本数据处理方法与机器学习理论，对文本分类模型的实现展开研究。实验部分采用Python进行编程，围绕以下内容展开：首先，综合阐述文本分类的相关理论与发展现状，介绍文本处理流程，使用TF-IDF进行特征提取，并对比jieba、SnowNLP、pkuseg三种分词工具，最终选定pkuseg作为本文数据的最优切词方案。其次，通过加权F1值、准确率等指标评估算法性能，除基础的KNN、决策树、支持向量机外，还引入了随机森林、GBDT、XGBoost、LightGBM等集成学习方法。实验表明，集成模型整体表现优于基础模型。最后，采用Stacking融合策略分别对四个基础模型与四个集成模型进行集成，结果发现融合后的模型多数情况下优于单个模型，整体体现出Stacking策略的优越性。其中，以梯度提升树作为次级学习器的Stacking集成模型效果最佳，其加权F1值达xxxx，准确率约为xxxx%，从而验证了Stacking集成算法在文本分类中的有效性与准确性。\n本文进一步将研究视角延伸至深度学习领域，探索基于神经网络的文本表示与分类模型。不同于传统机器学习方法依赖人工特征工程，深度学习模型能够自动从原始文本中学习层次化的特征表示，从而在复杂语义理解任务中展现出更强的能力。实验部分在先前预处理流程基础上，引入词向量（Word2Vec与BERT）作为文本的分布式表征，并构建了包括TextCNN、BiLSTM及其注意力机制变体在内的多种深度学习架构。通过对比分析发现，基于BERT预训练模型的微调方案在本文所使用的数据集上取得了最优性能，其F1值达到xxxx，准确率为xxxxx%，显著优于传统机器学习模型及浅层神经网络。同时，本文设计了一组对比实验，考察不同文本长度、类别不平衡程度以及数据增强策略对模型泛化能力的影响。结果表明，结合随机裁剪与回译的数据增强方法能够有效缓解过拟合问题，尤其在样本量较少的类别中，模型性能提升最为明显。本研究不仅验证了深度学习方法在中文文本分类任务中的有效性，也为后续面向特定领域的文本分析系统开发提供了可借鉴的技术路径与参数调优经验。'

function countText(text: string) {
  return Array.from(text).filter((char) => char.trim()).length
}

function clampBestOfN(value: number) {
  if (!Number.isFinite(value)) return 10
  return Math.max(0, Math.min(20, Math.round(value)))
}

function clampIterations(value: number) {
  if (!Number.isFinite(value)) return 2
  return Math.max(1, Math.min(5, Math.round(value)))
}

function parseSeed(value: string) {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const seed = Number(trimmed)
  return Number.isFinite(seed) ? Math.trunc(seed) : undefined
}

function useFocusTrap(open: boolean, containerRef: React.RefObject<HTMLElement | null>) {
  const previouslyFocused = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!open) return
    previouslyFocused.current = document.activeElement as HTMLElement

    const container = containerRef.current
    if (!container) return

    const focusFirst = () => {
      const first = container.querySelector<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')
      if (first) first.focus()
    }
    requestAnimationFrame(focusFirst)

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.preventDefault()
        return
      }
      if (e.key !== 'Tab') return

      const el = containerRef.current
      if (!el) return
      const focusable = el.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      )
      if (!focusable.length) return

      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault()
          last.focus()
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault()
          first.focus()
        }
      }
    }

    container.addEventListener('keydown', onKeyDown)
    return () => {
      container.removeEventListener('keydown', onKeyDown)
      if (previouslyFocused.current && previouslyFocused.current.focus) {
        previouslyFocused.current.focus()
      }
    }
  }, [open, containerRef])
}

function readUiPreferences(): UiPreferences {
  try {
    const saved = window.localStorage.getItem(UI_PREFERENCES_KEY)
    if (!saved) return defaultPreferences
    const parsed = JSON.parse(saved) as Partial<UiPreferences>
    return {
      lastAction: parsed.lastAction === 'nlp' ? 'nlp' : 'agent',
      platform: parsed.platform ?? defaultPreferences.platform,
      iterations: clampIterations(Number(parsed.iterations ?? defaultPreferences.iterations)),
      nlpEnabled: Boolean(parsed.nlpEnabled ?? defaultPreferences.nlpEnabled),
      nlpMode: parsed.nlpMode === 'auto' ? 'auto' : 'manual',
      nlpStyle: parsed.nlpStyle ?? defaultPreferences.nlpStyle,
      nlpBestOfN: clampBestOfN(Number(parsed.nlpBestOfN ?? defaultPreferences.nlpBestOfN)),
      nlpSeed: typeof parsed.nlpSeed === 'string' ? parsed.nlpSeed : defaultPreferences.nlpSeed,
      nlpAggressive: Boolean(parsed.nlpAggressive ?? defaultPreferences.nlpAggressive),
    }
  } catch {
    return defaultPreferences
  }
}

function persistUiPreferences(preferences: UiPreferences) {
  try {
    window.localStorage.setItem(UI_PREFERENCES_KEY, JSON.stringify(preferences))
  } catch {
    // Browser storage can be unavailable in private mode.
  }
}

export default function App() {
  const [settings, setSettings] = useState<SettingsView | null>(null)
  const [draft, setDraft] = useState<Record<string, string | boolean>>({})
  const [text, setText] = useState(initialText)
  const [platform, setPlatform] = useState<PlatformName>(defaultPreferences.platform)
  const [iterations, setIterations] = useState(defaultPreferences.iterations)
  const [nlpEnabled, setNlpEnabled] = useState(defaultPreferences.nlpEnabled)
  const [nlpMode, setNlpMode] = useState<NlpMode>(defaultPreferences.nlpMode)
  const [nlpStyle, setNlpStyle] = useState<NlpStyle>(defaultPreferences.nlpStyle)
  const [nlpBestOfN, setNlpBestOfN] = useState(defaultPreferences.nlpBestOfN)
  const [nlpSeed, setNlpSeed] = useState(defaultPreferences.nlpSeed)
  const [nlpAggressive, setNlpAggressive] = useState(defaultPreferences.nlpAggressive)
  const [activeTab, setActiveTab] = useState<'output' | 'diff' | 'history' | 'settings'>('output')
  const [output, setOutput] = useState('')
  const [rawOutput, setRawOutput] = useState('')
  const [diff, setDiff] = useState<DiffSpan[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('就绪')
  const [testResult, setTestResult] = useState('')
  const [copied, setCopied] = useState(false)
  const [showLengthModal, setShowLengthModal] = useState(false)
  const [lastAction, setLastAction] = useState<LastAction>(defaultPreferences.lastAction)
  const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null)
  const [canvasFrozen, setCanvasFrozen] = useState(false)

  const lengthModalRef = useRef<HTMLDivElement>(null)
  const deleteModalRef = useRef<HTMLDivElement>(null)
  useFocusTrap(showLengthModal, lengthModalRef)
  useFocusTrap(pendingDeleteId !== null, deleteModalRef)

  const charCount = useMemo(() => countText(text), [text])

  useEffect(() => {
    function onFrozen(e: Event) {
      const detail = (e as CustomEvent).detail
      console.warn(`[WindCanvas] Animation frozen due to low FPS (${Math.round(detail.avgFps)}fps avg)`)
      setCanvasFrozen(true)
    }
    document.addEventListener('windcanvas-frozen', onFrozen)
    return () => document.removeEventListener('windcanvas-frozen', onFrozen)
  }, [])

  useEffect(() => {
    const preferences = readUiPreferences()
    setLastAction(preferences.lastAction)
    setPlatform(preferences.platform)
    setIterations(preferences.iterations)
    setNlpEnabled(preferences.nlpEnabled)
    setNlpMode(preferences.nlpMode)
    setNlpStyle(preferences.nlpStyle)
    setNlpBestOfN(preferences.nlpBestOfN)
    setNlpSeed(preferences.nlpSeed)
    setNlpAggressive(preferences.nlpAggressive)
    void refresh(preferences)
  }, [])

  useEffect(() => {
    persistUiPreferences({
      lastAction,
      platform,
      iterations: clampIterations(iterations),
      nlpEnabled,
      nlpMode,
      nlpStyle,
      nlpBestOfN: clampBestOfN(nlpBestOfN),
      nlpSeed,
      nlpAggressive,
    })
  }, [lastAction, platform, iterations, nlpEnabled, nlpMode, nlpStyle, nlpBestOfN, nlpSeed, nlpAggressive])

  async function refresh(preferences = readUiPreferences()) {
    let nextSettings: SettingsView
    try {
      nextSettings = await api.getSettings()
    } catch (err) {
      console.error('[refresh] getSettings failed, using defaults:', err)
      nextSettings = {
        provider: 'openai',
        openai_model: '',
        anthropic_model: '',
        openai_base_url: 'https://api.openai.com/v1',
        anthropic_base_url: 'https://api.anthropic.com',
        openai_api_key_set: false,
        anthropic_api_key_set: false,
        openai_api_key_source: 'missing',
        anthropic_api_key_source: 'missing',
        stream: false,
        nlp_enabled: false,
        nlp_mode: 'manual',
        nlp_style: 'academic',
        openai_request_url: '',
        anthropic_request_url: '',
        warnings: [],
      }
      setStatus('无法连接后端，请确认服务已启动')
    }
    setSettings(nextSettings)
    let localOverrides: Record<string, string | boolean> = {}
    try {
      const savedDraft = window.localStorage.getItem('ai-cleaner-user-settings')
      if (savedDraft) localOverrides = JSON.parse(savedDraft) as Record<string, string | boolean>
    } catch {
      localOverrides = {}
    }
    setDraft({
      provider: nextSettings.provider,
      openai_model: nextSettings.openai_model,
      anthropic_model: nextSettings.anthropic_model,
      openai_base_url: nextSettings.openai_base_url,
      anthropic_base_url: nextSettings.anthropic_base_url,
      stream: nextSettings.stream,
      ...localOverrides,
    })
    setNlpEnabled(preferences.nlpEnabled)
    setNlpMode(preferences.nlpMode)
    setNlpStyle(preferences.nlpStyle)
    try {
      setHistory(await api.history())
    } catch (err) {
      console.error('[refresh] history failed:', err)
    }
  }

  function rememberAction(action: LastAction) {
    setLastAction(action)
  }

  function updatePlatform(nextPlatform: PlatformName) {
    setPlatform(nextPlatform)
    if (nextPlatform === 'novel') {
      setNlpStyle('novel')
    } else if (nlpStyle === 'novel') {
      setNlpStyle('academic')
    }
  }

  function updateDraft(key: string, value: string | boolean) {
    setDraft((current) => {
      const next = { ...current, [key]: value }
      try {
        window.localStorage.setItem('ai-cleaner-user-settings', JSON.stringify(next))
      } catch {
        // Browser storage can be unavailable in private mode.
      }
      return next
    })
  }

  async function saveSettings() {
    setStatus('保存设置中')
    const saved = await api.saveSettings({
      ...draft,
      nlp_enabled: nlpEnabled,
      nlp_mode: nlpMode,
      nlp_style: nlpStyle,
    })
    setSettings(saved)
    setStatus('设置已保存')
  }

  async function testProvider() {
    setStatus('测试 SDK 连接')
    const provider = String(draft.provider ?? settings?.provider ?? 'openai')
    const result = await api.testSettings({
      provider,
      model: provider === 'openai' ? draft.openai_model : draft.anthropic_model,
      base_url: provider === 'openai' ? draft.openai_base_url : draft.anthropic_base_url,
      api_key: provider === 'openai' ? draft.openai_api_key : draft.anthropic_api_key,
    })
    setTestResult(
      result.ok
        ? `OK · ${result.latency_ms}ms · ${result.request_url} · ${result.response_preview ?? ''}`
        : `失败 · ${result.request_url} · ${result.error ?? '未知错误'}`,
    )
    setStatus(result.ok ? '连接测试通过' : '连接测试失败')
  }

  function buildPayload() {
    const provider = String(draft.provider ?? settings?.provider ?? 'openai') as ProviderName
    const customApiKey = provider === 'openai'
      ? String(draft.openai_api_key ?? '').trim()
      : String(draft.anthropic_api_key ?? '').trim()
    return {
      text,
      platform,
      iterations: clampIterations(iterations),
      provider,
      model: provider === 'openai' ? String(draft.openai_model ?? '') : String(draft.anthropic_model ?? ''),
      base_url: provider === 'openai' ? String(draft.openai_base_url ?? '') : String(draft.anthropic_base_url ?? ''),
      ...(customApiKey ? { api_key: customApiKey } : {}),
      stream: true,
      nlp_enabled: nlpEnabled,
      nlp_mode: nlpMode,
      nlp_style: nlpStyle,
      nlp_aggressive: nlpAggressive,
      nlp_best_of_n: clampBestOfN(nlpBestOfN),
      nlp_seed: parseSeed(nlpSeed),
    }
  }

  async function runRewrite(force = false) {
    if (!force && (charCount < 300 || charCount > 1200)) {
      setShowLengthModal(true)
      return
    }
    setShowLengthModal(false)
    rememberAction('agent')
    setBusy(true)
    setOutput('')
    setRawOutput('')
    setDiff([])
    setActiveTab('output')
    setStatus('启动工作流')
    try {
      const payload = buildPayload()
      if (payload.stream) {
        let streamed = ''
        let currentDiff: DiffSpan[] = []
        await streamRewrite(payload, (event, data) => {
          const d = data as Record<string, unknown>
          if (event === 'stream_started') setStatus(`正在进行：已连接 · ${String(d.stream_id ?? '')}`)
          if (event === 'node_started') setStatus(`正在进行：${String(d.node)}`)
          if (event === 'nlp_stream_started') {
            streamed = ''
            setOutput('')
            setStatus(`学术降痕输出 · ${String(d.style ?? 'academic')}`)
          }
          if (event === 'llm_delta') {
            streamed += String(d.delta ?? '')
            setOutput(streamed)
          }
          if (event === 'nlp_delta') {
            streamed += String(d.delta ?? '')
            setOutput(streamed)
          }
          if (event === 'iteration_result') {
            setOutput(String(d.text ?? streamed))
          }
          if (event === 'nlp_result') {
            setOutput(String(d.text ?? ''))
          }
          if (event === 'diff_ready') {
            currentDiff = d.diff as DiffSpan[]
            setDiff(currentDiff)
          }
          if (event === 'done') {
            setOutput(String(d.text ?? ''))
            setRawOutput(String(d.raw_output ?? ''))
            setStatus(`完成 · #${String(d.id)}`)
          }
          if (event === 'error') throw new Error(`${String(d.error ?? '流式请求失败')}${d.stream_id ? ` · ${String(d.stream_id)}` : ''}`)
        })
        setDiff(currentDiff)
      } else {
        const result = await api.rewrite(payload)
        applyResult(result)
        setStatus(`完成 · #${result.id}`)
      }
      setHistory(await api.history())
    } catch (error) {
      setStatus(error instanceof Error ? error.message : '请求失败')
    } finally {
      setBusy(false)
    }
  }

  async function runNlpOnly() {
    rememberAction('nlp')
    setBusy(true)
    setOutput('')
    setRawOutput('')
    setDiff([])
    setActiveTab('output')
    setStatus('启动学术降痕')
    try {
      const payload = {
        text,
        platform,
        nlp_mode: nlpMode,
        nlp_style: nlpStyle,
        aggressive: nlpAggressive,
        best_of_n: clampBestOfN(nlpBestOfN),
        seed: parseSeed(nlpSeed),
      }
      let streamed = ''
      let currentDiff: DiffSpan[] = []
      await streamNlpRewrite(payload, (event, data) => {
        const d = data as Record<string, unknown>
        if (event === 'node_started') setStatus(`正在进行：${String(d.node)}`)
        if (event === 'nlp_stream_started') {
          streamed = ''
          setOutput('')
          setStatus(`学术降痕输出 · ${String(d.style ?? 'academic')}`)
        }
        if (event === 'nlp_delta') {
          streamed += String(d.delta ?? '')
          setOutput(streamed)
        }
        if (event === 'nlp_result') {
          setOutput(String(d.text ?? streamed))
        }
        if (event === 'diff_ready') {
          currentDiff = d.diff as DiffSpan[]
          setDiff(currentDiff)
        }
        if (event === 'done') {
          setOutput(String(d.text ?? streamed))
          setRawOutput(String(d.raw_output ?? ''))
          setStatus(`学术降痕完成 · ${String(d.nlp_style ?? nlpStyle)} · #${String(d.id)}`)
        }
        if (event === 'error') throw new Error(`${String(d.error ?? '学术降痕失败')}${d.stream_id ? ` · ${String(d.stream_id)}` : ''}`)
      })
      setDiff(currentDiff)
      setHistory(await api.history())
    } catch (error) {
      setStatus(error instanceof Error ? error.message : '学术降痕失败')
    } finally {
      setBusy(false)
    }
  }

  const activeTabIndex = { output: 0, diff: 1, history: 2, settings: 3 }[activeTab]

  async function copyOutput() {
    if (!output) return
    await navigator.clipboard.writeText(output)
    setCopied(true)
    setStatus('已复制到剪贴板')
    window.setTimeout(() => setCopied(false), 1400)
  }

  function applyResult(result: RewriteResponse) {
    setOutput(result.rewritten_text)
    setRawOutput(result.raw_output)
    setDiff(result.diff)
    setPlatform(result.platform)
  }

  async function openHistory(id: number) {
    const result = await api.historyDetail(id)
    applyResult(result)
    setText(result.original_text)
    setActiveTab('diff')
  }

  function requestDeleteHistory(id: number) {
    setPendingDeleteId(id)
  }

  async function confirmDelete() {
    if (pendingDeleteId == null) return
    const id = pendingDeleteId
    setPendingDeleteId(null)
    await api.deleteHistory(id)
    setHistory(await api.history())
  }

  function dismissDelete() {
    setPendingDeleteId(null)
  }

  return (
    <div className="app-shell">
      <WindCanvas />
      <header className="topbar animate-fade-in-up" style={{ animationDelay: '0.1s', opacity: 0 }}>
        <div className="brand-block">
          <span className="brand-mark" aria-hidden>
            <img src={logoUrl} alt="" width={28} height={28} />
          </span>
          <div>
            <h1>AI-Cleaner</h1>
            <p>404: AI signature not found</p>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="status-label">Workspace live</span>
          <div className="status">{status}</div>
        </div>
      </header>

      {canvasFrozen && (
        <div className="canvas-frozen-banner">
          <span>背景动画已因帧率不足自动冻结，不影响使用</span>
          <button type="button" onClick={() => setCanvasFrozen(false)}>关闭</button>
        </div>
      )}

      <main className="workspace">
        <section className="input-panel animate-fade-in-up" style={{ animationDelay: '0.2s', opacity: 0 }}>
          <div className="section-title">输入</div>
          <textarea value={text} onChange={(event) => setText(event.target.value)} />
          <div className="input-meta">
            <span className={charCount < 300 || charCount > 1200 ? 'count warn-count' : 'count'}>
              {charCount} 字
            </span>
            <select value={platform} onChange={(event) => updatePlatform(event.target.value as PlatformName)}>
              <option value="weipu">学术论文 Ⅰ</option>
              <option value="paperyy">学术论文 Ⅱ</option>
              <option value="paperpass">学术论文 Ⅲ</option>
              <option value="zhuque">学术论文 Ⅳ</option>
              <option value="novel">小说 Ⅰ</option>
            </select>
            <label className="stepper compact-field">
              <span className="label-with-help">
                迭代
                <Tooltip
                  label="迭代次数说明"
                  content="次数越多，改写更深，但更慢，也更容易偏离原文。"
                />
              </span>
              <input
                type="number"
                min={1}
                max={5}
                value={iterations}
                onChange={(event) => setIterations(clampIterations(Number(event.target.value)))}
              />
            </label>
          </div>
          <div className="nlp-controls">
            <label className="switch-card">
              <input
                type="checkbox"
                checked={nlpEnabled}
                onChange={(event) => setNlpEnabled(event.target.checked)}
              />
              <span className="switch-visual" aria-hidden="true" />
              <span className="switch-copy">
                <strong>Agent 后追加 NLP</strong>
                <small>主按钮完成后再做一次本地降 AIGC</small>
              </span>
            </label>
            <label>
              模式
              <select value={nlpMode} onChange={(event) => setNlpMode(event.target.value as NlpMode)}>
                <option value="manual">手动</option>
                <option value="auto">自动分类</option>
              </select>
            </label>
            <label>
              类型
              <select
                value={nlpStyle}
                disabled={nlpMode === 'auto'}
                onChange={(event) => setNlpStyle(event.target.value as NlpStyle)}
              >
                <option value="academic">学术</option>
                <option value="general">通用文本</option>
                <option value="long_blog">自媒体/长篇博客</option>
                <option value="novel">小说/长篇叙事</option>
              </select>
            </label>
            <label>
              <span className="label-with-help">
                候选
                <Tooltip
                  label="候选数量说明"
                  content="生成多个 NLP 改写候选并选择检测特征更低的一版；数值越大通常更稳，但耗时更长。"
                />
              </span>
              <input
                type="number"
                min={0}
                max={20}
                value={nlpBestOfN}
                onChange={(event) => setNlpBestOfN(clampBestOfN(Number(event.target.value)))}
              />
            </label>
            <label>
              <span className="label-with-help">
                Seed
                <Tooltip
                  label="Seed 参数说明"
                  content="随机种子。填同一个数字会尽量复现实验结果；留空则每次随机。"
                />
              </span>
              <input
                inputMode="numeric"
                placeholder="随机"
                value={nlpSeed}
                onChange={(event) => setNlpSeed(event.target.value)}
              />
            </label>
            <label className="aggressive-card">
              <input
                type="checkbox"
                checked={nlpAggressive}
                onChange={(event) => setNlpAggressive(event.target.checked)}
              />
              <span className="aggressive-copy">
                <strong>激进模式</strong>
                <small>更强改写，适合检测压力高的文本</small>
              </span>
              <span className="aggressive-pill" aria-hidden="true">
                {nlpAggressive ? 'ON' : 'OFF'}
              </span>
            </label>
          </div>
          <div className="primary-actions">
            <button
              className={lastAction === 'agent' ? 'mode-active' : 'secondary-button'}
              disabled={busy || !text.trim()}
              type="button"
              onClick={() => void runRewrite()}
            >
              <Send size={16} /> Agent 降 AIGC
            </button>
            <button
              className={lastAction === 'nlp' ? 'mode-active' : 'secondary-button'}
              disabled={busy || !text.trim()}
              type="button"
              onClick={() => void runNlpOnly()}
            >
              <WandSparkles size={16} /> 纯 NLP 降 AIGC
            </button>
            <button className="ghost-button" type="button" onClick={() => setText(initialText)}>
              <RotateCcw size={16} /> 示例
            </button>
            <button className="ghost-button" type="button" onClick={() => setText('')}>
              <Eraser size={16} /> 清空
            </button>
          </div>
        </section>

        <section className="result-panel animate-fade-in-up" style={{ animationDelay: '0.3s', opacity: 0 }}>
          <nav className="tabs" data-active={activeTabIndex}>
            <button className={activeTab === 'output' ? 'active' : ''} onClick={() => setActiveTab('output')}>
              <Send size={15} /> 输出
            </button>
            <button className={activeTab === 'diff' ? 'active' : ''} onClick={() => setActiveTab('diff')}>
              <FileClock size={15} /> 对比
            </button>
            <button className={activeTab === 'history' ? 'active' : ''} onClick={() => setActiveTab('history')}>
              <FileClock size={15} /> 历史
            </button>
            <button className={activeTab === 'settings' ? 'active' : ''} onClick={() => setActiveTab('settings')}>
              <Settings size={15} /> 设置
            </button>
          </nav>

          {activeTab === 'output' && (
            <div className="output-pane">
              <div className="output-toolbar">
                <button className="secondary-button copy-button" disabled={!output} type="button" onClick={() => void copyOutput()}>
                  {copied ? <Check size={15} /> : <Copy size={15} />} {copied ? '已复制' : '一键复制'}
                </button>
              </div>
              <TypewriterOutput text={output} placeholder="改写结果会显示在这里" />
              {rawOutput && rawOutput !== output && (
                <details>
                  <summary>Raw output</summary>
                  <pre>{rawOutput}</pre>
                </details>
              )}
            </div>
          )}
          {activeTab === 'diff' && <DiffView spans={diff} />}
          {activeTab === 'history' && (
            <HistoryList items={history} onOpen={(id) => void openHistory(id)} onDelete={(id) => requestDeleteHistory(id)} />
          )}
          {activeTab === 'settings' && (
            <SettingsPanel
              settings={settings}
              draft={draft}
              testResult={testResult}
              onDraft={updateDraft}
              onSave={() => void saveSettings()}
              onTest={() => void testProvider()}
            />
          )}
        </section>
      </main>

      {showLengthModal && (
        <div
          className="modal-backdrop animate-fade-in-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="length-modal-title"
          ref={lengthModalRef}
          onKeyDown={(e) => { if (e.key === 'Escape') setShowLengthModal(false) }}
        >
          <div className="modal animate-slide-up-overlay">
            <h2 id="length-modal-title">字数提示</h2>
            <p>当前文本约 {charCount} 字，建议字数处于 300-1200 字之间，否则效果可能不稳定。</p>
            <div className="modal-actions">
              <button className="secondary-button" type="button" onClick={() => setShowLengthModal(false)}>
                返回编辑
              </button>
              <button className="primary-button" type="button" onClick={() => void runRewrite(true)}>
                继续改写
              </button>
            </div>
          </div>
        </div>
      )}

      {pendingDeleteId !== null && (
        <div
          className="modal-backdrop animate-fade-in-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-modal-title"
          ref={deleteModalRef}
          onKeyDown={(e) => { if (e.key === 'Escape') dismissDelete() }}
        >
          <div className="modal animate-slide-up-overlay">
            <h2 id="delete-modal-title">确认删除</h2>
            <p>此操作不可撤销，确定要删除该条历史记录吗？</p>
            <div className="modal-actions">
              <button className="secondary-button" type="button" onClick={dismissDelete}>
                取消
              </button>
              <button className="danger-button" type="button" onClick={() => void confirmDelete()}>
                <Trash2 size={16} /> 删除
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
