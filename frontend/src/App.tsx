import { useEffect, useMemo, useState } from 'react'
import { Eraser, FileClock, RotateCcw, Send, Settings, Star, WandSparkles } from 'lucide-react'
import { DiffView } from './components/DiffView'
import { HistoryList } from './components/HistoryList'
import { SettingsPanel } from './components/SettingsPanel'
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

const initialText =
  '自然语言处理是一门融合了计算机科学、数学与语言学的综合性学科，而文本分类作为其重要的研究方向，在大数据时代具有显著意义——文本型数据凭借其存储轻便、描述力强的特点，成为最常见的电子数据类型之一。如何在海量文本中高效且准确地提取所需信息，已成为一个现实而迫切的问题。本文主要基于自然语言处理中的文本数据处理方法与机器学习理论，对文本分类模型的实现展开研究。实验部分采用Python进行编程，围绕以下内容展开：首先，综合阐述文本分类的相关理论与发展现状，介绍文本处理流程，使用TF-IDF进行特征提取，并对比jieba、SnowNLP、pkuseg三种分词工具，最终选定pkuseg作为本文数据的最优切词方案。其次，通过加权F1值、准确率等指标评估算法性能，除基础的KNN、决策树、支持向量机外，还引入了随机森林、GBDT、XGBoost、LightGBM等集成学习方法。实验表明，集成模型整体表现优于基础模型。最后，采用Stacking融合策略分别对四个基础模型与四个集成模型进行集成，结果发现融合后的模型多数情况下优于单个模型，整体体现出Stacking策略的优越性。其中，以梯度提升树作为次级学习器的Stacking集成模型效果最佳，其加权F1值达xxxx，准确率约为xxxx%，从而验证了Stacking集成算法在文本分类中的有效性与准确性。'

function countText(text: string) {
  return Array.from(text).filter((char) => char.trim()).length
}

function clampBestOfN(value: number) {
  if (!Number.isFinite(value)) return 10
  return Math.max(0, Math.min(20, Math.round(value)))
}

function parseSeed(value: string) {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const seed = Number(trimmed)
  return Number.isFinite(seed) ? Math.trunc(seed) : undefined
}

export default function App() {
  const [settings, setSettings] = useState<SettingsView | null>(null)
  const [draft, setDraft] = useState<Record<string, string | boolean>>({})
  const [text, setText] = useState(initialText)
  const [platform, setPlatform] = useState<PlatformName>('weipu')
  const [iterations, setIterations] = useState(1)
  const [nlpEnabled, setNlpEnabled] = useState(false)
  const [nlpMode, setNlpMode] = useState<NlpMode>('manual')
  const [nlpStyle, setNlpStyle] = useState<NlpStyle>('academic')
  const [nlpBestOfN, setNlpBestOfN] = useState(10)
  const [nlpSeed, setNlpSeed] = useState('')
  const [nlpAggressive, setNlpAggressive] = useState(false)
  const [activeTab, setActiveTab] = useState<'output' | 'diff' | 'history' | 'settings'>('output')
  const [output, setOutput] = useState('')
  const [rawOutput, setRawOutput] = useState('')
  const [diff, setDiff] = useState<DiffSpan[]>([])
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('就绪')
  const [testResult, setTestResult] = useState('')
  const [showLengthModal, setShowLengthModal] = useState(false)

  const charCount = useMemo(() => countText(text), [text])

  useEffect(() => {
    void refresh()
  }, [])

  async function refresh() {
    const nextSettings = await api.getSettings()
    setSettings(nextSettings)
    setDraft({
      provider: nextSettings.provider,
      openai_model: nextSettings.openai_model,
      anthropic_model: nextSettings.anthropic_model,
      openai_base_url: nextSettings.openai_base_url,
      anthropic_base_url: nextSettings.anthropic_base_url,
      stream: nextSettings.stream,
    })
    setNlpEnabled(nextSettings.nlp_enabled)
    setNlpMode(nextSettings.nlp_mode === 'off' ? 'manual' : nextSettings.nlp_mode)
    setNlpStyle(nextSettings.nlp_style)
    setHistory(await api.history())
  }

  function updateDraft(key: string, value: string | boolean) {
    setDraft((current) => ({ ...current, [key]: value }))
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
    return {
      text,
      platform,
      iterations,
      provider: draft.provider as ProviderName | undefined,
      // The primary rewrite action should always use SSE progress.
      // Without this, a slow LLM call leaves the UI stuck at “启动工作流” until the request finishes.
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
          if (event === 'node_started') setStatus(`节点：${String(d.node)}`)
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
          if (event === 'error') throw new Error(String(d.error ?? '流式请求失败'))
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
    setShowLengthModal(false)
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
        if (event === 'node_started') setStatus(`节点：${String(d.node)}`)
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
        if (event === 'error') throw new Error(String(d.error ?? '学术降痕失败'))
      })
      setDiff(currentDiff)
      setHistory(await api.history())
    } catch (error) {
      setStatus(error instanceof Error ? error.message : '学术降痕失败')
    } finally {
      setBusy(false)
    }
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

  async function deleteHistory(id: number) {
    await api.deleteHistory(id)
    setHistory(await api.history())
  }

  return (
    <div className="app-shell">
      <WindCanvas />
      <header className="topbar animate-fade-in-up" style={{ animationDelay: '0.1s', opacity: 0 }}>
        <div className="brand-block">
          <span className="brand-mark" aria-hidden>
            <Star size={20} fill="currentColor" />
          </span>
          <div>
            <h1>AI-Cleaner</h1>
            <p>学术改写工作台 · 流式输出 · 差异可视化</p>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="status-label">Workspace live</span>
          <div className="status">{status}</div>
        </div>
      </header>

      <main className="workspace">
        <section className="input-panel animate-fade-in-up" style={{ animationDelay: '0.2s', opacity: 0 }}>
          <div className="section-title">输入</div>
          <textarea value={text} onChange={(event) => setText(event.target.value)} />
          <div className="input-meta">
            <span className={charCount < 300 || charCount > 1200 ? 'count warn-count' : 'count'}>
              {charCount} 字
            </span>
            <select value={platform} onChange={(event) => setPlatform(event.target.value as PlatformName)}>
              <option value="weipu">维普</option>
              <option value="paperyy">PaperYY</option>
              <option value="paperpass">PaperPass</option>
              <option value="zhuque">腾讯朱雀</option>
            </select>
            <label className="stepper">
              迭代
              <input
                type="number"
                min={1}
                max={5}
                value={iterations}
                onChange={(event) => setIterations(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="nlp-controls">
            <label className="inline-check">
              <input
                type="checkbox"
                checked={nlpEnabled}
                onChange={(event) => setNlpEnabled(event.target.checked)}
              />
              改写后追加学术降痕
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
                <option value="academic">学术论文（主）</option>
                <option value="general">通用文本</option>
                <option value="long_blog">自媒体/长篇博客</option>
              </select>
            </label>
            <label>
              候选
              <input
                type="number"
                min={0}
                max={20}
                value={nlpBestOfN}
                onChange={(event) => setNlpBestOfN(clampBestOfN(Number(event.target.value)))}
              />
            </label>
            <label>
              Seed
              <input
                inputMode="numeric"
                placeholder="随机"
                value={nlpSeed}
                onChange={(event) => setNlpSeed(event.target.value)}
              />
            </label>
            <label className="inline-check">
              <input
                type="checkbox"
                checked={nlpAggressive}
                onChange={(event) => setNlpAggressive(event.target.checked)}
              />
              激进模式
            </label>
          </div>
          <div className="primary-actions">
            <button className="primary-button" disabled={busy || !text.trim()} type="button" onClick={() => void runRewrite()}>
              <Send size={16} /> 改写
            </button>
            <button className="secondary-button" disabled={busy || !text.trim()} type="button" onClick={() => void runNlpOnly()}>
              <WandSparkles size={16} /> 仅学术降痕
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
          <nav className="tabs">
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
            <HistoryList items={history} onOpen={(id) => void openHistory(id)} onDelete={(id) => void deleteHistory(id)} />
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
        <div className="modal-backdrop animate-fade-in-overlay" role="dialog" aria-modal="true">
          <div className="modal animate-slide-up-overlay">
            <h2>字数提示</h2>
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
    </div>
  )
}
