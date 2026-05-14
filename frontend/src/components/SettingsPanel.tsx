import { PlugZap, Save } from 'lucide-react'
import type { ProviderName, SettingsView } from '../types'

interface Props {
  settings: SettingsView | null
  draft: Record<string, string | boolean>
  testResult: string
  onDraft: (key: string, value: string | boolean) => void
  onSave: () => void
  onTest: () => void
}

export function SettingsPanel({ settings, draft, testResult, onDraft, onSave, onTest }: Props) {
  if (!settings) return <div className="empty-state">设置加载中</div>
  const provider = (draft.provider || settings.provider) as ProviderName
  const isOpenAI = provider === 'openai'
  const providerLabel = isOpenAI ? 'OpenAI' : 'Anthropic'
  const modelKey = isOpenAI ? 'openai_model' : 'anthropic_model'
  const baseUrlKey = isOpenAI ? 'openai_base_url' : 'anthropic_base_url'
  const apiKeyKey = isOpenAI ? 'openai_api_key' : 'anthropic_api_key'
  const apiKeySet = isOpenAI ? settings.openai_api_key_set : settings.anthropic_api_key_set
  const apiKeySource = isOpenAI ? settings.openai_api_key_source : settings.anthropic_api_key_source
  const baseUrl = String(draft[baseUrlKey] || (isOpenAI ? settings.openai_base_url : settings.anthropic_base_url))
  const requestUrl = isOpenAI
    ? `${baseUrl.replace(/\/$/, '')}/chat/completions`
    : `${baseUrl.replace(/\/$/, '')}/v1/messages`
  const providerWarning = isOpenAI
    ? !baseUrl.replace(/\/$/, '').endsWith('/v1')
      ? 'OpenAI Chat Completions 通常需要 base URL 以 /v1 结尾。'
      : ''
    : requestUrl.includes('/v1/v1')
      ? 'Anthropic 请求 URL 中出现 /v1/v1，请检查 base URL 是否重复包含 /v1。'
      : ''
  const liveWarnings = [providerWarning].filter((warning): warning is string => Boolean(warning))

  return (
    <section className="settings-panel">
      <div className="section-title">设置</div>
      <div className="privacy-note">
        目前处于演示模式，云端演示可填写自己的 API Key。浏览器设置优先于项目 env，并只保存在你自己的浏览器；当使用自定义 Key 发起改写时，服务器不会保存输入和输出正文。刷新页面仍保留，清除浏览器数据后会清空。
      </div>
      <label>
        SDK
        <select value={provider} onChange={(event) => onDraft('provider', event.target.value)}>
          <option value="openai">OpenAI SDK</option>
          <option value="anthropic">Anthropic SDK</option>
        </select>
      </label>
      <label>
        {providerLabel} Model
        <input value={String(draft[modelKey] ?? '')} onChange={(event) => onDraft(modelKey, event.target.value)} />
      </label>
      <label>
        {providerLabel} Base URL
        <input value={String(draft[baseUrlKey] ?? '')} onChange={(event) => onDraft(baseUrlKey, event.target.value)} />
      </label>
      <label>
        {providerLabel} API Key
        <input
          type="password"
          placeholder={apiKeySet ? `已配置 · ${apiKeySource}` : '可选：使用你的浏览器本地 API Key'}
          onChange={(event) => onDraft(apiKeyKey, event.target.value)}
        />
      </label>
      <div className="url-preview">{requestUrl}</div>
      {Array.from(new Set(liveWarnings)).map((warning) => (
        <div className="warning" key={warning}>
          {warning}
        </div>
      ))}
      <label className="stream-card">
        <input
          type="checkbox"
          checked={Boolean(draft.stream)}
          onChange={(event) => onDraft('stream', event.target.checked)}
        />
        <span className="stream-visual" aria-hidden="true">
          <span className="stream-dot" />
        </span>
        <span className="stream-copy">
          <strong>流式输出</strong>
          <small>{Boolean(draft.stream) ? '实时显示生成过程' : '等待完成后一次性显示'}</small>
        </span>
      </label>
      <div className="settings-actions">
        <button className="primary-button" type="button" onClick={onSave}>
          <Save size={16} /> 保存
        </button>
        <button className="secondary-button" type="button" onClick={onTest}>
          <PlugZap size={16} /> 测试
        </button>
      </div>
      {testResult && <div className="test-result">{testResult}</div>}
    </section>
  )
}
