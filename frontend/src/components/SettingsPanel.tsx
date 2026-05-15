import { PlugZap, Save } from 'lucide-react'
import type { ProviderName } from '../types'

interface Props {
  draft: Record<string, string | boolean>
  testResult: string
  onDraft: (key: string, value: string | boolean) => void
  onSave: () => void
  onTest: () => void
}

export function SettingsPanel({ draft, testResult, onDraft, onSave, onTest }: Props) {
  const provider = (draft.provider || 'openai') as ProviderName
  const isOpenAI = provider === 'openai'
  const rememberApiKeys = Boolean(draft.remember_api_keys)
  const rememberHistory = Boolean(draft.remember_history)
  const providerLabel = isOpenAI ? 'OpenAI' : 'Anthropic'
  const modelKey = isOpenAI ? 'openai_model' : 'anthropic_model'
  const baseUrlKey = isOpenAI ? 'openai_base_url' : 'anthropic_base_url'
  const apiKeyKey = isOpenAI ? 'openai_api_key' : 'anthropic_api_key'
  const baseUrl = String(draft[baseUrlKey] || '')
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
        云端演示版会临时通过后端转发你的请求，但不会在服务器保存 API Key、Base URL、输入正文、输出结果或历史记录。
        默认情况下，API Key 和历史只留在当前页面；只有勾选下方选项并点击“保存到浏览器”后，才会写入这台设备的浏览器。公网部署请务必启用 HTTPS。
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
        <input value={baseUrl} onChange={(event) => onDraft(baseUrlKey, event.target.value)} />
      </label>
      <form onSubmit={(e) => e.preventDefault()} autoComplete="on">
        <label>
          {providerLabel} API Key
          <input
            key={apiKeyKey}
            type="password"
            value={String(draft[apiKeyKey] ?? '')}
            placeholder="默认仅当前页面有效；勾选下方选项后才会写入浏览器"
            onChange={(event) => onDraft(apiKeyKey, event.target.value)}
          />
        </label>
      </form>
      <label className="stream-card">
        <input
          type="checkbox"
          checked={rememberApiKeys}
          onChange={(event) => onDraft('remember_api_keys', event.target.checked)}
        />
        <span className="stream-visual" aria-hidden="true">
          <span className="stream-dot" />
        </span>
        <span className="stream-copy">
          <strong>记住 API Key</strong>
          <small>{rememberApiKeys ? '会保存在当前浏览器资料中，刷新后仍可复用' : '默认更安全；刷新页面后需要重新输入'}</small>
        </span>
      </label>
      <label className="stream-card">
        <input
          type="checkbox"
          checked={rememberHistory}
          onChange={(event) => onDraft('remember_history', event.target.checked)}
        />
        <span className="stream-visual" aria-hidden="true">
          <span className="stream-dot" />
        </span>
        <span className="stream-copy">
          <strong>记住本地历史</strong>
          <small>{rememberHistory ? '改写历史会保存在当前浏览器资料中' : '默认不落地；关闭或刷新页面后会清空'}</small>
        </span>
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
          <Save size={16} /> 保存到浏览器
        </button>
        <button className="secondary-button" type="button" onClick={onTest}>
          <PlugZap size={16} /> 测试
        </button>
      </div>
      {testResult && <div className="test-result">{testResult}</div>}
    </section>
  )
}
