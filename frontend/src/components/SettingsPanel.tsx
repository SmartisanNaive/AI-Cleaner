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
  const provider = draft.provider as ProviderName
  const openaiBase = String(draft.openai_base_url || settings.openai_base_url)
  const anthropicBase = String(draft.anthropic_base_url || settings.anthropic_base_url)
  const requestUrl =
    provider === 'openai'
      ? `${openaiBase.replace(/\/$/, '')}/chat/completions`
      : `${anthropicBase.replace(/\/$/, '')}/v1/messages`
  const liveWarnings = [
    provider === 'openai' && !openaiBase.replace(/\/$/, '').endsWith('/v1')
      ? 'OpenAI Chat Completions 通常需要 base URL 以 /v1 结尾。'
      : '',
    provider === 'anthropic' && requestUrl.includes('/v1/v1')
      ? 'Anthropic 请求 URL 中出现 /v1/v1，请检查 base URL 是否重复包含 /v1。'
      : '',
    ...settings.warnings,
  ].filter((warning): warning is string => Boolean(warning))

  return (
    <section className="settings-panel">
      <div className="section-title">设置</div>
      <label>
        SDK
        <select value={provider} onChange={(event) => onDraft('provider', event.target.value)}>
          <option value="openai">OpenAI SDK</option>
          <option value="anthropic">Anthropic SDK</option>
        </select>
      </label>
      <label>
        OpenAI Model
        <input
          value={String(draft.openai_model ?? '')}
          onChange={(event) => onDraft('openai_model', event.target.value)}
        />
      </label>
      <label>
        Anthropic Model
        <input
          value={String(draft.anthropic_model ?? '')}
          onChange={(event) => onDraft('anthropic_model', event.target.value)}
        />
      </label>
      <label>
        OpenAI Base URL
        <input
          value={String(draft.openai_base_url ?? '')}
          onChange={(event) => onDraft('openai_base_url', event.target.value)}
        />
      </label>
      <label>
        Anthropic Base URL
        <input
          value={String(draft.anthropic_base_url ?? '')}
          onChange={(event) => onDraft('anthropic_base_url', event.target.value)}
        />
      </label>
      <label>
        OpenAI API Key
        <input
          type="password"
          placeholder={settings.openai_api_key_set ? `已配置 · ${settings.openai_api_key_source}` : '未配置'}
          onChange={(event) => onDraft('openai_api_key', event.target.value)}
        />
      </label>
      <label>
        Anthropic API Key
        <input
          type="password"
          placeholder={
            settings.anthropic_api_key_set ? `已配置 · ${settings.anthropic_api_key_source}` : '未配置'
          }
          onChange={(event) => onDraft('anthropic_api_key', event.target.value)}
        />
      </label>
      <div className="url-preview">{requestUrl}</div>
      {Array.from(new Set(liveWarnings)).map((warning) => (
        <div className="warning" key={warning}>
          {warning}
        </div>
      ))}
      <label className="inline-check">
        <input
          type="checkbox"
          checked={Boolean(draft.stream)}
          onChange={(event) => onDraft('stream', event.target.checked)}
        />
        流式输出
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
