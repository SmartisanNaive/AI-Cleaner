export type ProviderName = 'openai' | 'anthropic'
export type HistoryProviderName = ProviderName | 'local'
export type PlatformName = 'weipu' | 'paperyy' | 'paperpass' | 'zhuque'
export type NlpMode = 'off' | 'manual' | 'auto'
export type NlpStyle = 'academic' | 'general' | 'long_blog'

export interface SettingsView {
  provider: ProviderName
  openai_model: string
  anthropic_model: string
  openai_base_url: string
  anthropic_base_url: string
  openai_api_key_set: boolean
  anthropic_api_key_set: boolean
  openai_api_key_source: string
  anthropic_api_key_source: string
  stream: boolean
  nlp_enabled: boolean
  nlp_mode: NlpMode
  nlp_style: NlpStyle
  openai_request_url: string
  anthropic_request_url: string
  warnings: string[]
}

export interface RewriteRequest {
  text: string
  platform: PlatformName
  iterations: number
  provider?: ProviderName
  model?: string
  stream: boolean
  nlp_enabled?: boolean
  nlp_mode?: NlpMode
  nlp_style?: NlpStyle
  nlp_aggressive?: boolean
  nlp_best_of_n?: number
  nlp_seed?: number
}

export interface NlpRewriteRequest {
  text: string
  platform: PlatformName
  nlp_mode: NlpMode
  nlp_style: NlpStyle
  aggressive?: boolean
  best_of_n?: number
  seed?: number
}

export interface DiffSpan {
  kind: 'equal' | 'insert' | 'delete' | 'replace'
  original: string
  revised: string
}

export interface RewriteResponse {
  id: number
  original_text: string
  rewritten_text: string
  raw_output: string
  platform: PlatformName
  provider: HistoryProviderName
  model: string
  iterations: number
  warnings: string[]
  nlp_applied: boolean
  nlp_style?: NlpStyle
  diff: DiffSpan[]
  created_at: string
}

export interface HistoryItem {
  id: number
  platform: PlatformName
  provider: HistoryProviderName
  model: string
  original_preview: string
  rewritten_preview: string
  created_at: string
}
