import type { HistoryItem, LocalSettings, RewriteResponse } from '../types'

const SETTINGS_KEY = 'ai-cleaner-cloud-settings'
const HISTORY_KEY = 'ai-cleaner-cloud-history'
const HISTORY_LIMIT = 50

export const DEFAULT_SETTINGS: LocalSettings = {
  provider: 'openai',
  openai_model: 'gpt-5.4',
  anthropic_model: 'claude-4-6-sonnet',
  openai_base_url: 'https://api.openai.com/v1',
  anthropic_base_url: 'https://api.anthropic.com',
  openai_api_key: '',
  anthropic_api_key: '',
  remember_api_keys: false,
  remember_history: false,
  stream: true,
  nlp_enabled: false,
  nlp_mode: 'manual',
  nlp_style: 'academic',
  nlp_best_of_n: 10,
  nlp_seed: '',
  nlp_aggressive: false,
}

function safeParse<T>(raw: string | null): T | null {
  if (!raw) return null
  try {
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

function sanitizeSettings(settings: LocalSettings): LocalSettings {
  return {
    ...settings,
    openai_api_key: settings.remember_api_keys ? settings.openai_api_key : '',
    anthropic_api_key: settings.remember_api_keys ? settings.anthropic_api_key : '',
  }
}

export function loadLocalSettings(): LocalSettings {
  if (typeof window === 'undefined') return DEFAULT_SETTINGS
  const parsed = safeParse<Partial<LocalSettings>>(window.localStorage.getItem(SETTINGS_KEY))
  const merged = { ...DEFAULT_SETTINGS, ...(parsed ?? {}) }
  const sanitized = sanitizeSettings(merged)
  if (parsed && JSON.stringify(merged) !== JSON.stringify(sanitized)) {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(sanitized))
  }
  return sanitized
}

export function saveLocalSettings(settings: LocalSettings) {
  if (typeof window === 'undefined') return
  const sanitized = sanitizeSettings(settings)
  window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(sanitized))
  if (!sanitized.remember_history) {
    window.localStorage.removeItem(HISTORY_KEY)
  }
}

export function loadLocalHistory(rememberHistory = true): RewriteResponse[] {
  if (typeof window === 'undefined') return []
  if (!rememberHistory) {
    window.localStorage.removeItem(HISTORY_KEY)
    return []
  }
  const parsed = safeParse<RewriteResponse[]>(window.localStorage.getItem(HISTORY_KEY))
  return Array.isArray(parsed) ? parsed : []
}

export function saveLocalHistory(history: RewriteResponse[], rememberHistory = true) {
  if (typeof window === 'undefined') return
  if (!rememberHistory) {
    window.localStorage.removeItem(HISTORY_KEY)
    return
  }
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_LIMIT)))
}

export function upsertLocalHistory(
  record: RewriteResponse,
  existing: RewriteResponse[],
  rememberHistory = true,
): RewriteResponse[] {
  const next = [record, ...existing.filter((item) => item.id !== record.id)].slice(0, HISTORY_LIMIT)
  saveLocalHistory(next, rememberHistory)
  return next
}

export function deleteLocalHistory(
  id: number,
  existing: RewriteResponse[],
  rememberHistory = true,
): RewriteResponse[] {
  const next = existing.filter((item) => item.id !== id)
  saveLocalHistory(next, rememberHistory)
  return next
}

export function toHistoryItems(records: RewriteResponse[]): HistoryItem[] {
  return records.map((record) => ({
    id: record.id,
    platform: record.platform,
    provider: record.provider,
    model: record.model,
    original_preview: record.original_text.slice(0, 120),
    rewritten_preview: record.rewritten_text.slice(0, 120),
    created_at: record.created_at,
  }))
}
