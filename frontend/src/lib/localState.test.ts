import { afterEach, describe, expect, it } from 'vitest'
import {
  DEFAULT_SETTINGS,
  loadLocalHistory,
  loadLocalSettings,
  saveLocalSettings,
  upsertLocalHistory,
} from './localState'
import type { RewriteResponse } from '../types'

const SETTINGS_KEY = 'ai-cleaner-cloud-settings'
const HISTORY_KEY = 'ai-cleaner-cloud-history'

const sampleRecord: RewriteResponse = {
  id: 1,
  original_text: '原文',
  rewritten_text: '改写',
  raw_output: '改写',
  platform: 'weipu',
  provider: 'openai',
  model: 'gpt-5.4',
  iterations: 1,
  warnings: [],
  nlp_applied: false,
  diff: [],
  created_at: '2026-05-15T00:00:00.000Z',
}

afterEach(() => {
  window.localStorage.clear()
})

describe('localState', () => {
  it('scrubs saved API keys unless remember_api_keys is enabled', () => {
    window.localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        ...DEFAULT_SETTINGS,
        openai_api_key: 'sk-test',
        anthropic_api_key: 'ak-test',
      }),
    )

    const settings = loadLocalSettings()

    expect(settings.openai_api_key).toBe('')
    expect(settings.anthropic_api_key).toBe('')
    expect(window.localStorage.getItem(SETTINGS_KEY)).not.toContain('sk-test')
  })

  it('drops persisted history when remember_history is disabled', () => {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify([sampleRecord]))

    const history = loadLocalHistory(false)

    expect(history).toEqual([])
    expect(window.localStorage.getItem(HISTORY_KEY)).toBeNull()
  })

  it('saves keys and history only when the remember flags are enabled', () => {
    saveLocalSettings({
      ...DEFAULT_SETTINGS,
      openai_api_key: 'sk-test',
      remember_api_keys: true,
      remember_history: true,
    })

    const storedSettings = window.localStorage.getItem(SETTINGS_KEY) ?? ''
    expect(storedSettings).toContain('sk-test')

    upsertLocalHistory(sampleRecord, [], true)

    expect(loadLocalHistory(true)).toHaveLength(1)
  })
})
