import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

vi.mock('./lib/api', () => ({
  api: {
    getSettings: vi.fn(async () => ({
      provider: 'openai',
      openai_model: 'gpt-5.4',
      anthropic_model: 'claude-4-6-sonnet',
      openai_base_url: 'https://api.openai.com/v1',
      anthropic_base_url: 'https://api.anthropic.com',
      openai_api_key_set: false,
      anthropic_api_key_set: false,
      openai_api_key_source: 'missing',
      anthropic_api_key_source: 'missing',
      stream: true,
      nlp_enabled: false,
      nlp_mode: 'manual',
      nlp_style: 'academic',
      openai_request_url: 'https://api.openai.com/v1/chat/completions',
      anthropic_request_url: 'https://api.anthropic.com/v1/messages',
      warnings: [],
    })),
    history: vi.fn(async () => []),
    nlpRewrite: vi.fn(),
  },
  streamRewrite: vi.fn(),
  streamNlpRewrite: vi.fn(),
}))

describe('App', () => {
  it('renders the local rewrite workspace', async () => {
    render(<App />)
    expect(await screen.findByText('AI-Cleaner')).toBeInTheDocument()
    expect(screen.getByText('输入')).toBeInTheDocument()
    expect(screen.getByText('仅学术降痕')).toBeInTheDocument()
  })
})
