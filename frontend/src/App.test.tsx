import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

vi.mock('./lib/api', () => ({
  api: {
    testSettings: vi.fn(),
    rewrite: vi.fn(),
  },
  streamRewrite: vi.fn(),
  streamNlpRewrite: vi.fn(),
}))

describe('App', () => {
  it('renders the local rewrite workspace', async () => {
    render(<App />)
    expect(await screen.findByText('AI-Cleaner')).toBeInTheDocument()
    expect(screen.getByText('输入')).toBeInTheDocument()
    expect(screen.getByText('NLP 降 AIGC')).toBeInTheDocument()
  })
})
