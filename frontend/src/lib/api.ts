import type { HistoryItem, NlpRewriteRequest, RewriteRequest, RewriteResponse, SettingsView } from '../types'

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || response.statusText)
  }
  return response.json() as Promise<T>
}

export const api = {
  getSettings: () => jsonFetch<SettingsView>('/api/settings'),
  saveSettings: (payload: Partial<SettingsView> & Record<string, unknown>) =>
    jsonFetch<SettingsView>('/api/settings', { method: 'PUT', body: JSON.stringify(payload) }),
  testSettings: (payload: Record<string, unknown>) =>
    jsonFetch<{ ok: boolean; request_url: string; latency_ms: number; response_preview?: string; error?: string }>(
      '/api/settings/test',
      { method: 'POST', body: JSON.stringify(payload) },
    ),
  rewrite: (payload: RewriteRequest) =>
    jsonFetch<RewriteResponse>('/api/rewrite', { method: 'POST', body: JSON.stringify(payload) }),
  nlpRewrite: (payload: NlpRewriteRequest) =>
    jsonFetch<RewriteResponse>('/api/nlp', { method: 'POST', body: JSON.stringify(payload) }),
  history: () => jsonFetch<HistoryItem[]>('/api/history'),
  historyDetail: (id: number) => jsonFetch<RewriteResponse>(`/api/history/${id}`),
  deleteHistory: (id: number) => jsonFetch<{ ok: boolean }>(`/api/history/${id}`, { method: 'DELETE' }),
}

async function streamJson<TPayload>(
  url: string,
  payload: TPayload,
  onEvent: (event: string, data: unknown) => void,
): Promise<void> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok || !response.body) {
    throw new Error(await response.text())
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let lastEvent = ''
  let streamId = ''
  let handlerError: unknown = null

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const messages = buffer.split('\n\n')
      buffer = messages.pop() ?? ''
      for (const message of messages) {
        const eventLine = message.split('\n').find((line) => line.startsWith('event: '))
        const dataLine = message.split('\n').find((line) => line.startsWith('data: '))
        if (!eventLine || !dataLine) continue
        const event = eventLine.slice(7)
        const data = JSON.parse(dataLine.slice(6)) as Record<string, unknown>
        lastEvent = event
        streamId = String(data.stream_id ?? data.streamId ?? streamId)
        try {
          onEvent(event, data)
        } catch (error) {
          handlerError = error
          if (event !== 'error') {
            console.error('[streamJson] event handler failed', { url, streamId, event, message, error })
          }
          throw error
        }
      }
    }
  } catch (error) {
    if (handlerError === error) {
      throw error
    }

    console.error('[streamJson] stream read failed', {
      url,
      streamId,
      lastEvent,
      bufferedBytes: buffer.length,
      bufferedPreview: buffer.slice(0, 500),
      error,
    })
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`流式连接中断${streamId ? `（${streamId}）` : ''}：${message}`)
  }
}

export async function streamRewrite(
  payload: RewriteRequest,
  onEvent: (event: string, data: unknown) => void,
): Promise<void> {
  return streamJson('/api/rewrite/stream', payload, onEvent)
}

export async function streamNlpRewrite(
  payload: NlpRewriteRequest,
  onEvent: (event: string, data: unknown) => void,
): Promise<void> {
  return streamJson('/api/nlp/stream', payload, onEvent)
}
