import type { DiffSpan } from '../types'

export function DiffView({ spans }: { spans: DiffSpan[] }) {
  if (!spans.length) return <div className="empty-state">等待改写结果</div>
  return (
    <div className="diff-grid">
      <div className="diff-pane">
        <div className="pane-title">原文</div>
        <p>
          {spans.map((span, index) => (
            <span
              key={`o-${index}`}
              className={span.kind === 'delete' || span.kind === 'replace' ? 'diff-delete' : ''}
            >
              {span.original}
            </span>
          ))}
        </p>
      </div>
      <div className="diff-pane">
        <div className="pane-title">改写后</div>
        <p>
          {spans.map((span, index) => (
            <span
              key={`r-${index}`}
              className={span.kind === 'insert' || span.kind === 'replace' ? 'diff-insert' : ''}
            >
              {span.revised}
            </span>
          ))}
        </p>
      </div>
    </div>
  )
}

