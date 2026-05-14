import { Clock, Trash2 } from 'lucide-react'
import type { HistoryItem } from '../types'

export function HistoryList({
  items,
  onOpen,
  onDelete,
}: {
  items: HistoryItem[]
  onOpen: (id: number) => void
  onDelete: (id: number) => void
}) {
  return (
    <div className="history-list">
      {items.map((item) => (
        <article className="history-item" key={item.id}>
          <button type="button" className="history-main" onClick={() => onOpen(item.id)}>
            <span className="history-meta">
              <Clock size={14} /> {new Date(item.created_at).toLocaleString()} · {item.provider} ·{' '}
              {item.platform}
            </span>
            <span>{item.original_preview || '空记录'}</span>
          </button>
          <button
            className="icon-button"
            type="button"
            title="删除记录"
            onClick={() => onDelete(item.id)}
          >
            <Trash2 size={16} />
          </button>
        </article>
      ))}
      {!items.length && <div className="empty-state">暂无历史记录</div>}
    </div>
  )
}

