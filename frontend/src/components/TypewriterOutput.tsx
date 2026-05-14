import { useEffect, useMemo, useRef, useState } from 'react'
import { prepareWithSegments, layoutWithLines } from '@chenglou/pretext'

const TYPE_SPEED = 42
const TYPE_VARIATION = 0.72
const CATCH_UP_GAP = 260
const MAX_CATCH_UP_CHARS = 4
const PUNCTUATION_PAUSE = 4.2

interface Props {
  text: string
  placeholder: string
}

interface TypographyMetrics {
  font: string
  lineHeight: number
  contentWidth: number
}

function getTypographyMetrics(node: HTMLElement | null): TypographyMetrics {
  if (!node) {
    return {
      font: '15px "SourceHanSerifSC", "Inter", Georgia, serif',
      lineHeight: 28,
      contentWidth: 640,
    }
  }

  const style = window.getComputedStyle(node)
  const font = [
    style.fontStyle,
    style.fontVariant,
    style.fontWeight,
    style.fontSize,
    style.fontFamily,
  ].join(' ')
  const fontSize = Number.parseFloat(style.fontSize) || 15
  const lineHeight = style.lineHeight === 'normal' ? fontSize * 1.85 : Number.parseFloat(style.lineHeight) || fontSize * 1.85
  const paddingX = Number.parseFloat(style.paddingLeft) + Number.parseFloat(style.paddingRight)

  return {
    font,
    lineHeight,
    contentWidth: Math.max(1, node.clientWidth - paddingX),
  }
}

function nextDelay(remaining: number, nextChar: string) {
  const punctuationPause = /[。！？；：，,.!?;:\n]/.test(nextChar) ? PUNCTUATION_PAUSE : 1
  const catchUp = remaining > CATCH_UP_GAP ? Math.min(MAX_CATCH_UP_CHARS, Math.ceil(remaining / 90)) : 1
  const jitter = 1 + (Math.random() * 2 - 1) * TYPE_VARIATION
  return {
    delay: Math.max(14, (1000 / TYPE_SPEED) * jitter * punctuationPause),
    chars: catchUp,
  }
}

function layoutText(text: string, metrics: TypographyMetrics) {
  if (!text) return []
  const prepared = prepareWithSegments(text, metrics.font, { whiteSpace: 'pre-wrap' })
  return layoutWithLines(prepared, metrics.contentWidth, metrics.lineHeight).lines
}

export function TypewriterOutput({ text, placeholder }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const timeoutRef = useRef<number | null>(null)
  const typedIndexRef = useRef(Array.from(text).length)
  const visibleRef = useRef(text)
  const previousLineCountRef = useRef(0)

  const [visibleText, setVisibleText] = useState(text)
  const [metrics, setMetrics] = useState<TypographyMetrics>(() => getTypographyMetrics(null))
  const [lastTypedIndex, setLastTypedIndex] = useState(-1)
  const [carriageKey, setCarriageKey] = useState(0)

  useEffect(() => {
    const node = containerRef.current
    if (!node) return

    const update = () => setMetrics(getTypographyMetrics(node))
    update()

    const observer = new ResizeObserver(update)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  const lines = useMemo(() => layoutText(visibleText, metrics), [metrics, visibleText])

  useEffect(() => {
    if (lines.length > previousLineCountRef.current && previousLineCountRef.current > 0) {
      setCarriageKey((key) => key + 1)
    }
    previousLineCountRef.current = lines.length
  }, [lines.length])

  useEffect(() => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    if (!text) {
      typedIndexRef.current = 0
      visibleRef.current = ''
      setVisibleText('')
      setLastTypedIndex(-1)
      previousLineCountRef.current = 0
      return
    }

    if (!text.startsWith(visibleRef.current)) {
      typedIndexRef.current = 0
      visibleRef.current = ''
      setVisibleText('')
      setLastTypedIndex(-1)
      previousLineCountRef.current = 0
    }

    const tick = () => {
      const chars = Array.from(text)
      const remaining = chars.length - typedIndexRef.current

      if (remaining <= 0) {
        timeoutRef.current = null
        return
      }

      const nextChar = chars[typedIndexRef.current] ?? ''
      const { delay, chars: step } = nextDelay(remaining, nextChar)
      typedIndexRef.current = Math.min(chars.length, typedIndexRef.current + step)
      const nextVisible = chars.slice(0, typedIndexRef.current).join('')
      visibleRef.current = nextVisible
      setVisibleText(nextVisible)
      setLastTypedIndex(typedIndexRef.current - 1)

      timeoutRef.current = window.setTimeout(tick, delay)
    }

    timeoutRef.current = window.setTimeout(tick, 0)

    return () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
    }
  }, [text])

  const isTyping = Boolean(text && visibleText !== text)
  const lastLineIndex = Math.max(0, lines.length - 1)
  const lastLineText = lines[lastLineIndex]?.text ?? ''

  return (
    <div ref={containerRef} className="typewriter-output" aria-live="polite">
      {!visibleText && <span className="typewriter-placeholder">{placeholder}</span>}

      {visibleText && (
        <span className="typewriter-lines" style={{ lineHeight: `${metrics.lineHeight}px` }}>
          {lines.map((line, lineIndex) => {
            const isLastLine = lineIndex === lastLineIndex
            const textBeforeLastGlyph = isLastLine ? lastLineText.slice(0, -1) : line.text
            const lastGlyph = isLastLine ? lastLineText.slice(-1) : ''

            return (
              <span className="typewriter-line" key={`${line.start.segmentIndex}-${line.start.graphemeIndex}-${lineIndex}`}>
                <span className="typewriter-ink-run">
                  {textBeforeLastGlyph}
                  {isLastLine && lastGlyph && (
                    <span className="typewriter-glyph-pop" key={lastTypedIndex}>
                      {lastGlyph}
                    </span>
                  )}
                </span>
                {isLastLine && text && (
                  <span
                    key={carriageKey}
                    className={isTyping ? 'typewriter-caret typing carriage-return' : 'typewriter-caret'}
                  />
                )}
              </span>
            )
          })}
        </span>
      )}
    </div>
  )
}
