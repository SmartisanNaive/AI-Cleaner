import { useEffect, useRef, useState } from 'react'

const TYPE_CHARS_PER_SECOND = 420
const CATCH_UP_GAP = 160
const MAX_CATCH_UP_CHARS = 28

interface Props {
  text: string
  placeholder: string
}

export function TypewriterOutput({ text, placeholder }: Props) {
  const [visibleText, setVisibleText] = useState(text)
  const targetRef = useRef(text)
  const visibleRef = useRef(text)
  const indexRef = useRef(Array.from(text).length)
  const rafRef = useRef<number | null>(null)
  const lastFrameRef = useRef(0)

  useEffect(() => {
    targetRef.current = text

    if (!text) {
      visibleRef.current = ''
      indexRef.current = 0
      setVisibleText('')
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      return
    }

    if (!text.startsWith(visibleRef.current)) {
      visibleRef.current = ''
      indexRef.current = 0
      setVisibleText('')
    }

    function tick(timestamp: number) {
      const targetChars = Array.from(targetRef.current)
      if (indexRef.current >= targetChars.length) {
        rafRef.current = null
        lastFrameRef.current = 0
        return
      }

      if (!lastFrameRef.current) lastFrameRef.current = timestamp
      const elapsed = Math.max(0, timestamp - lastFrameRef.current)
      lastFrameRef.current = timestamp

      const remaining = targetChars.length - indexRef.current
      const baseStep = Math.max(1, Math.floor((elapsed / 1000) * TYPE_CHARS_PER_SECOND))
      const catchUpStep =
        remaining > CATCH_UP_GAP ? Math.min(MAX_CATCH_UP_CHARS, Math.ceil(remaining / 18)) : 0
      const step = Math.max(baseStep, catchUpStep)

      indexRef.current = Math.min(targetChars.length, indexRef.current + step)
      const nextVisible = targetChars.slice(0, indexRef.current).join('')
      visibleRef.current = nextVisible
      setVisibleText(nextVisible)
      rafRef.current = requestAnimationFrame(tick)
    }

    if (rafRef.current === null) {
      lastFrameRef.current = 0
      rafRef.current = requestAnimationFrame(tick)
    }

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
    }
  }, [text])

  const isTyping = text && visibleText !== text

  return (
    <div className="typewriter-output" aria-live="polite">
      <span>{visibleText || placeholder}</span>
      {text && <span className={isTyping ? 'typewriter-caret typing' : 'typewriter-caret'} />}
    </div>
  )
}
