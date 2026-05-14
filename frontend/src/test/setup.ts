import '@testing-library/jest-dom/vitest'

const gradientMock = {
  addColorStop: () => undefined,
}

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: () => ({
    clearRect: () => undefined,
    fillRect: () => undefined,
    fillText: () => undefined,
    drawImage: () => undefined,
    measureText: (text: string) => ({ width: text.length * 8 }),
    setTransform: () => undefined,
    createLinearGradient: () => gradientMock,
    createRadialGradient: () => gradientMock,
    save: () => undefined,
    restore: () => undefined,
    translate: () => undefined,
    rotate: () => undefined,
    font: '',
    fillStyle: '',
    shadowColor: '',
    shadowBlur: 0,
    textBaseline: 'alphabetic',
    globalAlpha: 1,
  }),
})

Object.defineProperty(window, 'requestAnimationFrame', {
  value: (cb: FrameRequestCallback) => window.setTimeout(() => cb(Date.now()), 16),
})

Object.defineProperty(window, 'cancelAnimationFrame', {
  value: (id: number) => window.clearTimeout(id),
})
