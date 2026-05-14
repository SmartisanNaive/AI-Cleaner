import '@testing-library/jest-dom/vitest'

const gradientMock = {
  addColorStop: () => undefined,
} as unknown as CanvasGradient

const canvasContextMock = {
  clearRect: () => undefined,
  fillRect: () => undefined,
  fillText: () => undefined,
  drawImage: () => undefined,
  getImageData: () => ({ data: new Uint8ClampedArray(4), width: 1, height: 1, colorSpace: 'srgb' }) as ImageData,
  putImageData: () => undefined,
  measureText: (text: string) => ({
    width: text.length * 8,
    actualBoundingBoxLeft: 0,
    actualBoundingBoxRight: text.length * 8,
    actualBoundingBoxAscent: 10,
    actualBoundingBoxDescent: 4,
    fontBoundingBoxAscent: 10,
    fontBoundingBoxDescent: 4,
    emHeightAscent: 10,
    emHeightDescent: 4,
    hangingBaseline: 8,
    alphabeticBaseline: 0,
    ideographicBaseline: -4,
  }) as TextMetrics,
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
  textBaseline: 'alphabetic' as CanvasTextBaseline,
  globalAlpha: 1,
}

class OffscreenCanvasMock {
  width: number
  height: number

  constructor(width: number, height: number) {
    this.width = width
    this.height = height
  }

  getContext(): OffscreenCanvasRenderingContext2D | null {
    return canvasContextMock as unknown as OffscreenCanvasRenderingContext2D
  }
}

Object.defineProperty(globalThis, 'OffscreenCanvas', {
  value: OffscreenCanvasMock,
  configurable: true,
})

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  value: () => canvasContextMock as unknown as CanvasRenderingContext2D,
})

Object.defineProperty(window, 'requestAnimationFrame', {
  value: (cb: FrameRequestCallback) => window.setTimeout(() => cb(Date.now()), 16),
})

Object.defineProperty(window, 'cancelAnimationFrame', {
  value: (id: number) => window.clearTimeout(id),
})

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(globalThis, 'ResizeObserver', {
  value: ResizeObserverMock,
  configurable: true,
})
