import { useEffect, useRef } from 'react'
import {
  BACKGROUND_FORMULA_SPRITES,
  type BackgroundFormulaSpriteAsset,
} from '../generated/backgroundFormulaSprites'

const FORMULA_COLORS = [
  '#0f172a',
  '#1d4ed8',
  '#7c3aed',
  '#0891b2',
  '#be123c',
  '#047857',
  '#b45309',
]

const WIND_DECAY = 0.94
const GUST_INTERVAL_MIN = 2800
const GUST_INTERVAL_MAX = 6200
const GUST_WIDTH = 280
const GUST_SPEED = 520
const GUST_STRENGTH = 0.34

interface FormulaSprite {
  canvas: HTMLCanvasElement
  width: number
  height: number
}

interface FormulaParticle {
  sprite: FormulaSprite
  drawWidth: number
  drawHeight: number
  x: number
  y: number
  baseX: number
  baseY: number
  vx: number
  vy: number
  angle: number
  angularVelocity: number
  phase: number
  mass: number
  opacity: number
}

const formulaImagePromises = new Map<string, Promise<HTMLImageElement>>()
const loadedFormulaImages = new Map<string, HTMLImageElement>()
const tintedSpriteCache = new Map<string, FormulaSprite>()

function randomBetween(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function targetFormulaHeight(viewportWidth: number) {
  if (viewportWidth < 640) return 21
  if (viewportWidth < 1440) return 25
  return 28
}

function targetFormulaMaxWidth(viewportWidth: number) {
  if (viewportWidth < 640) return 168
  if (viewportWidth < 1100) return 218
  if (viewportWidth < 1600) return 254
  return 290
}

function loadFormulaImage(asset: BackgroundFormulaSpriteAsset) {
  const cached = formulaImagePromises.get(asset.id)
  if (cached) return cached

  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    image.onload = () => {
      loadedFormulaImages.set(asset.id, image)
      resolve(image)
    }
    image.onerror = () => reject(new Error(`Failed to load ${asset.url}`))
    image.src = asset.url
  })

  formulaImagePromises.set(asset.id, promise)
  return promise
}

function createTintedSprite(asset: BackgroundFormulaSpriteAsset, color: string) {
  const cacheKey = `${asset.id}:${color}`
  const cached = tintedSpriteCache.get(cacheKey)
  if (cached) return cached

  const image = loadedFormulaImages.get(asset.id)
  if (!image) return null

  const width = Math.max(1, Math.ceil(asset.width))
  const height = Math.max(1, Math.ceil(asset.height))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.clearRect(0, 0, width, height)
  ctx.drawImage(image, 0, 0, width, height)
  ctx.globalCompositeOperation = 'source-in'
  ctx.fillStyle = color
  ctx.fillRect(0, 0, width, height)
  ctx.globalCompositeOperation = 'source-over'

  const sprite = { canvas, width, height }
  tintedSpriteCache.set(cacheKey, sprite)
  return sprite
}

export function WindCanvas() {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const canvasEl = canvas
    const ctx = context as CanvasRenderingContext2D
    let width = 0
    let height = 0
    let raf = 0
    let last = performance.now()
    let time = 0
    let windVx = 0
    let windVy = 0
    let globalWindX = 0
    let globalWindY = 0
    let gustTimer = 0
    let nextGust = randomBetween(GUST_INTERVAL_MIN, GUST_INTERVAL_MAX)
    let gustActive = false
    let gustX = -GUST_WIDTH
    let gustDirection = 1
    let assetsReady = false
    let disposed = false
    let particles: FormulaParticle[] = []
    const dpr = window.devicePixelRatio || 1
    const mouse = {
      x: -9999,
      y: -9999,
      prevX: -9999,
      prevY: -9999,
      smoothX: -9999,
      smoothY: -9999,
    }

    function build() {
      if (!assetsReady) return

      particles = []
      const padX = width < 640 ? 12 : 24
      const rowGap = width < 640 ? 16 : width < 1440 ? 20 : 22
      const baseHeight = targetFormulaHeight(width)
      const maxWidth = targetFormulaMaxWidth(width)
      let x = padX + randomBetween(-8, 18)
      let y = width < 640 ? 26 : 34
      let rowHeight = 0
      let assetIndex = 0
      let safety = 0
      let rowSprites: Array<{
        sprite: FormulaSprite
        drawWidth: number
        drawHeight: number
        x: number
      }> = []

      function commitRow() {
        if (!rowSprites.length) return

        const centerY = y + rowHeight / 2
        for (const rowSprite of rowSprites) {
          const { sprite, drawWidth, drawHeight, x: rowX } = rowSprite
          const baseX = rowX + drawWidth / 2 + randomBetween(-4, 4)
          const baseY = centerY + randomBetween(-3, 3)
          particles.push({
            sprite,
            drawWidth,
            drawHeight,
            x: baseX + randomBetween(-0.4, 0.4),
            y: baseY + randomBetween(-0.4, 0.4),
            baseX,
            baseY,
            vx: randomBetween(-0.05, 0.05),
            vy: randomBetween(-0.04, 0.04),
            angle: randomBetween(-0.025, 0.025),
            angularVelocity: 0,
            phase: randomBetween(0, Math.PI * 2),
            mass: randomBetween(1.2, 2.7) + drawWidth / 320 + rowHeight / 380,
            opacity: randomBetween(0.32, 0.58),
          })
        }

        y += rowHeight + rowGap + randomBetween(-4, 6)
        x = padX + randomBetween(-8, 18)
        rowHeight = 0
        rowSprites = []
      }

      while (y < height + rowGap * 2 && safety < 1000) {
        const asset = BACKGROUND_FORMULA_SPRITES[assetIndex % BACKGROUND_FORMULA_SPRITES.length]
        const colorIndex = (assetIndex + Math.floor(randomBetween(0, FORMULA_COLORS.length))) % FORMULA_COLORS.length
        const sprite = createTintedSprite(asset, FORMULA_COLORS[colorIndex])
        assetIndex += 1
        safety += 1

        if (!sprite) continue

        let drawHeight = baseHeight * randomBetween(0.92, 1.08)
        let drawWidth = sprite.width * (drawHeight / sprite.height)
        if (drawWidth > maxWidth) {
          const fitScale = maxWidth / drawWidth
          drawWidth *= fitScale
          drawHeight *= fitScale
        }

        const gap = randomBetween(width < 640 ? 14 : 24, width < 640 ? 24 : 42)
        if (x + drawWidth > width - padX && rowSprites.length) {
          commitRow()
          continue
        }

        rowSprites.push({ sprite, drawWidth, drawHeight, x })
        rowHeight = Math.max(rowHeight, drawHeight)
        x += drawWidth + gap
      }

      commitRow()
    }

    function resize() {
      width = window.innerWidth
      height = window.innerHeight
      canvasEl.width = width * dpr
      canvasEl.height = height * dpr
      canvasEl.style.width = `${width}px`
      canvasEl.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.imageSmoothingEnabled = true
      if (assetsReady) build()
    }

    function updateWind() {
      const pointerActive = mouse.x > -9000 && mouse.y > -9000
      const dx = pointerActive ? mouse.x - mouse.prevX : 0
      const dy = pointerActive ? mouse.y - mouse.prevY : 0
      windVx += (dx * 0.42 - windVx) * 0.16
      windVy += (dy * 0.28 - windVy) * 0.16
      windVx *= WIND_DECAY
      windVy *= WIND_DECAY
      if (pointerActive) {
        mouse.prevX += (mouse.x - mouse.prevX) * 0.22
        mouse.prevY += (mouse.y - mouse.prevY) * 0.22
        mouse.smoothX += (mouse.x - mouse.smoothX) * 0.18
        mouse.smoothY += (mouse.y - mouse.smoothY) * 0.18
      } else {
        mouse.prevX = -9999
        mouse.prevY = -9999
        mouse.smoothX = -9999
        mouse.smoothY = -9999
      }

      globalWindX = Math.sin(time * 0.32) * 0.08 + Math.sin(time * 0.73) * 0.035
      globalWindY = Math.cos(time * 0.24) * 0.025
    }

    function updateGust(dt: number) {
      gustTimer += dt
      if (!gustActive && gustTimer >= nextGust) {
        gustActive = true
        gustDirection = Math.random() < 0.5 ? 1 : -1
        gustX = gustDirection > 0 ? -GUST_WIDTH : width + GUST_WIDTH
        gustTimer = 0
        nextGust = randomBetween(GUST_INTERVAL_MIN, GUST_INTERVAL_MAX)
      }

      if (gustActive) {
        gustX += GUST_SPEED * (dt / 1000) * gustDirection
        if ((gustDirection > 0 && gustX > width + GUST_WIDTH) || (gustDirection < 0 && gustX < -GUST_WIDTH)) {
          gustActive = false
        }
      }
    }

    function gustAt(x: number) {
      if (!gustActive) return 0
      const dist = Math.abs(x - gustX)
      if (dist > GUST_WIDTH) return 0
      const falloff = 1 - dist / GUST_WIDTH
      return GUST_STRENGTH * falloff * falloff * gustDirection
    }

    function drawBackground() {
      const gradient = ctx.createLinearGradient(0, 0, 0, height)
      gradient.addColorStop(0, '#f9fafb')
      gradient.addColorStop(0.46, '#eef2f7')
      gradient.addColorStop(1, '#e6eaf0')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, width, height)

      const topGlow = ctx.createRadialGradient(width * 0.16, height * 0.06, 0, width * 0.16, height * 0.06, Math.max(width, height) * 0.7)
      topGlow.addColorStop(0, 'rgba(59,130,246,0.13)')
      topGlow.addColorStop(0.42, 'rgba(124,58,237,0.045)')
      topGlow.addColorStop(1, 'rgba(255,255,255,0)')
      ctx.fillStyle = topGlow
      ctx.fillRect(0, 0, width, height)

      const bottomGlow = ctx.createRadialGradient(width * 0.92, height * 0.92, 0, width * 0.92, height * 0.92, Math.max(width, height) * 0.62)
      bottomGlow.addColorStop(0, 'rgba(20,184,166,0.10)')
      bottomGlow.addColorStop(0.52, 'rgba(180,83,9,0.035)')
      bottomGlow.addColorStop(1, 'rgba(255,255,255,0)')
      ctx.fillStyle = bottomGlow
      ctx.fillRect(0, 0, width, height)
    }

    function draw(now: number) {
      const dt = Math.min(32, now - last)
      last = now
      time += dt / 1000

      updateWind()
      updateGust(dt)
      drawBackground()

      for (const particle of particles) {
        const dx = particle.baseX - mouse.smoothX
        const dy = particle.baseY - mouse.smoothY
        const dist = Math.sqrt(dx * dx + dy * dy)
        const radius = width < 640 ? 180 : 300
        const force = Math.max(0, 1 - dist / radius)
        const direction = Math.atan2(dy, dx)
        const eddy = Math.sin(time * 2.1 + particle.phase + particle.baseX * 0.006) * 0.014
        const gust = gustAt(particle.baseX)

        particle.vx += (windVx * force * 0.035 + Math.cos(direction) * force * 0.32 + globalWindX + gust + eddy) / particle.mass
        particle.vy += (windVy * force * 0.025 + Math.sin(direction) * force * 0.16 + globalWindY + eddy * 0.55) / particle.mass
        particle.vx += (particle.baseX - particle.x) * 0.010
        particle.vy += (particle.baseY - particle.y) * 0.010
        particle.vx *= 0.86
        particle.vy *= 0.86
        particle.x += particle.vx
        particle.y += particle.vy

        particle.angularVelocity += (particle.vx * 0.0018 + force * Math.sin(time + particle.phase) * 0.006 - particle.angle * 0.025) / particle.mass
        particle.angularVelocity *= 0.84
        particle.angle += particle.angularVelocity

        ctx.save()
        ctx.translate(particle.x, particle.y)
        ctx.rotate(particle.angle)
        ctx.globalAlpha = Math.min(0.84, particle.opacity + force * 0.18 + Math.abs(gust) * 0.16)
        ctx.drawImage(
          particle.sprite.canvas,
          -particle.drawWidth / 2,
          -particle.drawHeight / 2,
          particle.drawWidth,
          particle.drawHeight,
        )
        ctx.restore()
      }

      ctx.globalAlpha = 1
      raf = requestAnimationFrame(draw)
    }

    function onMove(event: MouseEvent) {
      if (mouse.x < -9000) {
        mouse.prevX = event.clientX
        mouse.prevY = event.clientY
        mouse.smoothX = event.clientX
        mouse.smoothY = event.clientY
      }
      mouse.x = event.clientX
      mouse.y = event.clientY
    }

    function onLeave() {
      mouse.x = -9999
      mouse.y = -9999
    }

    resize()
    void Promise.all(BACKGROUND_FORMULA_SPRITES.map(loadFormulaImage))
      .then(() => {
        if (disposed) return
        assetsReady = true
        build()
      })
      .catch((error) => {
        console.error('Failed to prepare background formula sprites.', error)
      })

    raf = requestAnimationFrame((now) => {
      last = now
      draw(now)
    })

    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseleave', onLeave)
    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseleave', onLeave)
    }
  }, [])

  return <canvas className="wind-canvas" ref={ref} aria-hidden />
}
