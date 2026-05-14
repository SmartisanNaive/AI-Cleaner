import { useEffect, useRef } from 'react'

const NLP_FORMULAS = [
  String.raw`\mathcal{L}_{ngram}=-(1/N)\sum_i \log_2 p(c_i | c_{i-2},c_{i-1})`,
  String.raw`p(c_i | c_{i-2},c_{i-1})=\lambda_3 p_3+\lambda_2 p_2+\lambda_1 p_1`,
  String.raw`\lambda_1+\lambda_2+\lambda_3=1`,
  String.raw`PPL(x)=2^{-(1/N)\sum_i \log_2 p(c_i)}`,
  String.raw`PPL_w=2^{-mean(\log_2 p_i)}`,
  String.raw`B=std(PPL_w)/mean(PPL_w)`,
  String.raw`H(C)=-\sum_c p(c)\log_2 p(c)`,
  String.raw`CV_H=\sigma(H_p)/\mu(H_p)`,
  String.raw`s_i=-\log_2 p(c_i)`,
  String.raw`rho_k=sum_t (s_t-mu)(s_{t-k}-mu) / sum_t (s_t-mu)^2`,
  String.raw`hat{s}_k=sum_t s_t exp(-2*pi*i*k*t/N)`,
  String.raw`SF=exp(mean(log(|hat{s}|^2)))/mean(|hat{s}|^2)`,
  String.raw`gamma_1=E[((s-mu)/sigma)^3]`,
  String.raw`gamma_2=E[((s-mu)/sigma)^4]-3`,
  String.raw`r_i=rank(c_i | c_{i-1})`,
  String.raw`GLTR_{10}=count(r_i<=10)/N`,
  String.raw`CV_len=sigma(L_sent)/mu(L_sent)`,
  String.raw`q_short=count(L_sent<10)/N_sent`,
  String.raw`D_comma=100*n_comma/N_char`,
  String.raw`D_trans=1000*n_trans/N_char`,
  String.raw`kappa=E[log p(x)-log p(x_tilde)]`,
  String.raw`Delta_bino=E[log p_primary(c_i)-log p_human(c_i)]`,
  String.raw`MATTR=E_t[unique(c_t...c_{t+w})/w]`,
  String.raw`z_j=(x_j-mu_j_train)/sigma_j_train`,
  String.raw`Pr(y=1 | x)=sigmoid(w^T z+b)`,
  String.raw`J=-(1/N)sum[y log p+(1-y)log(1-p)]+||w||^2/(2NC)`,
  String.raw`S=min(100,S_rule+sum_j omega_j I_j)`,
  String.raw`sigmoid(u)=1/(1+exp(-u))`,
]

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

interface FormulaToken {
  value: string
  color: string
  italic: boolean
  weight: '400' | '500' | '600'
}

function randomBetween(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function tokenizeFormula(formula: string, paletteOffset: number): FormulaToken[] {
  const pieces =
    formula.match(/\\[A-Za-z]+|[A-Za-z_]+|\d+(?:\.\d+)?|[=+\-*/^|(),.{}\[\]<>#:]+|\s+|./g) ?? [formula]

  return pieces.map((value, index) => {
    if (/^\s+$/.test(value)) {
      return { value, color: 'transparent', italic: false, weight: '400' }
    }
    if (/^[=+\-*/^|(),.{}\[\]<>#:]+$/.test(value)) {
      return { value, color: index % 3 === 0 ? '#475569' : '#64748b', italic: false, weight: '400' }
    }
    if (/^\d/.test(value)) {
      return { value, color: '#b45309', italic: false, weight: '600' }
    }
    if (/^\\(?:lambda|sigma|mu|rho|gamma|omega|pi|Delta|kappa)$/.test(value)) {
      return { value, color: '#7c3aed', italic: false, weight: '600' }
    }
    if (/^\\(?:sum|log|exp|mean|std|mathcal)$/.test(value)) {
      return { value, color: '#0891b2', italic: false, weight: '600' }
    }
    if (value.startsWith('\\')) {
      return { value, color: '#7c3aed', italic: false, weight: '600' }
    }
    if (/^[A-Z]/.test(value)) {
      return { value, color: FORMULA_COLORS[(paletteOffset + index + 1) % FORMULA_COLORS.length], italic: true, weight: '600' }
    }
    return {
      value,
      color: FORMULA_COLORS[(paletteOffset + index) % FORMULA_COLORS.length],
      italic: /[A-Za-z]/.test(value),
      weight: '500',
    }
  })
}

function fontForToken(size: number, token: FormulaToken) {
  const style = token.italic ? 'italic ' : ''
  return `${style}${token.weight} ${size}px "SourceHanSerifSC", "Times New Roman", "Georgia", serif`
}

function createFormulaSprite(formula: string, size: number, paletteOffset: number, dpr: number): FormulaSprite {
  const tokens = tokenizeFormula(formula, paletteOffset)
  const measureCanvas = document.createElement('canvas')
  const measureCtx = measureCanvas.getContext('2d')
  if (!measureCtx) {
    return { canvas: measureCanvas, width: 1, height: 1 }
  }

  const paddingX = Math.ceil(size * 0.64)
  const paddingY = Math.ceil(size * 0.46)
  const gap = size * 0.02
  const height = Math.ceil(size * 1.62 + paddingY * 2)
  let width = paddingX * 2

  for (const token of tokens) {
    measureCtx.font = fontForToken(size, token)
    width += measureCtx.measureText(token.value).width + gap
  }
  width = Math.ceil(width)

  const canvas = document.createElement('canvas')
  canvas.width = Math.ceil(width * dpr)
  canvas.height = Math.ceil(height * dpr)
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`

  const ctx = canvas.getContext('2d')
  if (!ctx) return { canvas, width, height }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.textBaseline = 'middle'
  ctx.shadowColor = 'rgba(255,255,255,0.55)'
  ctx.shadowBlur = 2

  let x = paddingX
  const y = height / 2
  for (const token of tokens) {
    ctx.font = fontForToken(size, token)
    ctx.fillStyle = token.color
    ctx.globalAlpha = token.color === 'transparent' ? 0 : 1
    ctx.fillText(token.value, x, y)
    x += ctx.measureText(token.value).width + gap
  }
  ctx.globalAlpha = 1

  return { canvas, width, height }
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

    async function ensureFormulaFontLoaded() {
      if (!document.fonts?.load) return
      await Promise.all([
        document.fonts.load('400 14px SourceHanSerifSC'),
        document.fonts.load('500 14px SourceHanSerifSC'),
        document.fonts.load('600 14px SourceHanSerifSC'),
      ])
    }

    function formulaSize() {
      if (width < 640) return 12
      if (width < 1440) return 14
      return 15
    }

    function build() {
      particles = []
      const size = formulaSize()
      const padX = width < 640 ? 12 : 24
      const lineHeight = width < 640 ? 34 : width < 1440 ? 39 : 42
      const spriteCache = new Map<string, FormulaSprite>()
      let x = padX
      let y = width < 640 ? 42 : 48
      let index = 0
      let safety = 0

      while (y < height + lineHeight * 2 && safety < 1000) {
        const formulaIndex = index % NLP_FORMULAS.length
        const formula = NLP_FORMULAS[formulaIndex]
        const key = `${formulaIndex}-${size}-${dpr}`
        let sprite = spriteCache.get(key)
        if (!sprite) {
          sprite = createFormulaSprite(formula, size, formulaIndex, dpr)
          spriteCache.set(key, sprite)
        }

        const gap = randomBetween(width < 640 ? 18 : 28, width < 640 ? 36 : 58)
        if (x + sprite.width > width - padX && x > padX) {
          x = padX + randomBetween(-8, 18)
          y += lineHeight + randomBetween(-4, 6)
          continue
        }

        const baseX = x + sprite.width / 2 + randomBetween(-4, 4)
        const baseY = y + randomBetween(-3, 3)
        particles.push({
          sprite,
          x: baseX + randomBetween(-0.4, 0.4),
          y: baseY + randomBetween(-0.4, 0.4),
          baseX,
          baseY,
          vx: randomBetween(-0.05, 0.05),
          vy: randomBetween(-0.04, 0.04),
          angle: randomBetween(-0.025, 0.025),
          angularVelocity: 0,
          phase: randomBetween(0, Math.PI * 2),
          mass: randomBetween(1.2, 2.7) + sprite.width / 360,
          opacity: randomBetween(0.36, 0.66),
        })

        x += sprite.width + gap
        index += 1
        safety += 1
      }
    }

    function resize() {
      width = window.innerWidth
      height = window.innerHeight
      canvasEl.width = width * dpr
      canvasEl.height = height * dpr
      canvasEl.style.width = `${width}px`
      canvasEl.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      build()
    }

    function updateWind() {
      const dx = mouse.x - mouse.prevX
      const dy = mouse.y - mouse.prevY
      windVx += (dx * 0.42 - windVx) * 0.16
      windVy += (dy * 0.28 - windVy) * 0.16
      windVx *= WIND_DECAY
      windVy *= WIND_DECAY
      mouse.prevX += (mouse.x - mouse.prevX) * 0.22
      mouse.prevY += (mouse.y - mouse.prevY) * 0.22
      mouse.smoothX += (mouse.x - mouse.smoothX) * 0.18
      mouse.smoothY += (mouse.y - mouse.smoothY) * 0.18

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
          -particle.sprite.width / 2,
          -particle.sprite.height / 2,
          particle.sprite.width,
          particle.sprite.height,
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
    void ensureFormulaFontLoaded().then(() => {
      build()
    })
    raf = requestAnimationFrame((now) => {
      last = now
      draw(now)
    })

    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseleave', onLeave)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseleave', onLeave)
    }
  }, [])

  return <canvas className="wind-canvas" ref={ref} aria-hidden />
}
