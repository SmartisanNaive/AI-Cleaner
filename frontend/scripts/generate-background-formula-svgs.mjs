import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const BACKGROUND_FORMULAS = [
  String.raw`\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`,
  String.raw`\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)`,
  String.raw`\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O`,
  String.raw`\mathrm{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2`,
  String.raw`\mathrm{LayerNorm}(x) + \mathrm{FFN}(x)`,
  String.raw`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)`,
  String.raw`PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)`,
  String.raw`\mathrm{RoPE}\!: \; q_m \cdot k_n \sim f(m - n)`,
  String.raw`\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\displaystyle\sum_j e^{z_j}}`,
  String.raw`\mathrm{GELU}(x) = \frac{x}{2}\!\left[1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]`,
  String.raw`\mathrm{SiLU}(x) = x \cdot \sigma(x)`,
  String.raw`\mathcal{L}_{\text{CLM}} = -\sum_t \log p(x_t \mid x_{<t})`,
  String.raw`\mathcal{L}_{\text{MLM}} = -\mathbb{E}\!\sum_{i \in \mathcal{M}} \log p(x_i \mid x_{\text{masked}})`,
  String.raw`\mathrm{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N}\log p(x_t \mid x_{<t})\right)`,
  String.raw`\mathcal{L}_{\text{ngram}} = -\frac{1}{N}\sum_i \log_2 p(c_i \mid c_{i-2},c_{i-1})`,
  String.raw`p(x_t) = \mathrm{softmax}\!\left(\frac{\mathrm{logit}_t}{T}\right)`,
  String.raw`p_{\text{nucleus}}(x) = \min\!\sum_{x:\,\mathrm{cumP}(x)\ge p} P(x)`,
  String.raw`H(p) = -\sum_x p(x)\log p(x)`,
  String.raw`D_{\mathrm{KL}}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)}`,
  String.raw`I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)}`,
  String.raw`\mathrm{BLEU} = \mathrm{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)`,
  String.raw`\mathrm{ROUGE\text{-}N} = \frac{\sum_S \mathrm{match}_n}{\sum_S \mathrm{ref}_n}`,
  String.raw`\mathrm{WER} = \frac{S + D + I}{N}`,
  String.raw`\text{TF-IDF}(t,d) = \mathrm{tf}(t,d) \cdot \log\frac{N}{\mathrm{df}(t)}`,
  String.raw`\mathrm{sim}_{\cos}(a,b) = \frac{a \cdot b}{\|a\|\;\|b\|}`,
  String.raw`\mathrm{LayerNorm}(x) = \gamma\,\frac{x - \mu}{\sigma} + \beta`,
  String.raw`\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}\,\gamma`,
  String.raw`\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}`,
  String.raw`g_t = \mathrm{clip}(g_t,\; -C,\; C)`,
  String.raw`\mathcal{L}_{\text{PPO}} = \mathbb{E}[r_t] - \beta\, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})`,
  String.raw`R(x,y) = r_\phi(x,y) - \beta\log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}`,
  String.raw`\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_y}{\pi_{\text{ref}}} - \beta\log\frac{\pi_{y'}}{\pi_{\text{ref}}}\right)\right]`,
  String.raw`\text{BPE}\!: \;\mathrm{merge}\;\arg\max_{\text{pair}} \mathrm{count}(\text{pair})`,
  String.raw`\mathrm{PPL}_w = 2^{-\mathrm{mean}(\log_2 p_i)}`,
  String.raw`B = \frac{\mathrm{std}(\mathrm{PPL}_w)}{\mathrm{mean}(\mathrm{PPL}_w)}`,
  String.raw`\kappa = \mathbb{E}\!\left[\log p(x) - \log p(\tilde{x})\right]`,
  String.raw`\mathrm{MATTR} = \mathbb{E}_t\!\left[\frac{\mathrm{unique}(c_t \dots c_{t+w})}{w}\right]`,
  String.raw`CV_H = \frac{\sigma(H_p)}{\mu(H_p)}`,
  String.raw`CV_{\text{len}} = \frac{\sigma(L_{\text{sent}})}{\mu(L_{\text{sent}})}`,
  String.raw`P(y{=}1 \mid x) = \sigma(w^\top z + b)`,
  String.raw`\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum\!\left[y\log p + (1{-}y)\log(1{-}p)\right]`,
  String.raw`z_j = \frac{x_j - \mu_j}{\sigma_j}`,
  String.raw`y = \sum_i g_i(x)\,E_i(x)`,
  String.raw`g(x) = \mathrm{softmax}\!\left(\mathrm{top}_k(W_g\,x)\right)`,
  String.raw`\mathrm{FlashAttn}\!: \; O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V`,
  String.raw`KV_{n+1} = \left[KV_n;\;(K_n,\, V_n)\right]`,
]

const BASE_HEIGHT = 128
const EM = 16
const EX = 8
const CONTAINER_WIDTH = 160 * EM

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '..')
const assetDir = path.join(projectRoot, 'src', 'assets', 'background-formulas')
const generatedDir = path.join(projectRoot, 'src', 'generated')
const generatedModulePath = path.join(generatedDir, 'backgroundFormulaSprites.ts')

function parseViewBox(svgMarkup) {
  const match = svgMarkup.match(/viewBox="([^"]+)"/)
  if (!match) {
    throw new Error('Generated SVG is missing a viewBox')
  }

  const parts = match[1].trim().split(/\s+/).map(Number)
  if (parts.length !== 4 || parts.some((value) => !Number.isFinite(value))) {
    throw new Error(`Invalid viewBox: ${match[1]}`)
  }

  return {
    minX: parts[0],
    minY: parts[1],
    width: parts[2],
    height: parts[3],
  }
}

function cleanSvgMarkup(svgMarkup) {
  const viewBox = parseViewBox(svgMarkup)
  const aspectRatio = viewBox.width / viewBox.height
  const widthPx = Number((BASE_HEIGHT * aspectRatio).toFixed(3))

  return {
    aspectRatio,
    width: widthPx,
    height: BASE_HEIGHT,
    svg: svgMarkup
      .replace(/\s(?:role|focusable|aria-hidden)="[^"]*"/g, '')
      .replace(/\sdata-[^=]+="[^"]*"/g, '')
      .replace(/\sstyle="[^"]*"/g, '')
      .replace(/ width="[^"]*"/, ` width="${widthPx}"`)
      .replace(/ height="[^"]*"/, ` height="${BASE_HEIGHT}"`)
      .replace('<svg ', '<svg preserveAspectRatio="xMidYMid meet" ')
      .trim(),
  }
}

function escapeForSource(text) {
  return text
    .replace(/\\/g, '\\\\')
    .replace(/`/g, '\\`')
    .replace(/\$\{/g, '\\${')
}

async function ensureMathJax() {
  global.MathJax = {
    loader: {
      paths: { mathjax: '@mathjax/src/bundle' },
      load: ['adaptors/liteDOM'],
      require: (file) => import(file),
    },
    output: {
      font: 'mathjax-newcm',
      fontCache: 'local',
    },
  }

  await import('@mathjax/src/bundle/tex-svg.js')
  await MathJax.startup.promise
}

async function renderFormulaSvg(formula) {
  const node = await MathJax.tex2svgPromise(formula, {
    display: false,
    em: EM,
    ex: EX,
    containerWidth: CONTAINER_WIDTH,
  })

  const adaptor = MathJax.startup.adaptor
  const svgNode = adaptor.tags(node, 'svg')[0]
  return adaptor.serializeXML(svgNode)
}

async function main() {
  await fs.mkdir(assetDir, { recursive: true })
  await fs.mkdir(generatedDir, { recursive: true })
  await ensureMathJax()

  const entries = []

  for (let index = 0; index < BACKGROUND_FORMULAS.length; index += 1) {
    const formula = BACKGROUND_FORMULAS[index]
    const id = `formula-${String(index).padStart(3, '0')}`
    const fileName = `${id}.svg`
    const svgMarkup = await renderFormulaSvg(formula)
    const cleaned = cleanSvgMarkup(svgMarkup)
    await fs.writeFile(path.join(assetDir, fileName), `${cleaned.svg}\n`, 'utf8')

    entries.push({
      id,
      fileName,
      formula,
      width: cleaned.width,
      height: cleaned.height,
      aspectRatio: cleaned.aspectRatio,
    })
  }

  const moduleSource = `/* eslint-disable */
// This file is auto-generated by scripts/generate-background-formula-svgs.mjs.
// Do not edit it by hand.

${entries.map((entry, index) => `import sprite${index}Url from '../assets/background-formulas/${entry.fileName}'`).join('\n')}

export interface BackgroundFormulaSpriteAsset {
  id: string
  formula: string
  url: string
  width: number
  height: number
  aspectRatio: number
}

export const BACKGROUND_FORMULA_SPRITES: BackgroundFormulaSpriteAsset[] = [
${entries.map((entry, index) => `  {
    id: '${entry.id}',
    formula: String.raw\`${escapeForSource(entry.formula)}\`,
    url: sprite${index}Url,
    width: ${entry.width},
    height: ${entry.height},
    aspectRatio: ${entry.aspectRatio},
  },`).join('\n')}
]
`

  await fs.writeFile(generatedModulePath, moduleSource, 'utf8')
  MathJax.done()
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
