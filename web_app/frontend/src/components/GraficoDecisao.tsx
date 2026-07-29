/**
 * Grafico de dispersao com regioes de decisao.
 *
 * As regioes vem do backend como uma grade de indices de classe. Elas sao
 * pintadas numa canvas na resolucao da grade e ampliadas com suavizacao,
 * o que evita o efeito "escadinha" nas fronteiras curvas. Quando o backend
 * envia as superficies de diferenca de score, a fronteira exata (nivel zero)
 * e tracada por marching squares — resultando em curvas realmente lisas.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Amostra, Limites } from '@/lib/types'
import { cap, cn, corDaClasse } from '@/lib/utils'
import { Legenda } from './ui'

export interface RetaFronteira {
  w: number[]
  b: number
  rotulo?: string
}

interface Props {
  amostras: Amostra[]
  limites: Limites
  eixoX: string
  eixoY: string
  /** Grade de regioes: grade[linha][coluna] = indice da classe. */
  grade?: number[][]
  classesGrade?: string[]
  /** Superficies de diferenca de score por par ("a|b"), para a curva exata. */
  superficies?: Record<string, number[][]>
  /** Retas w·x + b = 0 (fronteiras lineares). */
  retas?: RetaFronteira[]
  /** Marcadores extras (ex.: protótipos das classes). */
  marcadores?: { x: number; y: number; classe: string; rotulo: string }[]
  /** Ponto consultado pelo usuário. */
  destaque?: { x: number; y: number; classe?: string } | null
  altura?: number
  mostrarTreinoTeste?: boolean
  onClicar?: (x: number, y: number) => void
}

const MARGEM = { esquerda: 52, direita: 16, topo: 16, base: 44 }

export function GraficoDecisao({
  amostras,
  limites,
  eixoX,
  eixoY,
  grade,
  classesGrade = [],
  superficies,
  retas,
  marcadores,
  destaque,
  altura = 420,
  mostrarTreinoTeste = true,
  onClicar,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [largura, setLargura] = useState(720)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entrada]) => {
      setLargura(Math.max(320, entrada.contentRect.width))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const areaW = largura - MARGEM.esquerda - MARGEM.direita
  const areaH = altura - MARGEM.topo - MARGEM.base

  const escala = useMemo(() => {
    const { x_min, x_max, y_min, y_max } = limites
    const spanX = x_max - x_min || 1
    const spanY = y_max - y_min || 1
    return {
      px: (x: number) => MARGEM.esquerda + ((x - x_min) / spanX) * areaW,
      py: (y: number) => MARGEM.topo + areaH - ((y - y_min) / spanY) * areaH,
      inversoX: (px: number) => x_min + ((px - MARGEM.esquerda) / areaW) * spanX,
      inversoY: (py: number) =>
        y_min + ((MARGEM.topo + areaH - py) / areaH) * spanY,
    }
  }, [limites, areaW, areaH])

  /* --- Pintura das regioes de decisao (canvas, com suavizacao) ------------ */
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = largura * dpr
    canvas.height = altura * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, largura, altura)

    if (!grade?.length) return

    const linhas = grade.length
    const colunas = grade[0].length

    // 1) Desenha a grade na resolucao nativa numa canvas auxiliar
    const aux = document.createElement('canvas')
    aux.width = colunas
    aux.height = linhas
    const auxCtx = aux.getContext('2d')
    if (!auxCtx) return

    const img = auxCtx.createImageData(colunas, linhas)
    for (let i = 0; i < linhas; i++) {
      for (let j = 0; j < colunas; j++) {
        // A grade vem de baixo para cima (eixo Y crescente); a imagem e de cima
        // para baixo — por isso a linha e invertida aqui.
        const cor = corDaClasse(classesGrade[grade[linhas - 1 - i][j]] ?? '')
        const p = (i * colunas + j) * 4
        img.data[p] = parseInt(cor.slice(1, 3), 16)
        img.data[p + 1] = parseInt(cor.slice(3, 5), 16)
        img.data[p + 2] = parseInt(cor.slice(5, 7), 16)
        img.data[p + 3] = 56 // opacidade suave, para os pontos se destacarem
      }
    }
    auxCtx.putImageData(img, 0, 0)

    // 2) Amplia com interpolacao — fronteiras sem serrilhado
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.save()
    ctx.beginPath()
    ctx.rect(MARGEM.esquerda, MARGEM.topo, areaW, areaH)
    ctx.clip()
    ctx.drawImage(aux, MARGEM.esquerda, MARGEM.topo, areaW, areaH)
    ctx.restore()
  }, [grade, classesGrade, largura, altura, areaW, areaH])

  /* --- Curvas de nivel zero (fronteiras exatas) --------------------------- */
  const curvas = useMemo(() => {
    if (!superficies || !grade?.length) return []
    const linhas = grade.length
    const colunas = grade[0].length
    const { x_min, x_max, y_min, y_max } = limites
    const dx = (x_max - x_min) / (colunas - 1)
    const dy = (y_max - y_min) / (linhas - 1)

    const resultado: { d: string; chave: string }[] = []
    for (const [chave, campo] of Object.entries(superficies)) {
      const segmentos = marchingSquares(campo, dx, dy, x_min, y_min)
      // Mantem apenas a fronteira efetivamente visivel entre as duas regioes
      const d = segmentos
        .filter(([xa, ya, xb, yb]) => {
          const [ca, cb] = chave.split('|')
          return (
            fronteiraVisivel(xa, ya, ca, cb, grade, classesGrade, limites) ||
            fronteiraVisivel(xb, yb, ca, cb, grade, classesGrade, limites)
          )
        })
        .map(
          ([xa, ya, xb, yb]) =>
            `M${escala.px(xa).toFixed(1)} ${escala.py(ya).toFixed(1)}L${escala
              .px(xb)
              .toFixed(1)} ${escala.py(yb).toFixed(1)}`,
        )
        .join('')
      if (d) resultado.push({ d, chave })
    }
    return resultado
  }, [superficies, grade, classesGrade, limites, escala])

  const ticksX = useMemo(() => gerarTicks(limites.x_min, limites.x_max), [limites])
  const ticksY = useMemo(() => gerarTicks(limites.y_min, limites.y_max), [limites])

  const classesPresentes = useMemo(
    () => [...new Set(amostras.map((a) => a.classe))],
    [amostras],
  )

  return (
    <div ref={containerRef} className="w-full">
      <div className="relative" style={{ height: altura }}>
        <canvas
          ref={canvasRef}
          className="absolute inset-0"
          style={{ width: largura, height: altura }}
          aria-hidden
        />
        <svg
          width={largura}
          height={altura}
          className={cn('absolute inset-0', onClicar && 'cursor-crosshair')}
          onClick={
            onClicar
              ? (e) => {
                  const r = e.currentTarget.getBoundingClientRect()
                  const px = e.clientX - r.left
                  const py = e.clientY - r.top
                  if (
                    px >= MARGEM.esquerda &&
                    px <= MARGEM.esquerda + areaW &&
                    py >= MARGEM.topo &&
                    py <= MARGEM.topo + areaH
                  ) {
                    onClicar(escala.inversoX(px), escala.inversoY(py))
                  }
                }
              : undefined
          }
          role="img"
          aria-label={`Dispersão de ${eixoX} por ${eixoY} com regiões de decisão`}
        >
          {/* Grade de fundo */}
          <g opacity={0.5}>
            {ticksX.map((t) => (
              <line
                key={`gx-${t}`}
                x1={escala.px(t)}
                y1={MARGEM.topo}
                x2={escala.px(t)}
                y2={MARGEM.topo + areaH}
                stroke="hsl(var(--grid-line))"
                strokeWidth={1}
              />
            ))}
            {ticksY.map((t) => (
              <line
                key={`gy-${t}`}
                x1={MARGEM.esquerda}
                y1={escala.py(t)}
                x2={MARGEM.esquerda + areaW}
                y2={escala.py(t)}
                stroke="hsl(var(--grid-line))"
                strokeWidth={1}
              />
            ))}
          </g>

          {/* Curvas de fronteira exatas (Bayes) */}
          {curvas.map(({ d, chave }) => (
            <path
              key={chave}
              d={d}
              fill="none"
              stroke="hsl(var(--text-primary))"
              strokeWidth={1.7}
              strokeDasharray="5 4"
              strokeLinecap="round"
              opacity={0.75}
            />
          ))}

          {/* Retas de fronteira lineares */}
          {retas?.map((reta, i) => {
            const pontos = pontosDaReta(reta, limites)
            if (!pontos) return null
            return (
              <line
                key={i}
                x1={escala.px(pontos[0].x)}
                y1={escala.py(pontos[0].y)}
                x2={escala.px(pontos[1].x)}
                y2={escala.py(pontos[1].y)}
                stroke="hsl(var(--text-primary))"
                strokeWidth={1.8}
                strokeDasharray="6 4"
                strokeLinecap="round"
                opacity={0.8}
              />
            )
          })}

          {/* Amostras */}
          {amostras.map((a, i) => {
            const cor = corDaClasse(a.classe)
            const teste = mostrarTreinoTeste && !a.treino
            return (
              <circle
                key={i}
                cx={escala.px(a.x)}
                cy={escala.py(a.y)}
                r={teste ? 4.5 : 3.6}
                fill={teste ? 'none' : cor}
                stroke={cor}
                strokeWidth={teste ? 1.8 : 0.8}
                opacity={0.92}
              >
                <title>{`${cap(a.classe)} · (${a.x.toFixed(2)}, ${a.y.toFixed(2)}) · ${a.treino ? 'treino' : 'teste'}`}</title>
              </circle>
            )
          })}

          {/* Protótipos / marcadores */}
          {marcadores?.map((m, i) => (
            <g key={i}>
              <path
                d={estrela(escala.px(m.x), escala.py(m.y), 9, 4.5, 5)}
                fill={corDaClasse(m.classe)}
                stroke="hsl(var(--surface))"
                strokeWidth={1.6}
              >
                <title>{m.rotulo}</title>
              </path>
            </g>
          ))}

          {/* Ponto consultado */}
          {destaque && (
            <g>
              <circle
                cx={escala.px(destaque.x)}
                cy={escala.py(destaque.y)}
                r={11}
                fill="none"
                stroke={
                  destaque.classe ? corDaClasse(destaque.classe) : '#f59e0b'
                }
                strokeWidth={2}
                opacity={0.55}
              >
                <animate
                  attributeName="r"
                  values="8;13;8"
                  dur="1.8s"
                  repeatCount="indefinite"
                />
              </circle>
              <circle
                cx={escala.px(destaque.x)}
                cy={escala.py(destaque.y)}
                r={4}
                fill={destaque.classe ? corDaClasse(destaque.classe) : '#f59e0b'}
                stroke="hsl(var(--surface))"
                strokeWidth={1.5}
              />
            </g>
          )}

          {/* Eixos */}
          <line
            x1={MARGEM.esquerda}
            y1={MARGEM.topo + areaH}
            x2={MARGEM.esquerda + areaW}
            y2={MARGEM.topo + areaH}
            stroke="hsl(var(--border-strong))"
          />
          <line
            x1={MARGEM.esquerda}
            y1={MARGEM.topo}
            x2={MARGEM.esquerda}
            y2={MARGEM.topo + areaH}
            stroke="hsl(var(--border-strong))"
          />

          {ticksX.map((t) => (
            <text
              key={`tx-${t}`}
              x={escala.px(t)}
              y={MARGEM.topo + areaH + 16}
              textAnchor="middle"
              className="fill-[hsl(var(--text-muted))] text-[10px] tabular"
            >
              {t.toFixed(1)}
            </text>
          ))}
          {ticksY.map((t) => (
            <text
              key={`ty-${t}`}
              x={MARGEM.esquerda - 8}
              y={escala.py(t) + 3.5}
              textAnchor="end"
              className="fill-[hsl(var(--text-muted))] text-[10px] tabular"
            >
              {t.toFixed(1)}
            </text>
          ))}

          <text
            x={MARGEM.esquerda + areaW / 2}
            y={altura - 6}
            textAnchor="middle"
            className="fill-[hsl(var(--text-secondary))] text-[11px] font-medium"
          >
            {eixoX}
          </text>
          <text
            x={-(MARGEM.topo + areaH / 2)}
            y={13}
            textAnchor="middle"
            transform="rotate(-90)"
            className="fill-[hsl(var(--text-secondary))] text-[11px] font-medium"
          >
            {eixoY}
          </text>
        </svg>
      </div>

      <Legenda
        className="mt-3"
        titulo="Legenda"
        itens={[
          ...classesPresentes.map((c) => ({
            cor: corDaClasse(c),
            forma: 'circulo' as const,
            rotulo: cap(c),
          })),
          ...(mostrarTreinoTeste
            ? [
                {
                  cor: 'hsl(var(--text-muted))',
                  forma: 'circulo' as const,
                  rotulo: 'preenchido = treino',
                },
                {
                  cor: 'transparent',
                  forma: 'circulo' as const,
                  rotulo: 'vazado = teste',
                },
              ]
            : []),
          ...(grade?.length
            ? [
                {
                  cor: 'hsl(var(--text-muted))',
                  forma: 'quadrado' as const,
                  rotulo: 'fundo colorido = região de decisão da classe',
                },
              ]
            : []),
          ...(curvas.length || retas?.length
            ? [
                {
                  cor: 'hsl(var(--text-primary))',
                  forma: 'linha-tracejada' as const,
                  rotulo: 'fronteira de decisão',
                },
              ]
            : []),
          ...(marcadores?.length
            ? [
                {
                  cor: 'hsl(var(--text-muted))',
                  forma: 'triangulo' as const,
                  rotulo: 'estrela = protótipo (vetor médio)',
                },
              ]
            : []),
        ]}
      />
    </div>
  )
}

/* ------------------------------------------------------------ utilitarios -- */

/** Marching squares no nivel zero — devolve segmentos [xa, ya, xb, yb]. */
function marchingSquares(
  campo: number[][],
  dx: number,
  dy: number,
  x0: number,
  y0: number,
): [number, number, number, number][] {
  const segmentos: [number, number, number, number][] = []
  const linhas = campo.length
  const colunas = campo[0].length

  const interp = (v1: number, v2: number) =>
    Math.abs(v1 - v2) < 1e-12 ? 0.5 : v1 / (v1 - v2)

  for (let i = 0; i < linhas - 1; i++) {
    for (let j = 0; j < colunas - 1; j++) {
      const v = [campo[i][j], campo[i][j + 1], campo[i + 1][j + 1], campo[i + 1][j]]
      let caso = 0
      if (v[0] > 0) caso |= 1
      if (v[1] > 0) caso |= 2
      if (v[2] > 0) caso |= 4
      if (v[3] > 0) caso |= 8
      if (caso === 0 || caso === 15) continue

      const px = (c: number) => x0 + (j + c) * dx
      const py = (l: number) => y0 + (i + l) * dy

      const arestas: Record<string, [number, number]> = {
        base: [px(interp(v[0], v[1])), py(0)],
        direita: [px(1), py(interp(v[1], v[2]))],
        topo: [px(interp(v[3], v[2])), py(1)],
        esquerda: [px(0), py(interp(v[0], v[3]))],
      }

      const ligacoes: Record<number, [string, string][]> = {
        1: [['esquerda', 'base']],
        2: [['base', 'direita']],
        3: [['esquerda', 'direita']],
        4: [['direita', 'topo']],
        5: [
          ['esquerda', 'topo'],
          ['base', 'direita'],
        ],
        6: [['base', 'topo']],
        7: [['esquerda', 'topo']],
        8: [['topo', 'esquerda']],
        9: [['topo', 'base']],
        10: [
          ['topo', 'direita'],
          ['esquerda', 'base'],
        ],
        11: [['topo', 'direita']],
        12: [['direita', 'esquerda']],
        13: [['direita', 'base']],
        14: [['base', 'esquerda']],
      }

      for (const [a, b] of ligacoes[caso] ?? []) {
        segmentos.push([
          arestas[a][0],
          arestas[a][1],
          arestas[b][0],
          arestas[b][1],
        ])
      }
    }
  }
  return segmentos
}

/**
 * A superficie a-b tem nivel zero em toda parte onde as duas classes empatam,
 * inclusive em regioes dominadas por uma terceira classe. Aqui verificamos se
 * o ponto realmente fica na divisa visivel entre `ca` e `cb`.
 */
function fronteiraVisivel(
  x: number,
  y: number,
  ca: string,
  cb: string,
  grade: number[][],
  classes: string[],
  limites: Limites,
): boolean {
  const linhas = grade.length
  const colunas = grade[0].length
  const j = Math.round(
    ((x - limites.x_min) / (limites.x_max - limites.x_min)) * (colunas - 1),
  )
  const i = Math.round(
    ((y - limites.y_min) / (limites.y_max - limites.y_min)) * (linhas - 1),
  )
  const vizinhos = new Set<string>()
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      const li = i + di
      const lj = j + dj
      if (li < 0 || li >= linhas || lj < 0 || lj >= colunas) continue
      vizinhos.add(classes[grade[li][lj]] ?? '')
    }
  }
  return vizinhos.has(ca) && vizinhos.has(cb)
}

/** Intersecao da reta w·x + b = 0 com a janela de plotagem. */
function pontosDaReta(
  reta: RetaFronteira,
  lim: Limites,
): [{ x: number; y: number }, { x: number; y: number }] | null {
  const [w1, w2] = reta.w
  const b = reta.b
  if (Math.abs(w2) > 1e-9) {
    return [
      { x: lim.x_min, y: (-w1 * lim.x_min - b) / w2 },
      { x: lim.x_max, y: (-w1 * lim.x_max - b) / w2 },
    ]
  }
  if (Math.abs(w1) > 1e-9) {
    const x = -b / w1
    return [
      { x, y: lim.y_min },
      { x, y: lim.y_max },
    ]
  }
  return null
}

function gerarTicks(min: number, max: number, alvo = 6): number[] {
  const span = max - min
  if (span <= 0) return [min]
  const bruto = span / alvo
  const mag = Math.pow(10, Math.floor(Math.log10(bruto)))
  const norm = bruto / mag
  const passo = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag
  const ticks: number[] = []
  for (let t = Math.ceil(min / passo) * passo; t <= max + 1e-9; t += passo) {
    ticks.push(Number(t.toFixed(10)))
  }
  return ticks
}

function estrela(
  cx: number,
  cy: number,
  raioExterno: number,
  raioInterno: number,
  pontas: number,
): string {
  let d = ''
  for (let i = 0; i < pontas * 2; i++) {
    const r = i % 2 === 0 ? raioExterno : raioInterno
    const ang = (Math.PI / pontas) * i - Math.PI / 2
    d += `${i === 0 ? 'M' : 'L'}${(cx + r * Math.cos(ang)).toFixed(2)} ${(cy + r * Math.sin(ang)).toFixed(2)}`
  }
  return `${d}Z`
}
