/**
 * Diagrama SVG de uma arvore de decisao.
 *
 * Cada no interno mostra a condicao (`atributo <= limiar`), o numero de
 * amostras e a impureza; as folhas sao coloridas pela classe majoritaria,
 * com a barra de distribuicao das classes.
 */
import { useMemo } from 'react'
import { cap, cn, corDaClasse, num } from '@/lib/utils'
import { Legenda } from './ui'

export interface NoArvore {
  folha: boolean
  n_amostras: number
  impureza: number
  distribuicao: Record<string, number>
  profundidade: number
  classe?: string
  atributo?: number
  limiar?: number
  ganho?: number
  esquerda?: NoArvore
  direita?: NoArvore
}

interface NoPosicionado {
  no: NoArvore
  x: number
  y: number
  paiX?: number
  paiY?: number
  ramo?: 'sim' | 'nao'
}

const LARGURA_NO = 132
const ALTURA_NO = 54
const ESPACO_V = 92

export function ArvoreDecisao({
  arvore,
  nomesFeatures,
  criterio = 'gini',
  className,
}: {
  arvore: NoArvore
  nomesFeatures: string[]
  criterio?: string
  className?: string
}) {
  const { nos, largura, altura } = useMemo(
    () => posicionar(arvore),
    [arvore],
  )

  const classes = useMemo(() => {
    const s = new Set<string>()
    const visitar = (n: NoArvore) => {
      Object.keys(n.distribuicao ?? {}).forEach((c) => s.add(c))
      if (n.esquerda) visitar(n.esquerda)
      if (n.direita) visitar(n.direita)
    }
    visitar(arvore)
    return [...s].sort()
  }, [arvore])

  return (
    <div className={cn('w-full', className)}>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${largura} ${altura}`}
          className="h-auto w-full"
          style={{ minWidth: Math.min(largura, 1100) }}
          role="img"
          aria-label="Diagrama da árvore de decisão"
        >
          {/* Arestas */}
          {nos.map((p, i) =>
            p.paiX !== undefined && p.paiY !== undefined ? (
              <g key={`aresta-${i}`}>
                <path
                  d={`M${p.paiX} ${p.paiY + ALTURA_NO / 2} C${p.paiX} ${p.paiY + ESPACO_V * 0.6}, ${p.x} ${p.y - ESPACO_V * 0.35}, ${p.x} ${p.y - ALTURA_NO / 2}`}
                  fill="none"
                  stroke="hsl(var(--border-strong))"
                  strokeWidth={1.4}
                />
                <text
                  x={(p.paiX + p.x) / 2 + (p.ramo === 'sim' ? -14 : 14)}
                  y={p.paiY + ESPACO_V * 0.52}
                  textAnchor="middle"
                  className="text-[9px] font-medium"
                  fill="hsl(var(--text-muted))"
                  paintOrder="stroke"
                  stroke="hsl(var(--surface))"
                  strokeWidth={3}
                >
                  {p.ramo === 'sim' ? 'sim' : 'não'}
                </text>
              </g>
            ) : null,
          )}

          {/* Nos */}
          {nos.map((p, i) => {
            const n = p.no
            const cor = n.folha ? corDaClasse(n.classe ?? '') : '#f59e0b'
            const total = Object.values(n.distribuicao ?? {}).reduce(
              (s, v) => s + v,
              0,
            )
            return (
              <g key={`no-${i}`}>
                <rect
                  x={p.x - LARGURA_NO / 2}
                  y={p.y - ALTURA_NO / 2}
                  width={LARGURA_NO}
                  height={ALTURA_NO}
                  rx={7}
                  fill={cor}
                  fillOpacity={n.folha ? 0.16 : 0.09}
                  stroke={cor}
                  strokeWidth={n.folha ? 1.8 : 1.3}
                />

                {n.folha ? (
                  <>
                    <text
                      x={p.x}
                      y={p.y - 11}
                      textAnchor="middle"
                      className="text-[11px] font-bold"
                      fill={cor}
                    >
                      {cap(n.classe ?? '')}
                    </text>
                    <text
                      x={p.x}
                      y={p.y + 2}
                      textAnchor="middle"
                      className="text-[9px]"
                      fill="hsl(var(--text-muted))"
                    >
                      n={n.n_amostras} · {criterio}={num(n.impureza, 3)}
                    </text>
                  </>
                ) : (
                  <>
                    <text
                      x={p.x}
                      y={p.y - 11}
                      textAnchor="middle"
                      className="text-[10px] font-semibold"
                      fill="hsl(var(--text-primary))"
                    >
                      {abreviar(nomesFeatures[n.atributo ?? 0])} ≤{' '}
                      {num(n.limiar, 2)}
                    </text>
                    <text
                      x={p.x}
                      y={p.y + 2}
                      textAnchor="middle"
                      className="text-[9px]"
                      fill="hsl(var(--text-muted))"
                    >
                      n={n.n_amostras} · ganho={num(n.ganho, 3)}
                    </text>
                  </>
                )}

                {/* Barra de distribuição das classes */}
                {total > 0 && (
                  <g>
                    {(() => {
                      let deslocamento = 0
                      return classes.map((c) => {
                        const v = n.distribuicao?.[c] ?? 0
                        if (v === 0) return null
                        const w = (v / total) * (LARGURA_NO - 20)
                        const x = p.x - (LARGURA_NO - 20) / 2 + deslocamento
                        deslocamento += w
                        return (
                          <rect
                            key={c}
                            x={x}
                            y={p.y + 10}
                            width={w}
                            height={5}
                            rx={1.5}
                            fill={corDaClasse(c)}
                            opacity={0.85}
                          >
                            <title>{`${cap(c)}: ${v}`}</title>
                          </rect>
                        )
                      })
                    })()}
                  </g>
                )}
              </g>
            )
          })}
        </svg>
      </div>

      <Legenda
        className="mt-3"
        itens={[
          {
            cor: '#f59e0b',
            forma: 'quadrado',
            rotulo: 'nó de decisão',
            descricao: 'condição atributo ≤ limiar',
          },
          ...classes.map((c) => ({
            cor: corDaClasse(c),
            forma: 'quadrado' as const,
            rotulo: `folha ${cap(c)}`,
          })),
          {
            cor: 'hsl(var(--text-muted))',
            forma: 'quadrado',
            rotulo: 'barra inferior',
            descricao: 'distribuição das classes no nó',
          },
        ]}
      />
    </div>
  )
}

/** Layout: percorre em ordem e distribui as folhas horizontalmente. */
function posicionar(raiz: NoArvore) {
  const nos: NoPosicionado[] = []
  let proximaColuna = 0

  const visitar = (
    no: NoArvore,
    profundidade: number,
    paiX?: number,
    paiY?: number,
    ramo?: 'sim' | 'nao',
  ): number => {
    const y = profundidade * ESPACO_V + ALTURA_NO / 2 + 16

    if (no.folha) {
      const x = proximaColuna * (LARGURA_NO + 16) + LARGURA_NO / 2 + 10
      proximaColuna += 1
      nos.push({ no, x, y, paiX, paiY, ramo })
      return x
    }

    // Reserva o lugar do nó antes de descer, para o pai ficar centralizado
    const indice = nos.length
    nos.push({ no, x: 0, y, paiX, paiY, ramo })

    const xEsq = visitar(no.esquerda!, profundidade + 1, undefined, undefined, 'sim')
    const xDir = visitar(no.direita!, profundidade + 1, undefined, undefined, 'nao')
    const x = (xEsq + xDir) / 2
    nos[indice].x = x

    // Religa os filhos ao pai agora que a posição dele é conhecida
    for (const p of nos) {
      if (p.no === no.esquerda || p.no === no.direita) {
        p.paiX = x
        p.paiY = y
      }
    }
    return x
  }

  visitar(raiz, 0)

  const largura = Math.max(
    proximaColuna * (LARGURA_NO + 16) + 20,
    LARGURA_NO + 40,
  )
  const profundidadeMax = Math.max(...nos.map((p) => p.y)) + ALTURA_NO / 2 + 16
  return { nos, largura, altura: profundidadeMax }
}

function abreviar(nome?: string) {
  if (!nome) return '?'
  return nome
    .replace('Comprimento da ', 'Comp. ')
    .replace('Largura da ', 'Larg. ')
}
