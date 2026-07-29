/**
 * Lab 5.0 — XOR com MLP (slides 36-37 da Aula PR_711).
 *
 * Exemplo didatico do slide 37 + exercicio do XOR, com treino interativo,
 * superficie de saida da rede e memoria de calculo em LaTeX.
 */
import { useMutation, useQuery } from '@tanstack/react-query'
import { CheckCircle2, FileText, Play, RotateCcw, XCircle } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { DiagramaRede } from '@/components/DiagramaRede'
import { BlocoFormula } from '@/components/Formula'
import { GraficoLinha } from '@/components/GraficoLinha'
import { MemoriaCalculo } from '@/components/MemoriaCalculo'
import {
  Botao,
  Card,
  Carregando,
  ErroBox,
  Legenda,
  Metrica,
  Nota,
} from '@/components/ui'
import { api } from '@/lib/api'
import type { EstadoXor } from '@/lib/types'
import { cn, escalaDivergente, num } from '@/lib/utils'

export function PaginaLab50() {
  const [exercicio, setExercicio] = useState<string | null>(null)
  const [estado, setEstado] = useState<EstadoXor | null>(null)
  const [epocaAtual, setEpocaAtual] = useState(0)
  const [historico, setHistorico] = useState<number[]>([])

  const inicial = useQuery({
    queryKey: ['lab5', 'xor', 'inicial'],
    queryFn: () => api.lab5.xorInicial({ resolucao: 70 }),
  })

  useEffect(() => {
    if (inicial.data && !estado) {
      setEstado(inicial.data)
      setEpocaAtual(0)
      setHistorico([])
    }
  }, [inicial.data, estado])

  const memoria = useQuery({
    queryKey: ['lab5', 'memoria', exercicio],
    queryFn: () => api.lab5.memoria(exercicio!),
    enabled: !!exercicio,
  })

  const treinar = useMutation({
    mutationFn: (epocas: number) =>
      api.lab5.xorTreinar({
        epocas,
        taxa: 0.5,
        resolucao: 70,
        pesos_oculta: estado?.pesos.oculta,
        bias_oculta: estado?.pesos.bias_oculta,
        pesos_saida: estado?.pesos.saida,
        bias_saida: estado?.pesos.bias_saida,
      }),
    onSuccess: (novo, epocas) => {
      setEstado(novo)
      setEpocaAtual((e) => e + epocas)
      setHistorico((h) => [...h, ...novo.historico])
    },
  })

  const reiniciar = () => {
    if (inicial.data) {
      setEstado(inicial.data)
      setEpocaAtual(0)
      setHistorico([])
    }
  }

  const rodarExercicio = () => {
    reiniciar()
    // Roda exatamente 1 epoca a partir dos pesos iniciais — como pede o slide
    api.lab5
      .xorTreinar({ epocas: 1, taxa: 0.5, resolucao: 70 })
      .then((novo) => {
        setEstado(novo)
        setEpocaAtual(1)
        setHistorico(novo.historico)
      })
  }

  const dadosCurva = useMemo(
    () => historico.map((v, i) => ({ x: i + 1, erro: v })),
    [historico],
  )

  return (
    <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
      {/* -------------------------------------------------------- controles */}
      <div className="space-y-5">
        <Card titulo="sobre o lab 5.0">
          <p className="text-sm leading-relaxed text-secondary">
            O exercício do <strong>slide 36</strong> pede para resolver o XOR
            com uma MLP na arquitetura mínima da Fig. 12.28(b), treinando por{' '}
            <strong>apenas 1 época</strong>.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-secondary">
            O <strong>slide 37</strong> não resolve o XOR — é um exemplo
            didático genérico que demonstra a conta do backpropagation passo a
            passo, e serve de base antes de aplicá-la aqui.
          </p>
        </Card>

        <Card titulo="memórias de cálculo">
          <div className="space-y-2">
            <Botao
              variante="primario"
              className="w-full"
              onClick={() => setExercicio('didatico')}
            >
              <FileText size={15} />
              Exemplo didático (slide 37)
            </Botao>
            <Botao
              variante="secundario"
              className="w-full"
              onClick={() => setExercicio('xor')}
            >
              <FileText size={15} />
              Exercício XOR (slide 36)
            </Botao>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Cada janela traz o diagrama da rede, as fórmulas em LaTeX e a
            substituição numérica de cada etapa — forward, erro, deltas,
            atualização dos pesos e nova predição.
          </p>
        </Card>

        <Card titulo="treinamento interativo">
          <div className="space-y-2">
            <Botao
              variante="primario"
              className="w-full"
              onClick={rodarExercicio}
              disabled={treinar.isPending}
            >
              <Play size={15} />
              Rodar 1 época (o exercício)
            </Botao>
            <div className="grid grid-cols-2 gap-2">
              <Botao
                onClick={() => treinar.mutate(500)}
                disabled={treinar.isPending}
              >
                +500 épocas
              </Botao>
              <Botao
                onClick={() => treinar.mutate(2000)}
                disabled={treinar.isPending}
              >
                +2000 épocas
              </Botao>
            </div>
            <Botao
              variante="fantasma"
              className="w-full"
              onClick={reiniciar}
              disabled={treinar.isPending}
            >
              <RotateCcw size={14} />
              Reiniciar rede
            </Botao>
          </div>
          <Nota tom="atencao" className="mt-4">
            Com estes pesos iniciais o erro fica quase parado até a época ~500 e
            só converge de fato entre 2000 e 5000 — algo que a Regra Delta
            linear (Lab 2) <strong>nunca</strong> consegue fazer.
          </Nota>
        </Card>
      </div>

      {/* ------------------------------------------------------- resultados */}
      <div className="space-y-6">
        {inicial.isPending && (
          <Card>
            <Carregando texto="Inicializando a rede…" />
          </Card>
        )}
        {inicial.error && <ErroBox erro={inicial.error} />}

        {estado && (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Metrica
                rotulo="Época atual"
                valor={epocaAtual}
                detalhe={treinar.isPending ? 'treinando…' : undefined}
              />
              <Metrica
                rotulo="Erro médio da época"
                valor={
                  estado.erro_medio !== null ? num(estado.erro_medio, 5) : '—'
                }
              />
              <Metrica
                rotulo="Padrões corretos"
                valor={`${estado.acertos} / 4`}
                destaque={
                  estado.acertos === 4
                    ? 'bom'
                    : estado.acertos >= 2
                      ? 'medio'
                      : 'ruim'
                }
              />
            </div>

            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,420px)]">
              <Card titulo="saída da rede sobre o plano x₁ × x₂">
                <SuperficieXor estado={estado} />
              </Card>

              <Card titulo="convergência">
                {dadosCurva.length ? (
                  <>
                    <GraficoLinha
                      dados={dadosCurva}
                      series={[
                        { chave: 'erro', rotulo: 'erro médio', cor: '#f59e0b' },
                      ]}
                      rotuloX="época"
                      rotuloY="erro médio"
                      altura={300}
                    />
                    <Legenda
                      className="mt-2"
                      itens={[
                        {
                          cor: '#f59e0b',
                          forma: 'linha',
                          rotulo: 'erro médio da época',
                          descricao: 'média dos 4 padrões',
                        },
                      ]}
                    />
                  </>
                ) : (
                  <div className="flex h-[300px] items-center justify-center text-sm text-muted">
                    Treine a rede para ver a curva
                  </div>
                )}
              </Card>
            </div>

            <Card titulo="tabela-verdade e saídas atuais">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-subtle">
                      <th className="py-2 text-left text-[11px] font-semibold text-muted">
                        Padrão (x₁, x₂)
                      </th>
                      <th className="py-2 text-right text-[11px] font-semibold text-muted">
                        Alvo
                      </th>
                      <th className="py-2 text-right text-[11px] font-semibold text-muted">
                        Saída da rede
                      </th>
                      <th className="py-2 text-right text-[11px] font-semibold text-muted">
                        Classe prevista
                      </th>
                      <th className="py-2 text-right text-[11px] font-semibold text-muted">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {estado.resultados.map((r, i) => (
                      <tr key={i} className="border-b border-subtle/60 last:border-0">
                        <td className="py-2.5 font-mono text-secondary">
                          ({r.entrada.map((v) => v.toFixed(0)).join(', ')})
                        </td>
                        <td className="py-2.5 text-right tabular text-secondary">
                          {r.alvo}
                        </td>
                        <td className="py-2.5 text-right tabular font-semibold text-primary">
                          {num(r.saida, 4)}
                        </td>
                        <td className="py-2.5 text-right tabular text-secondary">
                          {r.previsto}
                        </td>
                        <td className="py-2.5 text-right">
                          <span
                            className={cn(
                              'inline-flex items-center gap-1 text-xs font-medium',
                              r.correto
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400',
                            )}
                          >
                            {r.correto ? (
                              <CheckCircle2 size={14} />
                            ) : (
                              <XCircle size={14} />
                            )}
                            {r.correto ? 'correto' : 'incorreto'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card titulo="arquitetura atual da rede">
              <DiagramaRede arquitetura={estado.arquitetura} />
              <BlocoFormula
                className="mt-4"
                titulo="a solução do XOR"
                latex={String.raw`\text{XOR}(x_1,x_2) = \sigma\!\left(w_7 h_1 + w_8 h_2 + b\right),\quad
                  h_i = \sigma\!\left(w_{i1}x_1 + w_{i2}x_2 + b_i\right)`}
                explicacao="A camada oculta transforma o espaço de entrada — no espaço de h₁ × h₂ o problema passa a ser linearmente separável."
              />
            </Card>
          </>
        )}
      </div>

      {exercicio && (
        <MemoriaCalculo
          traco={memoria.data}
          carregando={memoria.isPending}
          erro={memoria.error}
          onFechar={() => setExercicio(null)}
        />
      )}
    </div>
  )
}

/* ------------------------------------- superficie de saida (canvas + pontos) */
function SuperficieXor({ estado }: { estado: EstadoXor }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tamanho = 400
  const margem = 42

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = tamanho * dpr
    canvas.height = tamanho * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, tamanho, tamanho)

    const n = estado.superficie.length
    const aux = document.createElement('canvas')
    aux.width = n
    aux.height = n
    const auxCtx = aux.getContext('2d')
    if (!auxCtx) return

    const img = auxCtx.createImageData(n, n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // A superficie vem de baixo para cima; a imagem, de cima para baixo
        const cor = escalaDivergente(estado.superficie[n - 1 - i][j])
        const p = (i * n + j) * 4
        img.data[p] = parseInt(cor.slice(1, 3), 16)
        img.data[p + 1] = parseInt(cor.slice(3, 5), 16)
        img.data[p + 2] = parseInt(cor.slice(5, 7), 16)
        img.data[p + 3] = 235
      }
    }
    auxCtx.putImageData(img, 0, 0)

    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    const area = tamanho - margem - 12
    ctx.drawImage(aux, margem, 12, area, area)
  }, [estado.superficie])

  const lo = estado.limites.min
  const hi = estado.limites.max
  const area = tamanho - margem - 12
  const px = (x: number) => margem + ((x - lo) / (hi - lo)) * area
  const py = (y: number) => 12 + area - ((y - lo) / (hi - lo)) * area

  return (
    <div>
      <div className="flex flex-wrap items-start gap-4">
        <div className="relative shrink-0" style={{ width: tamanho, height: tamanho }}>
          <canvas
            ref={canvasRef}
            className="absolute inset-0 rounded"
            style={{ width: tamanho, height: tamanho }}
            aria-hidden
          />
          <svg
            width={tamanho}
            height={tamanho}
            className="absolute inset-0"
            role="img"
            aria-label="Superfície de saída da rede sobre o plano x1 × x2"
          >
            {estado.resultados.map((r, i) => {
              const [x1, x2] = r.entrada
              const cor = r.alvo === 1 ? '#f43f5e' : '#0ea5e9'
              const borda = r.correto ? '#15803d' : '#dc2626'
              return (
                <g key={i}>
                  {r.alvo === 1 ? (
                    <polygon
                      points={`${px(x1)},${py(x2) - 9} ${px(x1) + 8},${py(x2) + 6} ${px(x1) - 8},${py(x2) + 6}`}
                      fill={cor}
                      stroke={borda}
                      strokeWidth={2.5}
                    />
                  ) : (
                    <circle
                      cx={px(x1)}
                      cy={py(x2)}
                      r={8}
                      fill={cor}
                      stroke={borda}
                      strokeWidth={2.5}
                    />
                  )}
                  <text
                    x={px(x1) + 13}
                    y={py(x2) - 6}
                    className="text-[9px] tabular"
                    fill="hsl(var(--text-secondary))"
                    paintOrder="stroke"
                    stroke="hsl(var(--surface))"
                    strokeWidth={3}
                  >
                    {num(r.saida, 3)}
                  </text>
                </g>
              )
            })}

            {/* Eixos */}
            {[0, 0.5, 1].map((t) => (
              <g key={t}>
                <text
                  x={px(t)}
                  y={tamanho - 20}
                  textAnchor="middle"
                  className="text-[10px] tabular"
                  fill="hsl(var(--text-muted))"
                >
                  {t}
                </text>
                <text
                  x={margem - 8}
                  y={py(t) + 3}
                  textAnchor="end"
                  className="text-[10px] tabular"
                  fill="hsl(var(--text-muted))"
                >
                  {t}
                </text>
              </g>
            ))}
            <text
              x={margem + area / 2}
              y={tamanho - 4}
              textAnchor="middle"
              className="text-[11px] font-medium"
              fill="hsl(var(--text-secondary))"
            >
              x₁
            </text>
            <text
              x={-(12 + area / 2)}
              y={13}
              textAnchor="middle"
              transform="rotate(-90)"
              className="text-[11px] font-medium"
              fill="hsl(var(--text-secondary))"
            >
              x₂
            </text>
          </svg>
        </div>

        {/* Barra de cores */}
        <div className="flex items-center gap-2 pt-3">
          <div
            className="h-[260px] w-4 rounded border border-subtle"
            style={{
              background: `linear-gradient(to top, ${escalaDivergente(0)}, ${escalaDivergente(0.5)}, ${escalaDivergente(1)})`,
            }}
            aria-hidden
          />
          <div className="flex h-[260px] flex-col justify-between text-[10px] tabular text-muted">
            <span>1,0</span>
            <span>0,5</span>
            <span>0,0</span>
          </div>
          <span className="ml-1 text-[10px] leading-tight text-muted">
            saída
            <br />
            da rede
          </span>
        </div>
      </div>

      <Legenda
        className="mt-4"
        titulo="Legenda"
        itens={[
          { cor: '#0ea5e9', forma: 'quadrado', rotulo: 'azul', descricao: 'saída perto de 0 (classe 0)' },
          { cor: '#ffffff', forma: 'quadrado', rotulo: 'branco', descricao: 'saída perto de 0,5 (ambíguo)' },
          { cor: '#f43f5e', forma: 'quadrado', rotulo: 'vermelho', descricao: 'saída perto de 1 (classe 1)' },
          { cor: '#0ea5e9', forma: 'circulo', rotulo: 'círculo', descricao: 'padrão de classe 0' },
          { cor: '#f43f5e', forma: 'triangulo', rotulo: 'triângulo', descricao: 'padrão de classe 1' },
          { cor: '#15803d', forma: 'quadrado', rotulo: 'borda verde', descricao: 'acerto' },
          { cor: '#dc2626', forma: 'quadrado', rotulo: 'borda vermelha', descricao: 'erro' },
        ]}
      />
      <p className="mt-2 text-xs leading-relaxed text-muted">
        As faixas de cor são curvas de mesmo nível de saída (como as curvas de
        altitude de um mapa topográfico) — quanto mais próximas, mais rápido a
        saída muda naquela região.
      </p>
    </div>
  )
}
