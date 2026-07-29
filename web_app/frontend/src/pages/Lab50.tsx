/**
 * Lab 5.0 — XOR com MLP (slides 36-37 da Aula PR_711).
 *
 * Exemplo didatico do slide 37 + exercicio do XOR, com linha do tempo de
 * treinamento arrastavel, superficie de saida da rede e memoria de calculo.
 */
import { useQuery } from '@tanstack/react-query'
import { CheckCircle2, FileText, Sparkles, XCircle } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { DiagramaRede } from '@/components/DiagramaRede'
import { BlocoFormula } from '@/components/Formula'
import { LinhaDoTempo, usarLinhaDoTempo } from '@/components/LinhaDoTempo'
import { MemoriaCalculo } from '@/components/MemoriaCalculo'
import {
  Botao,
  Card,
  Carregando,
  ErroBox,
  Legenda,
  Metrica,
  Nota,
  Slider,
} from '@/components/ui'
import { api } from '@/lib/api'
import type { SnapshotRede } from '@/lib/types'
import type { PesosRede } from '@/lib/utils'
import {
  cn,
  escalaDivergente,
  forward,
  num,
  superficieDeSaida,
} from '@/lib/utils'

const PADROES_XOR = [
  { entrada: [0, 0], alvo: 0 },
  { entrada: [0, 1], alvo: 1 },
  { entrada: [1, 0], alvo: 1 },
  { entrada: [1, 1], alvo: 0 },
]

export function PaginaLab50() {
  const [exercicio, setExercicio] = useState<string | null>(null)
  // Os sliders mexem no rascunho; so o botao "Treinar" o promove a parametro
  // efetivo — assim arrastar nao dispara um treino a cada quadro.
  const [rascunho, setRascunho] = useState({ epocas: 5000, taxa: 0.5 })
  const [parametros, setParametros] = useState(rascunho)

  const memoria = useQuery({
    queryKey: ['lab5', 'memoria', exercicio],
    queryFn: () => api.lab5.memoria(exercicio!),
    enabled: !!exercicio,
  })

  const treino = useQuery({
    queryKey: ['lab5', 'trajetoria', 'xor', parametros],
    queryFn: () =>
      api.lab5.trajetoria('xor', { ...parametros, n_snapshots: 80 }),
  })

  const trajetoria = treino.data
  const { indice, setIndice, snapshot } = usarLinhaDoTempo(trajetoria)

  const epocas = rascunho.epocas
  const taxa = rascunho.taxa
  const setEpocas = (v: number) => setRascunho((r) => ({ ...r, epocas: v }))
  const setTaxa = (v: number) => setRascunho((r) => ({ ...r, taxa: v }))
  const treinar = (ajuste?: Partial<typeof rascunho>) => {
    const novo = { ...rascunho, ...ajuste }
    setRascunho(novo)
    setParametros(novo)
  }
  const pendente =
    treino.isFetching ||
    rascunho.epocas !== parametros.epocas ||
    rascunho.taxa !== parametros.taxa

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
            substituição numérica de cada etapa.
          </p>
        </Card>

        <Card titulo="treinar a rede">
          <div className="space-y-4">
            <Slider
              rotulo="Épocas a treinar"
              valor={epocas}
              onChange={setEpocas}
              min={1}
              max={20000}
              passo={100}
              formatar={(v) => v.toLocaleString('pt-BR')}
            />
            <Slider
              rotulo="Taxa de aprendizagem (η)"
              valor={taxa}
              onChange={setTaxa}
              min={0.05}
              max={2}
              passo={0.05}
              formatar={(v) => v.toFixed(2)}
            />
            <Botao
              variante="primario"
              className="w-full"
              onClick={() => treinar()}
              disabled={treino.isFetching}
            >
              <Sparkles size={15} />
              {treino.isFetching
                ? 'Treinando…'
                : pendente
                  ? 'Aplicar e treinar'
                  : 'Treinar de novo'}
            </Botao>
            <div className="grid grid-cols-2 gap-2">
              <Botao
                tamanho="sm"
                onClick={() => treinar({ epocas: 1 })}
                disabled={treino.isFetching}
              >
                Só 1 época
              </Botao>
              <Botao
                tamanho="sm"
                onClick={() => treinar({ epocas: 5000 })}
                disabled={treino.isFetching}
              >
                5000 épocas
              </Botao>
            </div>
          </div>
          <Nota tom="atencao" className="mt-4">
            Treine bastante e depois <strong>arraste a linha do tempo</strong>:
            com estes pesos o erro fica quase parado até a época ~500 e só
            converge de fato entre 2000 e 5000 — algo que a Regra Delta linear
            (Lab 2) <strong>nunca</strong> consegue fazer.
          </Nota>
        </Card>
      </div>

      {/* ------------------------------------------------------- resultados */}
      <div className="space-y-6">
        {treino.isPending && (
          <Card>
            <Carregando texto="Treinando a rede…" />
          </Card>
        )}
        {treino.error ? <ErroBox erro={treino.error} /> : null}

        {trajetoria && snapshot && (
          <>
            <ResumoEpoca snapshot={snapshot} />

            <Card titulo="linha do tempo do treinamento">
              <LinhaDoTempo
                trajetoria={trajetoria}
                indice={indice}
                onMudarIndice={setIndice}
              />
            </Card>

            <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)]">
              <Card titulo="saída da rede sobre o plano x₁ × x₂">
                <SuperficieXor snapshot={snapshot} />
              </Card>

              <Card titulo="tabela-verdade nesta época">
                <TabelaXor snapshot={snapshot} />
              </Card>
            </div>

            <Card titulo={`arquitetura na época ${snapshot.epoca}`}>
              <DiagramaRede
                arquitetura={{
                  ...trajetoria.arquitetura,
                  pesos_oculta: snapshot.pesos_oculta,
                  bias_oculta: snapshot.bias_oculta,
                  pesos_saida: snapshot.pesos_saida,
                  bias_saida: snapshot.bias_saida,
                }}
              />
              <BlocoFormula
                className="mt-4"
                titulo="a solução do XOR"
                latex={String.raw`\text{XOR}(x_1,x_2) = \sigma\!\left(w_7 h_1 + w_8 h_2 + b\right),\quad
                  h_i = \sigma\!\left(w_{i1}x_1 + w_{i2}x_2 + b_i\right)`}
                explicacao="A camada oculta transforma o espaço de entrada — no espaço de h₁ × h₂ o problema passa a ser linearmente separável. Arraste a linha do tempo e veja os pesos mudarem."
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

/* ------------------------------------------------------------------ resumo */
function ResumoEpoca({ snapshot }: { snapshot: SnapshotRede }) {
  const acertos = PADROES_XOR.filter((p, i) => {
    const saida = snapshot.saidas[i]?.[0] ?? 0.5
    return (saida >= 0.5 ? 1 : 0) === p.alvo
  }).length

  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <Metrica rotulo="Época" valor={snapshot.epoca} />
      <Metrica
        rotulo="Erro nesta época"
        valor={snapshot.erro !== null ? num(snapshot.erro, 6) : '—'}
      />
      <Metrica
        rotulo="Padrões corretos"
        valor={`${acertos} / 4`}
        destaque={acertos === 4 ? 'bom' : acertos >= 2 ? 'medio' : 'ruim'}
      />
    </div>
  )
}

/* ---------------------------------------------------------- tabela-verdade */
function TabelaXor({ snapshot }: { snapshot: SnapshotRede }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-subtle">
          <th className="py-2 text-left text-[11px] font-semibold text-muted">
            (x₁, x₂)
          </th>
          <th className="py-2 text-right text-[11px] font-semibold text-muted">
            Alvo
          </th>
          <th className="py-2 text-right text-[11px] font-semibold text-muted">
            Saída
          </th>
          <th className="py-2 text-right text-[11px] font-semibold text-muted">
            Status
          </th>
        </tr>
      </thead>
      <tbody>
        {PADROES_XOR.map((p, i) => {
          const saida = snapshot.saidas[i]?.[0] ?? 0.5
          const previsto = saida >= 0.5 ? 1 : 0
          const correto = previsto === p.alvo
          return (
            <tr key={i} className="border-b border-subtle/60 last:border-0">
              <td className="py-2.5 font-mono text-secondary">
                ({p.entrada.join(', ')})
              </td>
              <td className="py-2.5 text-right tabular text-secondary">
                {p.alvo}
              </td>
              <td className="py-2.5 text-right tabular font-semibold text-primary">
                {num(saida, 4)}
              </td>
              <td className="py-2.5 text-right">
                <span
                  className={cn(
                    'inline-flex items-center gap-1 text-xs font-medium',
                    correto
                      ? 'text-emerald-600 dark:text-emerald-400'
                      : 'text-rose-600 dark:text-rose-400',
                  )}
                >
                  {correto ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                  {correto ? 'correto' : 'incorreto'}
                </span>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/* --------------------------------------- superficie de saida (canvas local) */
function SuperficieXor({ snapshot }: { snapshot: SnapshotRede }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tamanho = 380
  const margem = 42
  const area = tamanho - margem - 12
  const lo = -0.35
  const hi = 1.35

  const pesos: PesosRede = useMemo(
    () => ({
      pesos_oculta: snapshot.pesos_oculta,
      bias_oculta: snapshot.bias_oculta,
      pesos_saida: snapshot.pesos_saida,
      bias_saida: snapshot.bias_saida,
    }),
    [snapshot],
  )

  // A superficie e recalculada no cliente a cada quadro do slider — como e
  // so uma rede 2-2-1 numa grade 70x70, o custo e desprezivel.
  const superficie = useMemo(
    () => superficieDeSaida(pesos, 70, lo, hi),
    [pesos],
  )

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

    const n = superficie.length
    const aux = document.createElement('canvas')
    aux.width = n
    aux.height = n
    const auxCtx = aux.getContext('2d')
    if (!auxCtx) return

    const img = auxCtx.createImageData(n, n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // A grade cresce de baixo para cima; a imagem, de cima para baixo
        const cor = escalaDivergente(superficie[n - 1 - i][j])
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
    ctx.drawImage(aux, margem, 12, area, area)
  }, [superficie, area])

  const px = (x: number) => margem + ((x - lo) / (hi - lo)) * area
  const py = (y: number) => 12 + area - ((y - lo) / (hi - lo)) * area

  return (
    <div>
      <div className="flex flex-wrap items-start gap-4">
        <div
          className="relative shrink-0"
          style={{ width: tamanho, height: tamanho }}
        >
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
            {PADROES_XOR.map((p, i) => {
              const [x1, x2] = p.entrada
              const saida = forward([x1, x2], pesos).saidas[0]
              const correto = (saida >= 0.5 ? 1 : 0) === p.alvo
              const cor = p.alvo === 1 ? '#f43f5e' : '#0ea5e9'
              const borda = correto ? '#15803d' : '#dc2626'
              return (
                <g key={i}>
                  {p.alvo === 1 ? (
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
                    {num(saida, 3)}
                  </text>
                </g>
              )
            })}

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

        <div className="flex items-center gap-2 pt-3">
          <div
            className="h-[240px] w-4 rounded border border-subtle"
            style={{
              background: `linear-gradient(to top, ${escalaDivergente(0)}, ${escalaDivergente(0.5)}, ${escalaDivergente(1)})`,
            }}
            aria-hidden
          />
          <div className="flex h-[240px] flex-col justify-between text-[10px] tabular text-muted">
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
