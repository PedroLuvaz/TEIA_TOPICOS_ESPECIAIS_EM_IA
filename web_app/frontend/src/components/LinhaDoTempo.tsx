/**
 * Linha do tempo de treinamento — slider arrastavel de epocas.
 *
 * O backend devolve o historico completo de erro mais snapshots dos pesos ao
 * longo do treino. Aqui o usuario arrasta (ou reproduz) para percorrer as
 * epocas: a curva de erro acumulado destaca o ponto atual e os pesos do
 * snapshot correspondente ficam disponiveis para quem consome o componente
 * recalcular a rede localmente — sem chamada de rede por quadro.
 */
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { SnapshotRede, Trajetoria } from '@/lib/types'
import { cn, num } from '@/lib/utils'
import { Botao, Legenda } from './ui'

const VELOCIDADES = [1, 2, 4] as const

export function usarLinhaDoTempo(trajetoria?: Trajetoria) {
  const [indice, setIndice] = useState(0)

  // Ao trocar de trajetoria (novo treino), volta para o inicio
  useEffect(() => {
    setIndice(0)
  }, [trajetoria])

  const snapshot: SnapshotRede | undefined = trajetoria?.snapshots[indice]
  return { indice, setIndice, snapshot }
}

export function LinhaDoTempo({
  trajetoria,
  indice,
  onMudarIndice,
  altura = 240,
}: {
  trajetoria: Trajetoria
  indice: number
  onMudarIndice: (i: number) => void
  altura?: number
}) {
  const [tocando, setTocando] = useState(false)
  const [velocidade, setVelocidade] = useState<(typeof VELOCIDADES)[number]>(1)

  const total = trajetoria.snapshots.length
  const snapshot = trajetoria.snapshots[indice]
  const ultimo = indice >= total - 1

  /* --- Reproducao automatica ------------------------------------------- */
  // Refs para o loop enxergar sempre os valores atuais sem se reinscrever
  const indiceRef = useRef(indice)
  const aoMudarRef = useRef(onMudarIndice)
  useEffect(() => {
    indiceRef.current = indice
  }, [indice])
  useEffect(() => {
    aoMudarRef.current = onMudarIndice
  }, [onMudarIndice])

  const parar = useCallback(() => setTocando(false), [])

  useEffect(() => {
    if (!tocando) return
    const id = window.setInterval(() => {
      const proximo = indiceRef.current + 1
      if (proximo >= total - 1) {
        aoMudarRef.current(total - 1)
        setTocando(false)
        return
      }
      aoMudarRef.current(proximo)
    }, 90 / velocidade)
    return () => window.clearInterval(id)
  }, [tocando, velocidade, total])

  const reproduzir = () => {
    if (ultimo) onMudarIndice(0)
    setTocando(true)
  }

  /* --- Curva de erro ---------------------------------------------------- */
  // Uma entrada por snapshot mantem o grafico leve mesmo com 50 mil epocas.
  // A epoca 0 fica de fora porque o eixo X usa escala logaritmica.
  const dados = useMemo(
    () =>
      trajetoria.snapshots
        .filter((s) => s.epoca >= 1)
        .map((s) => ({
          x: s.epoca,
          erro: trajetoria.historico[s.epoca - 1],
        }))
        .filter((p) => p.erro !== undefined),
    [trajetoria],
  )

  const erroAtual =
    snapshot.epoca === 0
      ? trajetoria.historico[0]
      : trajetoria.historico[snapshot.epoca - 1]

  const erroInicial = trajetoria.historico[0]
  const reducao =
    erroInicial > 0 ? (1 - erroAtual / erroInicial) * 100 : 0

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------ curva de erro --- */}
      <ResponsiveContainer width="100%" height={altura}>
        <LineChart
          data={dados}
          margin={{ top: 8, right: 16, bottom: 24, left: 4 }}
        >
          <CartesianGrid stroke="hsl(var(--grid-line))" strokeDasharray="3 3" />
          <XAxis
            dataKey="x"
            type="number"
            scale="log"
            domain={[1, trajetoria.epocas]}
            allowDataOverflow
            tick={{ fontSize: 11, fill: 'hsl(var(--text-muted))' }}
            stroke="hsl(var(--border-strong))"
            label={{
              value: 'época (escala logarítmica)',
              position: 'insideBottom',
              offset: -14,
              style: {
                fontSize: 11,
                fill: 'hsl(var(--text-secondary))',
                textAnchor: 'middle',
              },
            }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: 'hsl(var(--text-muted))' }}
            stroke="hsl(var(--border-strong))"
            width={62}
            label={{
              value: 'erro acumulado',
              angle: -90,
              position: 'insideLeft',
              style: {
                fontSize: 11,
                fill: 'hsl(var(--text-secondary))',
                textAnchor: 'middle',
              },
            }}
          />
          <Tooltip
            contentStyle={{
              background: 'hsl(var(--surface))',
              border: '1px solid hsl(var(--border-strong))',
              borderRadius: 10,
              fontSize: 12,
            }}
            labelStyle={{ color: 'hsl(var(--text-muted))', fontSize: 11 }}
            formatter={(v: number) => [num(v, 6), 'erro']}
            labelFormatter={(v) => `época ${v}`}
          />
          <Line
            type="monotone"
            dataKey="erro"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <ReferenceLine
            x={Math.max(1, snapshot.epoca)}
            stroke="hsl(var(--text-primary))"
            strokeDasharray="4 3"
            strokeWidth={1.5}
          />
          <ReferenceDot
            x={Math.max(1, snapshot.epoca)}
            y={erroAtual}
            r={5}
            fill="#f59e0b"
            stroke="hsl(var(--surface))"
            strokeWidth={2}
            isFront
          />
        </LineChart>
      </ResponsiveContainer>

      <Legenda
        itens={[
          { cor: '#f59e0b', forma: 'linha', rotulo: 'erro acumulado por época' },
          {
            cor: 'hsl(var(--text-primary))',
            forma: 'linha-tracejada',
            rotulo: 'época atual do slider',
          },
        ]}
      />

      {/* --------------------------------------------------- controles --- */}
      <div className="rounded-lg border border-subtle bg-sunken p-4">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <span className="kicker text-muted">arraste para percorrer o treino</span>
          <div className="flex items-baseline gap-4 text-sm">
            <span>
              <span className="text-muted">época </span>
              <span className="tabular font-semibold text-primary">
                {snapshot.epoca}
              </span>
              <span className="text-muted"> / {trajetoria.epocas}</span>
            </span>
            <span>
              <span className="text-muted">erro </span>
              <span className="tabular font-semibold text-accent-600 dark:text-accent-400">
                {num(erroAtual, 6)}
              </span>
            </span>
            {snapshot.epoca > 0 && (
              <span className="text-xs text-muted">
                ({reducao >= 0 ? '−' : '+'}
                {Math.abs(reducao).toFixed(1)}% vs. início)
              </span>
            )}
          </div>
        </div>

        <input
          type="range"
          min={0}
          max={total - 1}
          step={1}
          value={indice}
          onChange={(e) => {
            parar()
            onMudarIndice(Number(e.target.value))
          }}
          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-zinc-500/25 accent-accent-500"
          aria-label="Época do treinamento"
        />

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Botao
            tamanho="sm"
            variante="primario"
            onClick={tocando ? parar : reproduzir}
          >
            {tocando ? <Pause size={13} /> : <Play size={13} />}
            {tocando ? 'Pausar' : ultimo ? 'Reproduzir de novo' : 'Reproduzir'}
          </Botao>
          <Botao
            tamanho="sm"
            onClick={() => {
              parar()
              onMudarIndice(Math.max(0, indice - 1))
            }}
            disabled={indice === 0}
          >
            <SkipBack size={13} />
          </Botao>
          <Botao
            tamanho="sm"
            onClick={() => {
              parar()
              onMudarIndice(Math.min(total - 1, indice + 1))
            }}
            disabled={ultimo}
          >
            <SkipForward size={13} />
          </Botao>
          <Botao
            tamanho="sm"
            variante="fantasma"
            onClick={() => {
              parar()
              onMudarIndice(0)
            }}
          >
            <RotateCcw size={13} />
            Início
          </Botao>

          <span className="ml-auto flex items-center gap-1">
            <span className="mr-1 text-xs text-muted">velocidade</span>
            {VELOCIDADES.map((v) => (
              <button
                key={v}
                onClick={() => setVelocidade(v)}
                className={cn(
                  'rounded px-2 py-1 text-xs font-medium transition-colors',
                  velocidade === v
                    ? 'bg-accent-500/15 text-accent-700 dark:text-accent-400'
                    : 'text-muted hover:text-secondary',
                )}
              >
                {v}×
              </button>
            ))}
          </span>
        </div>
      </div>
    </div>
  )
}
