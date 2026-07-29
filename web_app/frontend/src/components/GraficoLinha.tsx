/** Grafico de linha tematizado (curvas de convergencia, erro por epoca). */
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { num } from '@/lib/utils'

export interface SerieLinha {
  chave: string
  rotulo: string
  cor: string
  tracejada?: boolean
}

export function GraficoLinha({
  dados,
  series,
  rotuloX,
  rotuloY,
  altura = 260,
  escalaLogX,
  referencia,
}: {
  dados: Record<string, number>[]
  series: SerieLinha[]
  rotuloX: string
  rotuloY: string
  altura?: number
  escalaLogX?: boolean
  referencia?: { y: number; rotulo: string; cor?: string }
}) {
  return (
    <ResponsiveContainer width="100%" height={altura}>
      <LineChart data={dados} margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
        <CartesianGrid stroke="hsl(var(--grid-line))" strokeDasharray="3 3" />
        <XAxis
          dataKey="x"
          scale={escalaLogX ? 'log' : 'auto'}
          domain={escalaLogX ? ['dataMin', 'dataMax'] : undefined}
          type="number"
          tick={{ fontSize: 11, fill: 'hsl(var(--text-muted))' }}
          stroke="hsl(var(--border-strong))"
          label={{
            value: rotuloX,
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
          width={56}
          label={{
            value: rotuloY,
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
            boxShadow: '0 8px 24px rgba(0,0,0,.12)',
          }}
          labelStyle={{ color: 'hsl(var(--text-muted))', fontSize: 11 }}
          formatter={(v: number, nome: string) => [num(v, 6), nome]}
          labelFormatter={(v) => `${rotuloX}: ${v}`}
        />
        {series.length > 1 && (
          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
            iconType="line"
            iconSize={14}
          />
        )}
        {referencia && (
          <ReferenceLine
            y={referencia.y}
            stroke={referencia.cor ?? 'hsl(var(--text-muted))'}
            strokeDasharray="5 4"
            label={{
              value: referencia.rotulo,
              position: 'insideTopRight',
              style: { fontSize: 10, fill: 'hsl(var(--text-muted))' },
            }}
          />
        )}
        {series.map((s) => (
          <Line
            key={s.chave}
            type="monotone"
            dataKey={s.chave}
            name={s.rotulo}
            stroke={s.cor}
            strokeWidth={2}
            strokeDasharray={s.tracejada ? '5 4' : undefined}
            dot={false}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
