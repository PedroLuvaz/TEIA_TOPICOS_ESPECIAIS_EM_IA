/**
 * Diagrama SVG da arquitetura de uma MLP totalmente conectada.
 *
 * Espelha o diagrama da janela de memoria de calculo da GUI desktop:
 * circulos = neuronios (coloridos por camada), linhas = conexoes rotuladas
 * com o peso, quadrado "+1" = bias (individual ou compartilhado pela camada).
 */
import type { Arquitetura } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Legenda } from './ui'

const COR_ENTRADA = '#94a3b8'
const COR_OCULTA = '#f59e0b'
const COR_SAIDA = '#0ea5e9'

interface Props {
  arquitetura: Arquitetura
  /** Ativacoes atuais de cada neuronio (preenche os circulos proporcionalmente). */
  ativacoes?: { ocultas?: number[]; saidas?: number[]; entradas?: number[] }
  mostrarPesos?: boolean
  altura?: number
  className?: string
}

export function DiagramaRede({
  arquitetura,
  ativacoes,
  mostrarPesos = true,
  altura,
  className,
}: Props) {
  const {
    rotulos_entrada: entradas,
    rotulos_ocultos: ocultos,
    rotulos_saida: saidas,
    pesos_oculta,
    bias_oculta,
    pesos_saida,
    bias_saida,
    bias_compartilhado,
  } = arquitetura

  const nMax = Math.max(entradas.length, ocultos.length, saidas.length)
  const espaco = 74
  const raio = 21
  const larguraSvg = 760
  const alturaSvg = altura ?? Math.max(240, nMax * espaco + 130)

  const xEntrada = 92
  const xOculta = larguraSvg / 2
  const xSaida = larguraSvg - 108

  const ys = (n: number) =>
    Array.from(
      { length: n },
      (_, i) => alturaSvg / 2 + 26 - ((n - 1) / 2 - i) * espaco,
    )

  const yEntrada = ys(entradas.length)
  const yOculta = ys(ocultos.length)
  const ySaida = ys(saidas.length)

  const yBias = 38

  return (
    <div className={cn('w-full', className)}>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${larguraSvg} ${alturaSvg}`}
          className="h-auto w-full min-w-[560px]"
          role="img"
          aria-label={`Arquitetura da rede ${entradas.length}-${ocultos.length}-${saidas.length}`}
        >
          {/* Titulos das camadas */}
          <TituloCamada x={xEntrada} texto="ENTRADA" cor="hsl(var(--text-muted))" />
          <TituloCamada x={xOculta} texto="OCULTA (σ)" cor={COR_OCULTA} />
          <TituloCamada x={xSaida} texto="SAÍDA (σ)" cor={COR_SAIDA} />

          {/* Conexoes entrada -> oculta */}
          {yOculta.map((yo, i) =>
            yEntrada.map((ye, j) => (
              <Conexao
                key={`eo-${i}-${j}`}
                x1={xEntrada + raio}
                y1={ye}
                x2={xOculta - raio}
                y2={yo}
                peso={pesos_oculta[i]?.[j]}
                mostrar={mostrarPesos}
                deslocamento={0.26 + 0.1 * (j % 3)}
              />
            )),
          )}

          {/* Conexoes oculta -> saida */}
          {ySaida.map((ysa, i) =>
            yOculta.map((yo, j) => (
              <Conexao
                key={`os-${i}-${j}`}
                x1={xOculta + raio}
                y1={yo}
                x2={xSaida - raio}
                y2={ysa}
                peso={pesos_saida[i]?.[j]}
                mostrar={mostrarPesos}
                deslocamento={0.26 + 0.1 * (j % 3)}
              />
            )),
          )}

          {/* Bias */}
          {bias_compartilhado ? (
            <>
              <BiasCompartilhado
                x={xOculta}
                y={yBias}
                alvos={yOculta}
                raio={raio}
                valor={bias_oculta[0]}
              />
              <BiasCompartilhado
                x={xSaida}
                y={yBias}
                alvos={ySaida}
                raio={raio}
                valor={bias_saida[0]}
              />
            </>
          ) : (
            <>
              {yOculta.map((y, i) => (
                <BiasIndividual key={`bo-${i}`} x={xOculta} y={y + raio + 13} valor={bias_oculta[i]} />
              ))}
              {ySaida.map((y, i) => (
                <BiasIndividual key={`bs-${i}`} x={xSaida} y={y + raio + 13} valor={bias_saida[i]} />
              ))}
            </>
          )}

          {/* Neuronios */}
          {yEntrada.map((y, i) => (
            <Neuronio
              key={`e-${i}`}
              x={xEntrada}
              y={y}
              r={raio}
              rotulo={entradas[i]}
              cor={COR_ENTRADA}
              preenchimento={ativacoes?.entradas?.[i]}
            />
          ))}
          {yOculta.map((y, i) => (
            <Neuronio
              key={`o-${i}`}
              x={xOculta}
              y={y}
              r={raio}
              rotulo={ocultos[i]}
              cor={COR_OCULTA}
              preenchimento={ativacoes?.ocultas?.[i]}
            />
          ))}
          {ySaida.map((y, i) => (
            <Neuronio
              key={`s-${i}`}
              x={xSaida}
              y={y}
              r={raio}
              rotulo={saidas[i]}
              cor={COR_SAIDA}
              preenchimento={ativacoes?.saidas?.[i]}
            />
          ))}
        </svg>
      </div>

      <Legenda
        className="mt-3"
        itens={[
          { cor: COR_ENTRADA, forma: 'circulo', rotulo: 'entrada', descricao: 'recebe o atributo direto' },
          { cor: COR_OCULTA, forma: 'circulo', rotulo: 'oculta', descricao: 'aplica a sigmoide' },
          { cor: COR_SAIDA, forma: 'circulo', rotulo: 'saída', descricao: 'ativação final' },
          { cor: 'hsl(var(--border-strong))', forma: 'linha', rotulo: 'conexão', descricao: 'rotulada com o peso' },
          bias_compartilhado
            ? { cor: COR_OCULTA, forma: 'quadrado', rotulo: '+1', descricao: 'bias único da camada' }
            : { cor: COR_OCULTA, forma: 'quadrado', rotulo: 'b=…', descricao: 'bias de cada neurônio' },
        ]}
      />
    </div>
  )
}

function TituloCamada({ x, texto, cor }: { x: number; texto: string; cor: string }) {
  return (
    <text
      x={x}
      y={16}
      textAnchor="middle"
      className="text-[10px] font-bold tracking-wider"
      fill={cor}
    >
      {texto}
    </text>
  )
}

function Conexao({
  x1,
  y1,
  x2,
  y2,
  peso,
  mostrar,
  deslocamento,
}: {
  x1: number
  y1: number
  x2: number
  y2: number
  peso?: number
  mostrar: boolean
  deslocamento: number
}) {
  const xm = x1 + (x2 - x1) * deslocamento
  const ym = y1 + (y2 - y1) * deslocamento
  const intensidade = peso === undefined ? 1 : Math.min(1, Math.abs(peso) / 0.8)
  return (
    <g>
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="hsl(var(--border-strong))"
        strokeWidth={0.8 + intensidade * 1.1}
        opacity={0.45 + intensidade * 0.4}
      />
      {mostrar && peso !== undefined && (
        <text
          x={xm}
          y={ym}
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-[8.5px] tabular"
          fill="hsl(var(--text-muted))"
          paintOrder="stroke"
          stroke="hsl(var(--surface))"
          strokeWidth={3}
          strokeLinejoin="round"
        >
          {formatarPeso(peso)}
        </text>
      )}
    </g>
  )
}

function Neuronio({
  x,
  y,
  r,
  rotulo,
  cor,
  preenchimento,
}: {
  x: number
  y: number
  r: number
  rotulo: string
  cor: string
  preenchimento?: number
}) {
  const t = preenchimento === undefined ? 0.14 : 0.14 + preenchimento * 0.72
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={cor} opacity={t} />
      <circle cx={x} cy={y} r={r} fill="none" stroke={cor} strokeWidth={2} />
      <text
        x={x}
        y={y}
        textAnchor="middle"
        dominantBaseline="middle"
        className="text-[10px] font-bold"
        fill={cor}
      >
        {rotulo.length > 9 ? `${rotulo.slice(0, 8)}…` : rotulo}
      </text>
      {preenchimento !== undefined && (
        <text
          x={x}
          y={y + r + 11}
          textAnchor="middle"
          className="text-[8px] tabular"
          fill="hsl(var(--text-muted))"
        >
          {preenchimento.toFixed(3)}
        </text>
      )}
    </g>
  )
}

function BiasIndividual({ x, y, valor }: { x: number; y: number; valor?: number }) {
  if (valor === undefined) return null
  return (
    <text
      x={x}
      y={y}
      textAnchor="middle"
      className="text-[8.5px] tabular"
      fill={COR_OCULTA}
    >
      b={formatarPeso(valor)}
    </text>
  )
}

function BiasCompartilhado({
  x,
  y,
  alvos,
  raio,
  valor,
}: {
  x: number
  y: number
  alvos: number[]
  raio: number
  valor?: number
}) {
  return (
    <g>
      {alvos.map((ya, i) => (
        <line
          key={i}
          x1={x}
          y1={y + 11}
          x2={x}
          y2={ya - raio}
          stroke={COR_OCULTA}
          strokeWidth={1}
          strokeDasharray="2 3"
          opacity={0.7}
        />
      ))}
      <rect
        x={x - 12}
        y={y - 11}
        width={24}
        height={22}
        rx={4}
        fill={COR_OCULTA}
        fillOpacity={0.16}
        stroke={COR_OCULTA}
        strokeWidth={1.4}
      />
      <text
        x={x}
        y={y}
        textAnchor="middle"
        dominantBaseline="middle"
        className="text-[9px] font-bold"
        fill={COR_OCULTA}
      >
        +1
      </text>
      {valor !== undefined && (
        <text
          x={x + 20}
          y={y}
          dominantBaseline="middle"
          className="text-[8.5px] tabular"
          fill={COR_OCULTA}
        >
          b={formatarPeso(valor)}
        </text>
      )}
    </g>
  )
}

function formatarPeso(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(1)
  return Number(v.toFixed(3)).toString()
}
