/** Matriz de confusao e tabelas de metricas de qualidade. */
import type { Relatorio } from '@/lib/types'
import { cap, cn, corDaClasse, interpretarKappa, num, pct, sci } from '@/lib/utils'
import { Badge, Legenda } from './ui'

/* ------------------------------------------------------- Matriz de confusao */
export function MatrizConfusao({
  relatorio,
  classes,
  editavel,
  onEditar,
}: {
  relatorio: Relatorio
  classes: string[]
  editavel?: boolean
  onEditar?: (predito: string, real: string, valor: number) => void
}) {
  const { matriz } = relatorio
  const valores = classes.flatMap((p) => classes.map((r) => matriz[p]?.[r] ?? 0))
  const maximo = Math.max(...valores, 1)

  const totalLinha = (p: string) =>
    classes.reduce((s, r) => s + (matriz[p]?.[r] ?? 0), 0)
  const totalColuna = (r: string) =>
    classes.reduce((s, p) => s + (matriz[p]?.[r] ?? 0), 0)
  const total = valores.reduce((s, v) => s + v, 0)

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-1 text-sm">
          <thead>
            <tr>
              <th className="rounded-md bg-sunken px-3 py-2 text-left text-[11px] font-semibold text-muted">
                Predito \ Real
              </th>
              {classes.map((c) => (
                <th
                  key={c}
                  className="rounded-md px-3 py-2 text-center text-[11px] font-semibold text-white"
                  style={{ backgroundColor: corDaClasse(c) }}
                >
                  {cap(c)}
                </th>
              ))}
              <th className="rounded-md bg-sunken px-3 py-2 text-center text-[11px] font-semibold text-muted">
                Total
              </th>
            </tr>
          </thead>
          <tbody>
            {classes.map((p) => (
              <tr key={p}>
                <th
                  className="rounded-md px-3 py-2 text-left text-[11px] font-semibold text-white"
                  style={{ backgroundColor: corDaClasse(p) }}
                >
                  {cap(p)}
                </th>
                {classes.map((r) => {
                  const v = matriz[p]?.[r] ?? 0
                  const diagonal = p === r
                  const t = v / maximo
                  return (
                    <td key={r} className="p-0">
                      <div
                        className={cn(
                          'flex items-center justify-center rounded-md tabular font-semibold',
                          'transition-colors',
                          diagonal
                            ? 'text-emerald-700 dark:text-emerald-300'
                            : v > 0
                              ? 'text-rose-700 dark:text-rose-300'
                              : 'text-muted',
                        )}
                        style={{
                          backgroundColor: diagonal
                            ? `rgba(16,185,129,${0.08 + t * 0.28})`
                            : v > 0
                              ? `rgba(244,63,94,${0.08 + t * 0.28})`
                              : 'transparent',
                          minHeight: '2.6rem',
                        }}
                      >
                        {editavel && onEditar ? (
                          <input
                            type="number"
                            min={0}
                            value={v}
                            onChange={(e) =>
                              onEditar(p, r, Math.max(0, Number(e.target.value) || 0))
                            }
                            className="w-14 bg-transparent text-center tabular font-semibold outline-none"
                            aria-label={`Predito ${p}, real ${r}`}
                          />
                        ) : (
                          v
                        )}
                      </div>
                    </td>
                  )
                })}
                <td className="rounded-md bg-sunken text-center tabular text-secondary">
                  {totalLinha(p)}
                </td>
              </tr>
            ))}
            <tr>
              <th className="rounded-md bg-sunken px-3 py-2 text-left text-[11px] font-semibold text-muted">
                Total
              </th>
              {classes.map((r) => (
                <td
                  key={r}
                  className="rounded-md bg-sunken text-center tabular text-secondary"
                >
                  {totalColuna(r)}
                </td>
              ))}
              <td className="rounded-md bg-sunken text-center tabular font-semibold text-primary">
                {total}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <Legenda
        itens={[
          { cor: 'rgba(16,185,129,0.35)', rotulo: 'diagonal = acertos' },
          { cor: 'rgba(244,63,94,0.35)', rotulo: 'fora da diagonal = erros' },
          {
            cor: 'hsl(var(--text-muted))',
            rotulo: 'intensidade da cor = quantidade de amostras',
          },
        ]}
      />
    </div>
  )
}

/* -------------------------------------------------------- Metricas globais */
export function ResumoGlobal({ relatorio }: { relatorio: Relatorio }) {
  const k = interpretarKappa(relatorio.kappa)
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <ItemMetrica
        rotulo="Acerto Global (Ag)"
        valor={pct(relatorio.acerto_global)}
        formula="Ag = Σ diagonal / N"
      />
      <ItemMetrica
        rotulo="Kappa (κ)"
        valor={num(relatorio.kappa, 4)}
        badge={<Badge tom={k.tom === 'bom' ? 'bom' : k.tom === 'medio' ? 'medio' : 'ruim'}>{k.rotulo}</Badge>}
        formula="κ = (Ag − Ac) / (1 − Ac)"
      />
      <ItemMetrica
        rotulo="Tau (τ)"
        valor={num(relatorio.tau, 4)}
        formula="τ assume classes equiprováveis"
      />
      <ItemMetrica
        rotulo="Var(κ)"
        valor={sci(relatorio.variancia_kappa)}
        formula="usada no teste Z"
      />
    </div>
  )
}

function ItemMetrica({
  rotulo,
  valor,
  badge,
  formula,
}: {
  rotulo: string
  valor: string
  badge?: React.ReactNode
  formula?: string
}) {
  return (
    <div className="rounded-lg border border-subtle bg-sunken px-4 py-3">
      <div className="kicker mb-1 !text-[10px] text-muted">{rotulo}</div>
      <div className="flex items-center gap-2">
        <span className="tabular text-xl font-semibold text-primary">{valor}</span>
        {badge}
      </div>
      {formula && (
        <div className="mt-1 font-mono text-[10px] text-muted">{formula}</div>
      )}
    </div>
  )
}

/* ------------------------------------------------------- Metricas por classe */
const COLUNAS = [
  { chave: 'acuracia_produtor', rotulo: 'Ac. Produtor', ajuda: 'recall / sensibilidade', tipo: 'pct' },
  { chave: 'acuracia_usuario', rotulo: 'Ac. Usuário', ajuda: 'precisão', tipo: 'pct' },
  { chave: 'especificidade', rotulo: 'Especificidade', ajuda: 'VN / (VN + FP)', tipo: 'pct' },
  { chave: 'f1', rotulo: 'F1', ajuda: 'média harmônica (β=1)', tipo: 'num' },
  { chave: 'f2', rotulo: 'F2', ajuda: 'mais peso à revocação', tipo: 'num' },
  { chave: 'mcc', rotulo: 'MCC', ajuda: 'coef. de Matthews', tipo: 'num' },
] as const

export function TabelaPorClasse({
  relatorio,
  classes,
}: {
  relatorio: Relatorio
  classes: string[]
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-subtle">
            <th className="py-2 pr-3 text-left text-[11px] font-semibold text-muted">
              Classe
            </th>
            {COLUNAS.map((c) => (
              <th
                key={c.chave}
                className="px-2 py-2 text-right text-[11px] font-semibold text-muted"
                title={c.ajuda}
              >
                {c.rotulo}
              </th>
            ))}
            <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
              VP / FP / FN
            </th>
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => {
            const m = relatorio.por_classe[c]
            if (!m) return null
            return (
              <tr key={c} className="border-b border-subtle/60 last:border-0">
                <td className="py-2.5 pr-3">
                  <span className="inline-flex items-center gap-2">
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: corDaClasse(c) }}
                    />
                    <span className="font-medium text-primary">{cap(c)}</span>
                  </span>
                </td>
                {COLUNAS.map((col) => {
                  const v = m[col.chave]
                  return (
                    <td
                      key={col.chave}
                      className="px-2 py-2.5 text-right tabular text-secondary"
                    >
                      {col.tipo === 'pct' ? pct(v) : num(v, 4)}
                    </td>
                  )
                })}
                <td className="px-2 py-2.5 text-right tabular text-xs text-muted">
                  {m.vp} / {m.fp} / {m.fn}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* --------------------------------------------------------- Tabela de testes */
export function TabelaTestesZ({
  comparacoes,
}: {
  comparacoes: {
    nome_a: string
    nome_b: string
    z: number
    p: number
    significativo: boolean
  }[]
}) {
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-subtle">
              <th className="py-2 pr-3 text-left text-[11px] font-semibold text-muted">
                Par de classificadores
              </th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                Z
              </th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                p-valor
              </th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                Veredito (α = 5%)
              </th>
            </tr>
          </thead>
          <tbody>
            {comparacoes.map((c, i) => (
              <tr key={i} className="border-b border-subtle/60 last:border-0">
                <td className="py-2.5 pr-3 text-secondary">
                  {c.nome_a} <span className="text-muted">×</span> {c.nome_b}
                </td>
                <td className="px-2 py-2.5 text-right tabular text-primary">
                  {num(c.z, 4)}
                </td>
                <td className="px-2 py-2.5 text-right tabular text-secondary">
                  {num(c.p, 6)}
                </td>
                <td className="px-2 py-2.5 text-right">
                  <Badge tom={c.significativo ? 'ruim' : 'bom'}>
                    {c.significativo ? 'diferença significativa' : 'sem diferença'}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted">
        Z = (κ₁ − κ₂) / √(Var(κ₁) + Var(κ₂)). Rejeita-se a hipótese de
        equivalência quando |Z| &gt; 1,96 (p &lt; 0,05).
      </p>
    </div>
  )
}
