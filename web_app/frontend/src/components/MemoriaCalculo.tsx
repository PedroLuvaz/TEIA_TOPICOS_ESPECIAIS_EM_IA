/**
 * Memoria de calculo do Lab 5 — equivalente web da janela LaTeX da GUI.
 *
 * Renderiza o traco devolvido por /api/lab5/memoria/{id}: diagrama da rede,
 * formula geral de cada etapa e a substituicao numerica correspondente.
 */
import { motion } from 'motion/react'
import { X } from 'lucide-react'
import { useEffect } from 'react'
import type { Traco, TracoEpoca, TracoPassoUnico } from '@/lib/types'
import { num } from '@/lib/utils'
import { DiagramaRede } from './DiagramaRede'
import { BlocoFormula, Formula } from './Formula'
import { Badge, Botao, Carregando, ErroBox, Nota } from './ui'

export function MemoriaCalculo({
  traco,
  carregando,
  erro,
  onFechar,
}: {
  traco?: Traco
  carregando?: boolean
  erro?: unknown
  onFechar: () => void
}) {
  useEffect(() => {
    const aoTeclar = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onFechar()
    }
    document.addEventListener('keydown', aoTeclar)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', aoTeclar)
      document.body.style.overflow = ''
    }
  }, [onFechar])

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/55 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onFechar} aria-hidden />
      <motion.div
        initial={{ opacity: 0, y: 16, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full max-w-5xl rounded-2xl border border-subtle bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Memória de cálculo"
      >
        <header className="sticky top-0 z-10 flex items-start justify-between gap-4 rounded-t-2xl border-b border-subtle bg-surface/95 px-6 py-4 backdrop-blur">
          <div>
            <p className="kicker">Memória de cálculo</p>
            <h2 className="mt-0.5 text-lg font-semibold text-primary">
              {traco?.config.titulo ?? 'Carregando…'}
            </h2>
            {traco && (
              <p className="mt-0.5 text-sm text-muted">{traco.config.subtitulo}</p>
            )}
          </div>
          <Botao variante="fantasma" tamanho="sm" onClick={onFechar}>
            <X size={16} />
            Fechar
          </Botao>
        </header>

        <div className="space-y-7 px-6 py-6">
          {carregando && <Carregando texto="Montando a memória de cálculo…" />}
          {erro ? <ErroBox erro={erro} /> : null}
          {traco?.tipo === 'passo-unico' && <PassoUnico traco={traco} />}
          {traco?.tipo === 'epoca' && <Epoca traco={traco} />}
        </div>
      </motion.div>
    </div>
  )
}

/* ------------------------------------------------------------------ secoes */
function Secao({
  numero,
  titulo,
  children,
}: {
  numero: number
  titulo: string
  children: React.ReactNode
}) {
  return (
    <section className="space-y-3">
      <h3 className="flex items-center gap-2.5 text-sm font-semibold text-primary">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent-500/15 text-xs font-bold text-accent-700 dark:text-accent-400">
          {numero}
        </span>
        {titulo}
      </h3>
      <div className="space-y-3 pl-8.5">{children}</div>
    </section>
  )
}

function Conta({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md bg-sunken px-3 py-2 font-mono text-[12.5px] leading-relaxed text-secondary">
      {children}
    </div>
  )
}

/* ------------------------------------------------------- traco passo unico */
function PassoUnico({ traco }: { traco: TracoPassoUnico }) {
  const eta = traco.config.taxa
  return (
    <>
      <Secao numero={1} titulo="Arquitetura da rede">
        <DiagramaRede arquitetura={traco.arquitetura} />
        {traco.config.nota && <Nota tom="info">{traco.config.nota}</Nota>}
      </Secao>

      <Secao numero={2} titulo="Modelo do neurônio">
        <BlocoFormula
          latex={String.raw`z_i(l) \;=\; \sum_j w_{ij}(l)\, a_j(l-1) \;+\; b_i(l)
            \qquad a_i(l) \;=\; \sigma\!\left(z_i(l)\right)
            \qquad \sigma(z) = \frac{1}{1+e^{-z}}`}
          explicacao="Cada neurônio soma as entradas ponderadas pelos pesos, adiciona o bias e aplica a sigmoide."
        />
      </Secao>

      <Secao numero={3} titulo="Alimentação adiante · camada oculta">
        {traco.forward_oculta.map((n) => (
          <div key={n.nome} className="space-y-1.5">
            <p className="kicker text-muted">neurônio {n.nome}</p>
            <Conta>
              net<sub>{n.nome}</sub> ={' '}
              {n.termos
                .map((t) => `${num(t.entrada, 2)}·${num(t.peso, 4)}`)
                .join('  +  ')}{' '}
              + {num(n.bias, 4)} = <strong>{num(n.net, 6)}</strong>
            </Conta>
            <Conta>
              out<sub>{n.nome}</sub> = σ({num(n.net, 6)}) ={' '}
              <strong>{num(n.out, 6)}</strong>
            </Conta>
          </div>
        ))}
      </Secao>

      <Secao numero={4} titulo="Alimentação adiante · camada de saída">
        {traco.forward_saida.map((n) => (
          <div key={n.nome} className="space-y-1.5">
            <p className="kicker text-muted">neurônio {n.nome}</p>
            <Conta>
              net = {n.termos
                .map((t) => `${num(t.entrada, 6)}·${num(t.peso, 4)}`)
                .join('  +  ')}{' '}
              + {num(n.bias, 4)} = <strong>{num(n.net, 6)}</strong>
            </Conta>
            <Conta>
              out = σ({num(n.net, 6)}) = <strong>{num(n.out, 6)}</strong>
            </Conta>
          </div>
        ))}
      </Secao>

      <Secao numero={5} titulo="Erro total">
        <BlocoFormula latex={String.raw`E \;=\; \frac{1}{2}\sum_i (t_i - z_i)^2`} />
        {traco.erro.por_saida.map((e) => (
          <Conta key={e.nome}>
            E<sub>{e.nome}</sub> = ½·({num(e.alvo, 2)} − {num(e.saida, 6)})² ={' '}
            <strong>{num(e.erro, 6)}</strong>
          </Conta>
        ))}
        <div className="rounded-md bg-accent-500/10 px-3 py-2 font-mono text-sm font-semibold text-accent-700 dark:text-accent-400">
          E total = {num(traco.erro.total, 6)}
        </div>
      </Secao>

      <Secao numero={6} titulo="Retropropagação · deltas da camada de saída">
        <BlocoFormula
          latex={String.raw`\delta_o \;=\; (z_o - t_o)\; z_o\,(1 - z_o)`}
          explicacao="Combina o erro bruto com a derivada da sigmoide — que trava o ajuste quando o neurônio já está saturado."
        />
        {traco.deltas_saida.map((d) => (
          <Conta key={d.nome}>
            δ<sub>{d.nome}</sub> = ({num(d.saida, 6)} − {num(d.alvo, 2)})·
            {num(d.saida, 6)}·(1 − {num(d.saida, 6)}) ={' '}
            <strong>{num(d.delta, 6)}</strong>
          </Conta>
        ))}
      </Secao>

      <Secao numero={7} titulo="Retropropagação · deltas da camada oculta">
        <BlocoFormula
          latex={String.raw`\delta_h \;=\; \left(\sum_o \delta_o\, w_{ho}\right)\;
            \text{out}_h\,(1 - \text{out}_h)`}
          explicacao="A soma percorre TODAS as conexões de saída do neurônio oculto — cada uma contribui com seu próprio delta."
        />
        {traco.deltas_oculta.map((d) => (
          <div key={d.nome} className="space-y-1.5">
            <p className="kicker text-muted">neurônio {d.nome}</p>
            <Conta>
              Σ ={' '}
              {d.contribuicoes
                .map((c) => `${num(c.delta, 6)}·${num(c.peso, 4)}`)
                .join('  +  ')}{' '}
              = {num(d.contribuicoes.reduce((s, c) => s + c.produto, 0), 6)}
            </Conta>
            <Conta>
              δ<sub>{d.nome}</sub> ={' '}
              {num(d.contribuicoes.reduce((s, c) => s + c.produto, 0), 6)}·
              {num(d.out, 6)}·(1 − {num(d.out, 6)}) ={' '}
              <strong>{num(d.delta, 6)}</strong>
            </Conta>
          </div>
        ))}
      </Secao>

      <Secao numero={8} titulo={`Atualização dos pesos (η = ${eta})`}>
        <BlocoFormula
          latex={String.raw`w_{\text{novo}} = w - \eta\,\delta\,\text{entrada}
            \qquad b_{\text{novo}} = b - \eta\,\delta`}
        />
        {traco.config.bias_compartilhado && (
          <Nota tom="atencao" titulo="Bias compartilhado por camada">
            Este exemplo usa um único bias por camada. O gradiente do bias soma
            os deltas de <em>todos</em> os neurônios da camada, em vez de usar
            apenas o delta de um.
          </Nota>
        )}

        <p className="kicker text-muted">camada de saída</p>
        {traco.atualizacao.saida.map((a, i) => (
          <Conta key={i}>
            w({a.origem}→{a.destino}) = {num(a.antes, 5)} − {eta}·
            {num(a.delta, 6)}·{num(a.entrada, 6)} ={' '}
            <strong>{num(a.depois, 6)}</strong>
          </Conta>
        ))}

        <p className="kicker pt-1 text-muted">camada oculta</p>
        {traco.atualizacao.oculta.map((a, i) => (
          <Conta key={i}>
            w({a.origem}→{a.destino}) = {num(a.antes, 5)} − {eta}·
            {num(a.delta, 6)}·{num(a.entrada, 4)} ={' '}
            <strong>{num(a.depois, 6)}</strong>
          </Conta>
        ))}

        <p className="kicker pt-1 text-muted">bias</p>
        {traco.atualizacao.bias.map((b, i) => (
          <Conta key={i}>
            {b.nome} = {num(b.antes, 5)} − {eta}·
            {b.deltas.length > 1
              ? `(${b.deltas.map((d) => num(d, 6)).join(' + ')})`
              : num(b.soma_deltas, 6)}{' '}
            = <strong>{num(b.depois, 6)}</strong>
          </Conta>
        ))}
      </Secao>

      <Secao numero={9} titulo="Nova predição · após 1 atualização">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-subtle">
                <th className="py-2 text-left text-[11px] font-semibold text-muted">
                  Saída
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Alvo
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Antes
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Depois
                </th>
              </tr>
            </thead>
            <tbody>
              {traco.nova_predicao.saidas.map((s) => (
                <tr key={s.nome} className="border-b border-subtle/60 last:border-0">
                  <td className="py-2 font-medium text-primary">{s.nome}</td>
                  <td className="py-2 text-right tabular text-secondary">
                    {num(s.alvo, 2)}
                  </td>
                  <td className="py-2 text-right tabular text-muted">
                    {num(s.antes, 6)}
                  </td>
                  <td className="py-2 text-right tabular font-semibold text-primary">
                    {num(s.depois, 6)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div
          className={`rounded-lg border px-4 py-3 ${
            traco.nova_predicao.reduziu
              ? 'border-emerald-500/30 bg-emerald-500/5'
              : 'border-rose-500/30 bg-rose-500/5'
          }`}
        >
          <p className="font-mono text-sm">
            E: {num(traco.nova_predicao.erro_antes, 6)} →{' '}
            <strong>{num(traco.nova_predicao.erro_depois, 6)}</strong>{' '}
            <Badge tom={traco.nova_predicao.reduziu ? 'bom' : 'ruim'}>
              {traco.nova_predicao.reduziu ? 'erro reduziu' : 'erro aumentou'}
            </Badge>
          </p>
          <p className="mt-1.5 text-xs text-secondary">
            {traco.nova_predicao.reduziu
              ? 'O passo de gradiente descendente moveu os pesos na direção correta.'
              : 'Neste passo o erro subiu — pode ocorrer com taxa de aprendizagem alta.'}
          </p>
        </div>
      </Secao>
    </>
  )
}

/* ------------------------------------------------------------ traco por epoca */
function Epoca({ traco }: { traco: TracoEpoca }) {
  return (
    <>
      <Secao numero={1} titulo="Arquitetura e pesos iniciais">
        <DiagramaRede arquitetura={traco.arquitetura} />
        {traco.config.nota && <Nota tom="info">{traco.config.nota}</Nota>}
        <div className="rounded-lg border border-subtle bg-sunken p-3">
          <p className="kicker mb-2">tabela-verdade do XOR</p>
          <div className="flex flex-wrap gap-2 font-mono text-xs">
            {traco.padroes.map((p, i) => (
              <span
                key={i}
                className="rounded bg-surface px-2 py-1 text-secondary"
              >
                ({p.entrada.map((v) => v.toFixed(0)).join(', ')}) → {p.alvo}
              </span>
            ))}
          </div>
        </div>
      </Secao>

      <Secao numero={2} titulo="Fórmulas aplicadas a cada padrão">
        <BlocoFormula
          latex={String.raw`z = \sum_j w_j a_j + b \qquad a = \sigma(z)`}
        />
        <BlocoFormula
          latex={String.raw`\delta_o = (z_o - t_o) z_o (1-z_o) \qquad
            \delta_h = \left(\sum_o \delta_o w_{ho}\right)\text{out}_h(1-\text{out}_h)`}
        />
        <BlocoFormula
          latex={String.raw`w_{\text{novo}} = w - \eta\,\delta\,\text{entrada}`}
          explicacao="Modo online: os pesos são atualizados após cada padrão, antes de processar o próximo."
        />
      </Secao>

      <Secao numero={3} titulo="A época · 4 padrões em sequência">
        <p className="text-sm leading-relaxed text-secondary">
          O erro de cada padrão é calculado <strong>antes</strong> da
          atualização daquele padrão — mesma convenção do Perceptron e da Regra
          Delta neste projeto.
        </p>
        {traco.passos.map((p) => (
          <div
            key={p.indice}
            className="rounded-lg border border-subtle bg-sunken p-3"
          >
            <p className="kicker mb-2">
              padrão {p.indice} · x = ({p.entrada.map((v) => v.toFixed(0)).join(', ')}
              ) · alvo = {p.alvo}
            </p>
            <div className="space-y-1 font-mono text-[12px] text-secondary">
              <div>
                out_h = [{p.saida_oculta.map((v) => num(v, 4)).join(', ')}] ·
                out = <strong>{num(p.saida, 4)}</strong> · erro ={' '}
                {num(p.erro, 5)}
              </div>
              <div>
                δ_saída = {num(p.delta_saida[0], 6)} · δ_h = [
                {p.delta_oculta.map((v) => num(v, 6)).join(', ')}]
              </div>
            </div>
          </div>
        ))}
        <div className="rounded-md bg-accent-500/10 px-3 py-2 font-mono text-sm font-semibold text-accent-700 dark:text-accent-400">
          Erro médio da época = {num(traco.erro_medio, 5)}
        </div>
      </Secao>

      <Secao numero={4} titulo="Resultado após a época">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-subtle">
                <th className="py-2 text-left text-[11px] font-semibold text-muted">
                  Padrão
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Alvo
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Antes
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Depois
                </th>
                <th className="py-2 text-right text-[11px] font-semibold text-muted">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {traco.resultados.map((r, i) => (
                <tr key={i} className="border-b border-subtle/60 last:border-0">
                  <td className="py-2 font-mono text-secondary">
                    ({r.entrada.map((v) => v.toFixed(0)).join(', ')})
                  </td>
                  <td className="py-2 text-right tabular text-secondary">
                    {r.alvo}
                  </td>
                  <td className="py-2 text-right tabular text-muted">
                    {num(r.antes, 4)}
                  </td>
                  <td className="py-2 text-right tabular font-semibold text-primary">
                    {num(r.depois, 4)}
                  </td>
                  <td className="py-2 text-right">
                    <Badge tom={r.correto ? 'bom' : 'ruim'}>
                      {r.correto ? 'correto' : 'incorreto'}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Nota tom="atencao" titulo={`${traco.acertos}/${traco.total} padrões corretos após 1 época`}>
          O XOR não é linearmente separável: as saídas permanecem próximas de{' '}
          <Formula>{String.raw`0{,}5`}</Formula> (região de máxima incerteza da
          sigmoide) após apenas uma época. São necessárias muitas épocas de
          gradiente descendente para a rede separar os 4 padrões — o que
          confirma, na prática, por que a camada oculta não linear é
          indispensável.
        </Nota>
      </Secao>
    </>
  )
}
