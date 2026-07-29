/**
 * Construtor de Rede — monte a MLP camada a camada.
 *
 * Permite adicionar/remover neuronios, editar cada peso e bias na mao,
 * definir os padroes de treino e acompanhar o resultado na mesma linha do
 * tempo dos exercicios da aula. Toda a matematica continua no backend.
 */
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Dices,
  FileText,
  Minus,
  Plus,
  Sparkles,
  Trash2,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { DiagramaRede } from '@/components/DiagramaRede'
import { LinhaDoTempo, usarLinhaDoTempo } from '@/components/LinhaDoTempo'
import { MemoriaCalculo } from '@/components/MemoriaCalculo'
import {
  Badge,
  Botao,
  Card,
  Carregando,
  ErroBox,
  Metrica,
  Nota,
  Slider,
} from '@/components/ui'
import { api } from '@/lib/api'
import type { Traco } from '@/lib/types'
import { cn, forward, num } from '@/lib/utils'

interface Padrao {
  entrada: number[]
  alvo: number[]
}

interface EstadoRede {
  pesos_oculta: number[][]
  bias_oculta: number[]
  pesos_saida: number[][]
  bias_saida: number[]
  padroes: Padrao[]
}

/* ------------------------------------------------------------ predefinidos */
const PRESETS: Record<string, { rotulo: string; descricao: string; rede: EstadoRede }> = {
  xor: {
    rotulo: 'XOR',
    descricao: 'não linearmente separável — precisa da camada oculta',
    rede: {
      pesos_oculta: [
        [0.5, 0.5],
        [-0.5, -0.5],
      ],
      bias_oculta: [-0.2, 0.3],
      pesos_saida: [[0.6, -0.6]],
      bias_saida: [-0.1],
      padroes: [
        { entrada: [0, 0], alvo: [0] },
        { entrada: [0, 1], alvo: [1] },
        { entrada: [1, 0], alvo: [1] },
        { entrada: [1, 1], alvo: [0] },
      ],
    },
  },
  and: {
    rotulo: 'AND',
    descricao: 'linearmente separável — converge rápido',
    rede: {
      pesos_oculta: [
        [0.3, 0.3],
        [-0.2, 0.4],
      ],
      bias_oculta: [0.1, -0.1],
      pesos_saida: [[0.5, 0.5]],
      bias_saida: [-0.2],
      padroes: [
        { entrada: [0, 0], alvo: [0] },
        { entrada: [0, 1], alvo: [0] },
        { entrada: [1, 0], alvo: [0] },
        { entrada: [1, 1], alvo: [1] },
      ],
    },
  },
  didatico: {
    rotulo: 'Slide 37',
    descricao: 'o exemplo didático da aula, com uma amostra só',
    rede: {
      pesos_oculta: [
        [0.15, 0.2],
        [0.25, 0.3],
      ],
      bias_oculta: [0.35, 0.35],
      pesos_saida: [
        [0.4, 0.45],
        [0.5, 0.55],
      ],
      bias_saida: [0.6, 0.6],
      padroes: [{ entrada: [0.05, 0.1], alvo: [0.01, 0.99] }],
    },
  },
}

export function PaginaConstrutor() {
  const [rede, setRede] = useState<EstadoRede>(PRESETS.xor.rede)
  const [epocas, setEpocas] = useState(4000)
  const [taxa, setTaxa] = useState(0.5)
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  const nEntradas = rede.pesos_oculta[0]?.length ?? 0
  const nOcultos = rede.bias_oculta.length
  const nSaidas = rede.bias_saida.length

  const corpo = useMemo(
    () => ({ ...rede, taxa, epocas, n_snapshots: 80 }),
    [rede, taxa, epocas],
  )

  const treino = useMutation({
    mutationFn: () => api.lab5.redeTrajetoria(corpo),
  })

  const memoria = useQuery({
    queryKey: ['lab5', 'rede', 'memoria', corpo],
    queryFn: () => api.lab5.redeMemoria(corpo) as Promise<Traco>,
    enabled: memoriaAberta,
  })

  const trajetoria = treino.data
  const { indice, setIndice, snapshot } = usarLinhaDoTempo(trajetoria)

  /* ---------------------------------------------- alteracoes na topologia */
  const mudarEntradas = (delta: number) => {
    const alvo = Math.max(1, Math.min(8, nEntradas + delta))
    if (alvo === nEntradas) return
    setRede((r) => ({
      ...r,
      pesos_oculta: r.pesos_oculta.map((linha) => redimensionar(linha, alvo)),
      padroes: r.padroes.map((p) => ({
        ...p,
        entrada: redimensionar(p.entrada, alvo, 0),
      })),
    }))
  }

  const mudarOcultos = (delta: number) => {
    const alvo = Math.max(1, Math.min(10, nOcultos + delta))
    if (alvo === nOcultos) return
    setRede((r) => ({
      ...r,
      pesos_oculta: redimensionarMatriz(r.pesos_oculta, alvo, nEntradas),
      bias_oculta: redimensionar(r.bias_oculta, alvo),
      pesos_saida: r.pesos_saida.map((linha) => redimensionar(linha, alvo)),
    }))
  }

  const mudarSaidas = (delta: number) => {
    const alvo = Math.max(1, Math.min(6, nSaidas + delta))
    if (alvo === nSaidas) return
    setRede((r) => ({
      ...r,
      pesos_saida: redimensionarMatriz(r.pesos_saida, alvo, nOcultos),
      bias_saida: redimensionar(r.bias_saida, alvo),
      padroes: r.padroes.map((p) => ({
        ...p,
        alvo: redimensionar(p.alvo, alvo, 0),
      })),
    }))
  }

  const sortearPesos = () =>
    setRede((r) => ({
      ...r,
      pesos_oculta: r.pesos_oculta.map((l) => l.map(() => aleatorio())),
      bias_oculta: r.bias_oculta.map(() => aleatorio()),
      pesos_saida: r.pesos_saida.map((l) => l.map(() => aleatorio())),
      bias_saida: r.bias_saida.map(() => aleatorio()),
    }))

  const arquitetura = {
    rotulos_entrada: Array.from({ length: nEntradas }, (_, i) => `x${i + 1}`),
    rotulos_ocultos: Array.from({ length: nOcultos }, (_, i) => `h${i + 1}`),
    rotulos_saida: Array.from({ length: nSaidas }, (_, i) => `y${i + 1}`),
    pesos_oculta: rede.pesos_oculta,
    bias_oculta: rede.bias_oculta,
    pesos_saida: rede.pesos_saida,
    bias_saida: rede.bias_saida,
    bias_compartilhado: false,
    texto: `${nEntradas}-${nOcultos}-${nSaidas}`,
  }

  // Saidas atuais, calculadas no cliente para dar retorno imediato ao editar
  const saidasAgora = rede.padroes.map(
    (p) => forward(p.entrada, rede).saidas,
  )

  return (
    <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
      {/* ----------------------------------------------------- construcao */}
      <div className="space-y-5">
        <Card titulo="topologia da rede">
          <div className="space-y-3">
            <ContadorCamada
              rotulo="Entradas"
              valor={nEntradas}
              onMudar={mudarEntradas}
              min={1}
              max={8}
              cor="#94a3b8"
            />
            <ContadorCamada
              rotulo="Neurônios ocultos"
              valor={nOcultos}
              onMudar={mudarOcultos}
              min={1}
              max={10}
              cor="#f59e0b"
            />
            <ContadorCamada
              rotulo="Saídas"
              valor={nSaidas}
              onMudar={mudarSaidas}
              min={1}
              max={6}
              cor="#0ea5e9"
            />
          </div>
          <div className="mt-4 flex items-center justify-between rounded-lg bg-sunken px-3 py-2">
            <span className="text-xs text-muted">Arquitetura</span>
            <Badge tom="info">{arquitetura.texto}</Badge>
          </div>
          <Botao className="mt-3 w-full" onClick={sortearPesos}>
            <Dices size={14} />
            Sortear pesos
          </Botao>
        </Card>

        <Card titulo="começar de um exemplo">
          <div className="grid gap-2">
            {Object.entries(PRESETS).map(([chave, p]) => (
              <button
                key={chave}
                onClick={() => setRede(clonar(p.rede))}
                className="rounded-lg border border-subtle bg-sunken px-3 py-2 text-left transition-colors hover:border-accent-500/50"
              >
                <span className="text-sm font-medium text-primary">
                  {p.rotulo}
                </span>
                <span className="mt-0.5 block text-xs text-muted">
                  {p.descricao}
                </span>
              </button>
            ))}
          </div>
        </Card>

        <Card titulo="treinar">
          <div className="space-y-4">
            <Slider
              rotulo="Épocas"
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
              onClick={() => treino.mutate()}
              disabled={treino.isPending}
            >
              <Sparkles size={15} />
              {treino.isPending ? 'Treinando…' : 'Treinar esta rede'}
            </Botao>
            <Botao className="w-full" onClick={() => setMemoriaAberta(true)}>
              <FileText size={14} />
              Memória de cálculo (1 passo)
            </Botao>
          </div>
        </Card>
      </div>

      {/* ------------------------------------------------------ resultados */}
      <div className="space-y-6">
        <Card titulo="diagrama da rede">
          <DiagramaRede arquitetura={arquitetura} />
          <Nota tom="info" className="mt-3">
            O diagrama reflete os pesos abaixo em tempo real — a espessura de
            cada conexão acompanha a magnitude do peso.
          </Nota>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card titulo="pesos · entrada → oculta">
            <MatrizPesos
              matriz={rede.pesos_oculta}
              bias={rede.bias_oculta}
              rotuloLinha={(i) => `h${i + 1}`}
              rotuloColuna={(j) => `x${j + 1}`}
              onMudarPeso={(i, j, v) =>
                setRede((r) => {
                  const m = r.pesos_oculta.map((l) => [...l])
                  m[i][j] = v
                  return { ...r, pesos_oculta: m }
                })
              }
              onMudarBias={(i, v) =>
                setRede((r) => {
                  const b = [...r.bias_oculta]
                  b[i] = v
                  return { ...r, bias_oculta: b }
                })
              }
            />
          </Card>

          <Card titulo="pesos · oculta → saída">
            <MatrizPesos
              matriz={rede.pesos_saida}
              bias={rede.bias_saida}
              rotuloLinha={(i) => `y${i + 1}`}
              rotuloColuna={(j) => `h${j + 1}`}
              onMudarPeso={(i, j, v) =>
                setRede((r) => {
                  const m = r.pesos_saida.map((l) => [...l])
                  m[i][j] = v
                  return { ...r, pesos_saida: m }
                })
              }
              onMudarBias={(i, v) =>
                setRede((r) => {
                  const b = [...r.bias_saida]
                  b[i] = v
                  return { ...r, bias_saida: b }
                })
              }
            />
          </Card>
        </div>

        <Card
          titulo="padrões de treino"
          acao={
            <Botao
              tamanho="sm"
              onClick={() =>
                setRede((r) => ({
                  ...r,
                  padroes: [
                    ...r.padroes,
                    {
                      entrada: Array(nEntradas).fill(0),
                      alvo: Array(nSaidas).fill(0),
                    },
                  ],
                }))
              }
            >
              <Plus size={13} />
              Adicionar padrão
            </Botao>
          }
        >
          <TabelaPadroes
            padroes={rede.padroes}
            saidasAgora={saidasAgora}
            nEntradas={nEntradas}
            nSaidas={nSaidas}
            onMudar={(idx, campo, pos, v) =>
              setRede((r) => {
                const ps = r.padroes.map((p) => ({
                  entrada: [...p.entrada],
                  alvo: [...p.alvo],
                }))
                ps[idx][campo][pos] = v
                return { ...r, padroes: ps }
              })
            }
            onRemover={(idx) =>
              setRede((r) => ({
                ...r,
                padroes:
                  r.padroes.length > 1
                    ? r.padroes.filter((_, i) => i !== idx)
                    : r.padroes,
              }))
            }
          />
          <p className="mt-3 text-xs text-muted">
            A coluna <strong>saída agora</strong> é calculada no navegador com
            os pesos atuais — dá retorno imediato enquanto você edita, antes
            mesmo de treinar.
          </p>
        </Card>

        {treino.error ? <ErroBox erro={treino.error} /> : null}
        {treino.isPending && (
          <Card>
            <Carregando texto="Treinando a rede montada…" />
          </Card>
        )}

        {trajetoria && snapshot && (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Metrica rotulo="Época" valor={snapshot.epoca} />
              <Metrica
                rotulo="Erro nesta época"
                valor={snapshot.erro !== null ? num(snapshot.erro, 6) : '—'}
              />
              <Metrica
                rotulo="Erro final"
                valor={num(trajetoria.historico.at(-1), 6)}
                destaque={
                  (trajetoria.historico.at(-1) ?? 1) < 0.01 ? 'bom' : 'medio'
                }
              />
            </div>

            <Card titulo="linha do tempo do treinamento">
              <LinhaDoTempo
                trajetoria={trajetoria}
                indice={indice}
                onMudarIndice={setIndice}
              />
              <div className="mt-5">
                <p className="kicker mb-2">saídas nesta época</p>
                <TabelaSaidas
                  padroes={trajetoria.padroes}
                  saidas={snapshot.saidas}
                />
              </div>
            </Card>

            <Card titulo={`rede treinada — época ${snapshot.epoca}`}>
              <DiagramaRede
                arquitetura={{
                  ...arquitetura,
                  pesos_oculta: snapshot.pesos_oculta,
                  bias_oculta: snapshot.bias_oculta,
                  pesos_saida: snapshot.pesos_saida,
                  bias_saida: snapshot.bias_saida,
                }}
              />
              <Botao
                className="mt-3"
                tamanho="sm"
                onClick={() =>
                  setRede((r) => ({
                    ...r,
                    pesos_oculta: snapshot.pesos_oculta.map((l) => [...l]),
                    bias_oculta: [...snapshot.bias_oculta],
                    pesos_saida: snapshot.pesos_saida.map((l) => [...l]),
                    bias_saida: [...snapshot.bias_saida],
                  }))
                }
              >
                Adotar estes pesos como ponto de partida
              </Botao>
            </Card>
          </>
        )}
      </div>

      {memoriaAberta && (
        <MemoriaCalculo
          traco={memoria.data}
          carregando={memoria.isPending}
          erro={memoria.error}
          onFechar={() => setMemoriaAberta(false)}
        />
      )}
    </div>
  )
}

/* ------------------------------------------------------------ componentes */
function ContadorCamada({
  rotulo,
  valor,
  onMudar,
  min,
  max,
  cor,
}: {
  rotulo: string
  valor: number
  onMudar: (delta: number) => void
  min: number
  max: number
  cor: string
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="flex items-center gap-2 text-sm text-secondary">
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: cor }}
        />
        {rotulo}
      </span>
      <span className="flex items-center gap-1.5">
        <Botao
          tamanho="sm"
          onClick={() => onMudar(-1)}
          disabled={valor <= min}
          className="!px-2"
        >
          <Minus size={13} />
        </Botao>
        <span className="w-7 text-center tabular text-sm font-semibold text-primary">
          {valor}
        </span>
        <Botao
          tamanho="sm"
          onClick={() => onMudar(1)}
          disabled={valor >= max}
          className="!px-2"
        >
          <Plus size={13} />
        </Botao>
      </span>
    </div>
  )
}

function CampoNumero({
  valor,
  onMudar,
  largura = 'w-[68px]',
}: {
  valor: number
  onMudar: (v: number) => void
  largura?: string
}) {
  return (
    <input
      type="number"
      step={0.05}
      value={Number.isFinite(valor) ? valor : 0}
      onChange={(e) => {
        const v = Number(e.target.value)
        onMudar(Number.isFinite(v) ? v : 0)
      }}
      className={cn(
        'rounded border border-subtle bg-surface px-1.5 py-1 text-center',
        'tabular text-xs text-primary transition-colors',
        'hover:border-strong focus:border-accent-500 focus:outline-none',
        largura,
      )}
    />
  )
}

function MatrizPesos({
  matriz,
  bias,
  rotuloLinha,
  rotuloColuna,
  onMudarPeso,
  onMudarBias,
}: {
  matriz: number[][]
  bias: number[]
  rotuloLinha: (i: number) => string
  rotuloColuna: (j: number) => string
  onMudarPeso: (i: number, j: number, v: number) => void
  onMudarBias: (i: number, v: number) => void
}) {
  const colunas = matriz[0]?.length ?? 0
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="pr-2" />
            {Array.from({ length: colunas }, (_, j) => (
              <th key={j} className="px-1 pb-1.5 font-mono text-muted">
                {rotuloColuna(j)}
              </th>
            ))}
            <th className="px-1 pb-1.5 font-mono text-accent-600 dark:text-accent-400">
              bias
            </th>
          </tr>
        </thead>
        <tbody>
          {matriz.map((linha, i) => (
            <tr key={i}>
              <th className="pr-2 text-right font-mono font-medium text-secondary">
                {rotuloLinha(i)}
              </th>
              {linha.map((peso, j) => (
                <td key={j} className="px-0.5 py-0.5">
                  <CampoNumero
                    valor={peso}
                    onMudar={(v) => onMudarPeso(i, j, v)}
                  />
                </td>
              ))}
              <td className="px-0.5 py-0.5">
                <CampoNumero
                  valor={bias[i]}
                  onMudar={(v) => onMudarBias(i, v)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TabelaPadroes({
  padroes,
  saidasAgora,
  nEntradas,
  nSaidas,
  onMudar,
  onRemover,
}: {
  padroes: Padrao[]
  saidasAgora: number[][]
  nEntradas: number
  nSaidas: number
  onMudar: (
    idx: number,
    campo: 'entrada' | 'alvo',
    pos: number,
    v: number,
  ) => void
  onRemover: (idx: number) => void
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-subtle">
            <th
              colSpan={nEntradas}
              className="pb-1.5 text-left font-semibold text-muted"
            >
              Entradas
            </th>
            <th
              colSpan={nSaidas}
              className="pb-1.5 text-left font-semibold text-muted"
            >
              Alvo
            </th>
            <th
              colSpan={nSaidas}
              className="pb-1.5 text-left font-semibold text-muted"
            >
              Saída agora
            </th>
            <th className="pb-1.5" />
          </tr>
        </thead>
        <tbody>
          {padroes.map((p, idx) => (
            <tr key={idx} className="border-b border-subtle/50 last:border-0">
              {p.entrada.map((v, j) => (
                <td key={`e${j}`} className="px-0.5 py-1">
                  <CampoNumero
                    valor={v}
                    onMudar={(nv) => onMudar(idx, 'entrada', j, nv)}
                  />
                </td>
              ))}
              {p.alvo.map((v, j) => (
                <td key={`a${j}`} className="px-0.5 py-1">
                  <CampoNumero
                    valor={v}
                    onMudar={(nv) => onMudar(idx, 'alvo', j, nv)}
                  />
                </td>
              ))}
              {(saidasAgora[idx] ?? []).map((s, j) => {
                const perto = Math.abs(s - p.alvo[j]) < 0.1
                return (
                  <td key={`s${j}`} className="px-1.5 py-1">
                    <span
                      className={cn(
                        'tabular font-medium',
                        perto
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : 'text-secondary',
                      )}
                    >
                      {num(s, 4)}
                    </span>
                  </td>
                )
              })}
              <td className="py-1 pl-2">
                <button
                  onClick={() => onRemover(idx)}
                  disabled={padroes.length <= 1}
                  className="text-muted transition-colors hover:text-rose-500 disabled:opacity-30"
                  aria-label={`Remover padrão ${idx + 1}`}
                >
                  <Trash2 size={13} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TabelaSaidas({
  padroes,
  saidas,
}: {
  padroes: { entrada: number[]; alvo: number[] }[]
  saidas: number[][]
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-subtle">
            <th className="py-1.5 text-left text-[11px] font-semibold text-muted">
              Entrada
            </th>
            <th className="py-1.5 text-right text-[11px] font-semibold text-muted">
              Alvo
            </th>
            <th className="py-1.5 text-right text-[11px] font-semibold text-muted">
              Saída
            </th>
            <th className="py-1.5 text-right text-[11px] font-semibold text-muted">
              Erro
            </th>
          </tr>
        </thead>
        <tbody>
          {padroes.map((p, i) => {
            const s = saidas[i] ?? []
            const erro =
              p.alvo.reduce((acc, a, j) => acc + (a - (s[j] ?? 0)) ** 2, 0) / 2
            return (
              <tr key={i} className="border-b border-subtle/50 last:border-0">
                <td className="py-1.5 font-mono text-xs text-secondary">
                  [{p.entrada.map((v) => num(v, 2)).join(', ')}]
                </td>
                <td className="py-1.5 text-right font-mono text-xs text-secondary">
                  [{p.alvo.map((v) => num(v, 2)).join(', ')}]
                </td>
                <td className="py-1.5 text-right font-mono text-xs font-semibold text-primary">
                  [{s.map((v) => num(v, 4)).join(', ')}]
                </td>
                <td className="py-1.5 text-right tabular text-xs text-muted">
                  {num(erro, 6)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ------------------------------------------------------------- utilitarios */
function aleatorio() {
  return Number((Math.random() - 0.5).toFixed(3))
}

function redimensionar(vetor: number[], tamanho: number, preencher?: number) {
  const novo = vetor.slice(0, tamanho)
  while (novo.length < tamanho) {
    novo.push(preencher ?? aleatorio())
  }
  return novo
}

function redimensionarMatriz(
  matriz: number[][],
  linhas: number,
  colunas: number,
) {
  const nova = matriz.slice(0, linhas).map((l) => redimensionar(l, colunas))
  while (nova.length < linhas) {
    nova.push(Array.from({ length: colunas }, aleatorio))
  }
  return nova
}

function clonar(r: EstadoRede): EstadoRede {
  return {
    pesos_oculta: r.pesos_oculta.map((l) => [...l]),
    bias_oculta: [...r.bias_oculta],
    pesos_saida: r.pesos_saida.map((l) => [...l]),
    bias_saida: [...r.bias_saida],
    padroes: r.padroes.map((p) => ({
      entrada: [...p.entrada],
      alvo: [...p.alvo],
    })),
  }
}
