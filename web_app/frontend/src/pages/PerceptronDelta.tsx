/** Lab 2 — Perceptron de Rosenblatt, Regra Delta e o limite do XOR. */
import { useQuery } from '@tanstack/react-query'
import { CheckCircle2, FileText, XCircle } from 'lucide-react'
import { useState } from 'react'
import { PainelConfig, usarConfig, usarMetadata } from '@/components/Controles'
import { BlocoFormula } from '@/components/Formula'
import { GraficoDecisao } from '@/components/GraficoDecisao'
import { GraficoLinha } from '@/components/GraficoLinha'
import { MemoriaGenerica } from '@/components/MemoriaGenerica'
import { MatrizConfusao } from '@/components/Metricas'
import {
  Botao,
  Card,
  Carregando,
  ErroBox,
  Legenda,
  Metrica,
  Nota,
  Segmentos,
  Select,
  Slider,
  Vazio,
} from '@/components/ui'
import { api } from '@/lib/api'
import { cap, cn, corDaClasse, num, pct } from '@/lib/utils'

type Modo = 'perceptron' | 'delta' | 'ova' | 'xor'

export function PaginaPerceptronDelta() {
  const [modo, setModo] = useState<Modo>('perceptron')

  return (
    <div className="space-y-6">
      <Segmentos
        valor={modo}
        onChange={setModo}
        opcoes={[
          { valor: 'perceptron', rotulo: 'Perceptron' },
          { valor: 'delta', rotulo: 'Regra Delta' },
          { valor: 'ova', rotulo: 'Delta OvA (multiclasse)' },
          { valor: 'xor', rotulo: 'XOR — limite linear' },
        ]}
      />

      {modo === 'xor' ? (
        <PainelXor />
      ) : modo === 'ova' ? (
        <PainelOva />
      ) : (
        <PainelBinario algoritmo={modo} />
      )}
    </div>
  )
}

/* --------------------------------------------------- binario (perceptron/delta) */
function PainelBinario({ algoritmo }: { algoritmo: 'perceptron' | 'delta' }) {
  const { config, set } = usarConfig()
  const { data: meta } = usarMetadata()
  const [par, setPar] = useState(0)
  const [taxa, setTaxa] = useState(algoritmo === 'perceptron' ? 0.03 : 0.02)
  const [epocas, setEpocas] = useState(100)
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  const pares = meta?.pares ?? [{ pos: 'setosa', neg: 'versicolor' } as const]
  const parAtual = pares[Math.min(par, pares.length - 1)]

  const q = useQuery({
    queryKey: ['pd', algoritmo, config, parAtual, taxa, epocas],
    queryFn: () =>
      api.perceptronDelta.binario({
        ...config,
        algoritmo,
        classe_pos: parAtual.pos,
        classe_neg: parAtual.neg,
        taxa,
        max_epocas: epocas,
      }),
  })

  const memoria = useQuery({
    queryKey: ['pd', 'memoria', algoritmo, config, parAtual, taxa, epocas],
    queryFn: () =>
      api.perceptronDelta.memoria({
        ...config,
        algoritmo,
        classe_pos: parAtual.pos,
        classe_neg: parAtual.neg,
        taxa,
        max_epocas: epocas,
      }),
    enabled: memoriaAberta,
  })

  const d = q.data
  const ehPerceptron = algoritmo === 'perceptron'

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set}>
          <Select
            rotulo="Par de classes"
            valor={String(par)}
            onChange={(v) => setPar(Number(v))}
            opcoes={pares.map((p, i) => ({
              valor: String(i),
              rotulo: `${cap(p.pos)} × ${cap(p.neg)}`,
            }))}
          />
          <Slider
            rotulo="Taxa de aprendizagem (η)"
            valor={taxa}
            onChange={setTaxa}
            min={0.005}
            max={0.2}
            passo={0.005}
            formatar={(v) => v.toFixed(3)}
          />
          <Slider
            rotulo="Máximo de épocas"
            valor={epocas}
            onChange={setEpocas}
            min={10}
            max={500}
            passo={10}
          />
        </PainelConfig>

        <Card titulo="memória de cálculo">
          <Botao
            variante="primario"
            className="w-full"
            onClick={() => setMemoriaAberta(true)}
          >
            <FileText size={15} />
            Ver teoria e cálculos
          </Botao>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Bias trick, regra de atualização, pesos treinados e a
            classificação de uma amostra com substituição numérica.
          </p>
        </Card>

        <Card titulo={ehPerceptron ? 'perceptron de rosenblatt' : 'regra delta'}>
          <p className="text-sm leading-relaxed text-secondary">
            {ehPerceptron
              ? 'Ativação por limiar. Os pesos só mudam quando a amostra é classificada errado — e a convergência é garantida apenas se as classes forem linearmente separáveis.'
              : 'Saída linear contínua (sem limiar). Minimiza o erro quadrático médio por gradiente descendente, ajustando os pesos em todas as amostras.'}
          </p>
          <BlocoFormula
            className="mt-3"
            titulo={ehPerceptron ? 'ativação' : 'saída linear'}
            latex={
              ehPerceptron
                ? String.raw`y = \text{sgn}(w^{T}x_{\text{aug}}) = \begin{cases}+1 & w^{T}x \ge 0\\ -1 & \text{caso contrário}\end{cases}`
                : String.raw`y = w^{T} x_{\text{aug}} \qquad \text{MSE} = \frac{1}{N}\sum (d - y)^2`
            }
          />
          <BlocoFormula
            className="mt-2"
            titulo="atualização"
            latex={
              ehPerceptron
                ? String.raw`w \leftarrow w + \eta\,(d - y)\, x_{\text{aug}}`
                : String.raw`w \leftarrow w + \eta\,(d - y)\, x_{\text{aug}}`
            }
            explicacao={
              ehPerceptron
                ? 'Aplicada somente nas amostras classificadas incorretamente.'
                : 'Aplicada em todas as amostras, a cada época.'
            }
          />
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metrica
                rotulo="Acurácia (teste)"
                valor={pct(d.acuracia_teste)}
                destaque={d.acuracia_teste >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica rotulo="Acurácia (treino)" valor={pct(d.acuracia_treino)} />
              <Metrica
                rotulo="Épocas treinadas"
                valor={d.epocas}
                detalhe={`limite ${epocas}`}
              />
              <Metrica
                rotulo={ehPerceptron ? 'Convergiu?' : 'MSE final'}
                valor={
                  ehPerceptron
                    ? d.convergiu
                      ? 'Sim'
                      : 'Não'
                    : num(d.historico.at(-1), 5)
                }
                destaque={d.convergiu ? 'bom' : 'medio'}
              />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="fronteira de decisão">
                {d.bidimensional ? (
                  <GraficoDecisao
                    amostras={d.amostras}
                    limites={d.limites}
                    eixoX={d.eixo_x}
                    eixoY={d.eixo_y}
                    retas={[{ w: [d.pesos[1], d.pesos[2]], b: d.pesos[0] }]}
                    altura={340}
                  />
                ) : (
                  <Vazio texto="A fronteira só é desenhável no modo 2D (pétalas ou sépalas)." />
                )}
              </Card>

              <Card
                titulo={
                  ehPerceptron ? 'erros por época' : 'convergência do MSE'
                }
              >
                <GraficoLinha
                  dados={d.historico.map((v, i) => ({ x: i + 1, valor: v }))}
                  series={[
                    {
                      chave: 'valor',
                      rotulo: ehPerceptron ? 'erros' : 'MSE',
                      cor: ehPerceptron ? '#8b5cf6' : '#f59e0b',
                    },
                  ]}
                  rotuloX="época"
                  rotuloY={ehPerceptron ? 'nº de erros' : 'MSE'}
                  altura={300}
                />
                <Legenda
                  className="mt-2"
                  itens={[
                    {
                      cor: ehPerceptron ? '#8b5cf6' : '#f59e0b',
                      forma: 'linha',
                      rotulo: ehPerceptron
                        ? 'erros de classificação na época'
                        : 'erro quadrático médio na época',
                    },
                  ]}
                />
              </Card>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="pesos treinados">
                <div className="space-y-2">
                  {d.pesos.map((w, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between rounded-lg border border-subtle bg-sunken px-3 py-2"
                    >
                      <span className="font-mono text-xs text-secondary">
                        {i === 0 ? 'w₀ (bias)' : `w${i}`}
                      </span>
                      <span className="tabular text-sm font-semibold text-primary">
                        {num(w, 6)}
                      </span>
                    </div>
                  ))}
                </div>
                {d.bidimensional && (
                  <p className="mt-3 break-all font-mono text-xs text-muted">
                    Reta: {num(d.pesos[0], 4)} + {num(d.pesos[1], 4)}·x₁ +{' '}
                    {num(d.pesos[2], 4)}·x₂ = 0
                  </p>
                )}
              </Card>

              <Card titulo="matriz de confusão (teste)">
                <MatrizConfusao
                  relatorio={d.relatorio}
                  classes={[d.classe_pos, d.classe_neg]}
                />
              </Card>
            </div>
          </>
        )}
      </div>

      {memoriaAberta && (
        <MemoriaGenerica
          traco={memoria.data}
          carregando={memoria.isPending}
          erro={memoria.error}
          onFechar={() => setMemoriaAberta(false)}
        />
      )}
    </div>
  )
}

/* -------------------------------------------------------------- Delta OvA */
function PainelOva() {
  const { config, set } = usarConfig()
  const [taxa, setTaxa] = useState(0.02)
  const [epocas, setEpocas] = useState(200)

  const q = useQuery({
    queryKey: ['pd', 'ova', config, taxa, epocas],
    queryFn: () =>
      api.perceptronDelta.ova({ ...config, taxa, max_epocas: epocas }),
  })

  const d = q.data
  const classes = ['setosa', 'versicolor', 'virginica']

  const dadosCurva =
    d &&
    Array.from({ length: d.epocas }, (_, i) => {
      const ponto: Record<string, number> = { x: i + 1 }
      for (const c of classes) ponto[c] = d.historico[c]?.[i] ?? 0
      return ponto
    })

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set}>
          <Slider
            rotulo="Taxa de aprendizagem (η)"
            valor={taxa}
            onChange={setTaxa}
            min={0.005}
            max={0.1}
            passo={0.005}
            formatar={(v) => v.toFixed(3)}
          />
          <Slider
            rotulo="Épocas"
            valor={epocas}
            onChange={setEpocas}
            min={20}
            max={600}
            passo={20}
          />
        </PainelConfig>

        <Card titulo="um-contra-todos (ova)">
          <p className="text-sm leading-relaxed text-secondary">
            Treina um classificador binário por classe (a classe alvo vira{' '}
            <span className="font-mono">+1</span>, todas as outras{' '}
            <span className="font-mono">−1</span>) e decide pelo maior net.
          </p>
          <BlocoFormula
            className="mt-3"
            latex={String.raw`\text{classe}^* = \arg\max_c \; w_c^{T} x_{\text{aug}}`}
          />
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && dadosCurva && (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Metrica
                rotulo="Acerto Global"
                valor={pct(d.relatorio.acerto_global)}
                destaque={d.relatorio.acerto_global >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica rotulo="Kappa" valor={num(d.relatorio.kappa, 4)} />
              <Metrica rotulo="Épocas" valor={d.epocas} />
            </div>

            <Card titulo="convergência por classificador">
              <GraficoLinha
                dados={dadosCurva}
                series={classes.map((c) => ({
                  chave: c,
                  rotulo: cap(c),
                  cor: corDaClasse(c),
                }))}
                rotuloX="época"
                rotuloY="MSE"
                altura={300}
              />
              <Legenda
                className="mt-2"
                titulo="Curvas"
                itens={classes.map((c) => ({
                  cor: corDaClasse(c),
                  forma: 'linha' as const,
                  rotulo: `${cap(c)} vs resto`,
                }))}
              />
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="dispersão das amostras">
                {d.bidimensional ? (
                  <GraficoDecisao
                    amostras={d.amostras}
                    limites={d.limites}
                    eixoX={d.eixo_x}
                    eixoY={d.eixo_y}
                    altura={340}
                  />
                ) : (
                  <Vazio texto="Visualização 2D indisponível para 4 features." />
                )}
              </Card>
              <Card titulo="matriz de confusão">
                <MatrizConfusao relatorio={d.relatorio} classes={classes} />
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------- XOR */
function PainelXor() {
  const [taxa, setTaxa] = useState(0.02)
  const [epocas, setEpocas] = useState(300)

  const q = useQuery({
    queryKey: ['pd', 'xor', taxa, epocas],
    queryFn: () => api.perceptronDelta.xor({ taxa, max_epocas: epocas }),
  })

  const d = q.data

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <Card titulo="parâmetros">
          <div className="space-y-4">
            <Slider
              rotulo="Taxa de aprendizagem (η)"
              valor={taxa}
              onChange={setTaxa}
              min={0.005}
              max={0.2}
              passo={0.005}
              formatar={(v) => v.toFixed(3)}
            />
            <Slider
              rotulo="Épocas"
              valor={epocas}
              onChange={setEpocas}
              min={50}
              max={2000}
              passo={50}
            />
          </div>
        </Card>

        <Card titulo="o problema xor">
          <p className="text-sm leading-relaxed text-secondary">
            O XOR é o contraexemplo clássico: nenhuma reta separa{' '}
            <span className="font-mono">{'{(0,0),(1,1)}'}</span> de{' '}
            <span className="font-mono">{'{(0,1),(1,0)}'}</span>. Por isso o MSE
            estaciona em <strong>0,25</strong> e nunca zera — não importa
            quantas épocas.
          </p>
          <Nota tom="atencao" className="mt-3">
            Este é exatamente o limite que motiva a camada oculta não linear do{' '}
            <strong>Lab 5.0</strong>, onde a mesma tabela-verdade é resolvida
            com uma MLP.
          </Nota>
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Metrica
                rotulo="MSE final"
                valor={num(d.mse_final, 5)}
                detalhe="teórico: 0,25"
                destaque="medio"
              />
              <Metrica
                rotulo="Padrões corretos"
                valor={`${d.acertos} / 4`}
                destaque={d.acertos === 4 ? 'bom' : 'ruim'}
              />
              <Metrica rotulo="Épocas" valor={d.historico.length} />
            </div>

            <Card titulo="convergência do MSE">
              <GraficoLinha
                dados={d.historico.map((v, i) => ({ x: i + 1, mse: v }))}
                series={[{ chave: 'mse', rotulo: 'MSE', cor: '#f59e0b' }]}
                rotuloX="época"
                rotuloY="MSE"
                altura={300}
                referencia={{ y: 0.25, rotulo: 'piso teórico = 0,25' }}
              />
              <Legenda
                className="mt-2"
                itens={[
                  { cor: '#f59e0b', forma: 'linha', rotulo: 'MSE por época' },
                  {
                    cor: 'hsl(var(--text-muted))',
                    forma: 'linha-tracejada',
                    rotulo: 'piso teórico (0,25)',
                    descricao: 'o MSE nunca desce abaixo disso',
                  },
                ]}
              />
            </Card>

            <Card titulo="tabela-verdade e saídas da rede">
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
                        net
                      </th>
                      <th className="py-2 text-right text-[11px] font-semibold text-muted">
                        Previsto
                      </th>
                      <th className="py-2 text-right text-[11px] font-semibold text-muted">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.padroes.map((p, i) => (
                      <tr key={i} className="border-b border-subtle/60 last:border-0">
                        <td className="py-2.5 font-mono text-secondary">
                          ({p.x1.toFixed(0)}, {p.x2.toFixed(0)})
                        </td>
                        <td className="py-2.5 text-right tabular text-secondary">
                          {p.alvo}
                        </td>
                        <td className="py-2.5 text-right tabular text-primary">
                          {num(p.net, 4)}
                        </td>
                        <td className="py-2.5 text-right tabular text-secondary">
                          {p.previsto}
                        </td>
                        <td className="py-2.5 text-right">
                          <span
                            className={cn(
                              'inline-flex items-center gap-1 text-xs font-medium',
                              p.correto
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400',
                            )}
                          >
                            {p.correto ? (
                              <CheckCircle2 size={14} />
                            ) : (
                              <XCircle size={14} />
                            )}
                            {p.correto ? 'correto' : 'incorreto'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <Nota tom="atencao" titulo="Conclusão" className="mt-4">
                Com apenas {d.acertos} de 4 padrões corretos e o MSE preso em{' '}
                {num(d.mse_final, 4)}, fica demonstrado o limite teórico dos
                classificadores lineares. A solução exige uma camada oculta —
                veja o Lab 5.0.
              </Nota>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
