/**
 * Seminário — Florestas Aleatórias (Random Forests).
 *
 * Implementacao propria em Python puro: arvore CART, bagging, subespaco
 * aleatorio, erro out-of-bag e importancia dos atributos.
 */
import { useQuery } from '@tanstack/react-query'
import { FileText, Target } from 'lucide-react'
import { useState } from 'react'
import { ArvoreDecisao } from '@/components/ArvoreDecisao'
import {
  PainelConfig,
  usarConfig,
  usarDataset,
} from '@/components/Controles'
import { BlocoFormula } from '@/components/Formula'
import { GraficoDecisao } from '@/components/GraficoDecisao'
import { MemoriaGenerica } from '@/components/MemoriaGenerica'
import {
  MatrizConfusao,
  ResumoGlobal,
  TabelaPorClasse,
  TabelaTestesZ,
} from '@/components/Metricas'
import {
  Badge,
  Botao,
  Card,
  Carregando,
  ErroBox,
  Metrica,
  Nota,
  Segmentos,
  Select,
  Slider,
} from '@/components/ui'
import { api, type ParamsFloresta } from '@/lib/api'
import {
  CORES_MODELO,
  cap,
  classesDoRelatorio,
  corDaClasse,
  num,
  pct,
} from '@/lib/utils'

type Modo = 'floresta' | 'arvores' | 'comparativo'

export function PaginaFloresta() {
  const [modo, setModo] = useState<Modo>('floresta')
  const { config, set } = usarConfig()
  const [nArvores, setNArvores] = useState(50)
  const [criterio, setCriterio] = useState<'gini' | 'entropia'>('gini')
  const [profundidade, setProfundidade] = useState(0) // 0 = sem limite
  const [maxAtributos, setMaxAtributos] = useState('sqrt')
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  const params: ParamsFloresta = {
    ...config,
    n_arvores: nArvores,
    criterio,
    profundidade_max: profundidade > 0 ? profundidade : undefined,
    max_atributos: maxAtributos,
  }

  const memoria = useQuery({
    queryKey: ['floresta', 'memoria', params],
    queryFn: () => api.floresta.memoria(params),
    enabled: memoriaAberta,
  })

  const controles = (
    <div className="space-y-5">
      <PainelConfig config={config} set={set}>
        <Slider
          rotulo="Número de árvores"
          valor={nArvores}
          onChange={setNArvores}
          min={1}
          max={200}
          passo={1}
        />
        <Select
          rotulo="Critério de impureza"
          valor={criterio}
          onChange={(v) => setCriterio(v as 'gini' | 'entropia')}
          opcoes={[
            { valor: 'gini', rotulo: 'Gini' },
            { valor: 'entropia', rotulo: 'Entropia' },
          ]}
        />
        <Select
          rotulo="Atributos por nó"
          valor={maxAtributos}
          onChange={setMaxAtributos}
          opcoes={[
            { valor: 'sqrt', rotulo: '√p (padrão)' },
            { valor: 'log2', rotulo: 'log₂ p' },
            { valor: 'todos', rotulo: 'Todos (vira bagging puro)' },
          ]}
        />
        <Slider
          rotulo="Profundidade máxima"
          valor={profundidade}
          onChange={setProfundidade}
          min={0}
          max={12}
          passo={1}
          formatar={(v) => (v === 0 ? 'sem limite' : String(v))}
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
          Impureza, ganho, bagging, subespaço aleatório, erro out-of-bag,
          importância dos atributos e a votação de uma amostra.
        </p>
      </Card>
    </div>
  )

  return (
    <div className="space-y-6">
      <Segmentos
        valor={modo}
        onChange={setModo}
        opcoes={[
          { valor: 'floresta', rotulo: 'A floresta' },
          { valor: 'arvores', rotulo: 'Árvores individuais' },
          { valor: 'comparativo', rotulo: 'Comparativo' },
        ]}
      />

      {modo === 'floresta' && (
        <PainelFloresta
          params={params}
          controles={controles}
          nArvores={nArvores}
        />
      )}
      {modo === 'arvores' && (
        <PainelArvores params={params} controles={controles} criterio={criterio} />
      )}
      {modo === 'comparativo' && (
        <PainelComparativo params={params} controles={controles} />
      )}

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

/* ------------------------------------------------------------- a floresta */
function PainelFloresta({
  params,
  controles,
  nArvores,
}: {
  params: ParamsFloresta
  controles: React.ReactNode
  nArvores: number
}) {
  const [consulta, setConsulta] = useState<{ x: number; y: number } | null>(null)

  const q = useQuery({
    queryKey: ['floresta', 'treinar', params],
    queryFn: () => api.floresta.treinar(params),
  })

  const regioes = useQuery({
    queryKey: ['floresta', 'regioes', params],
    queryFn: () => api.floresta.regioes({ ...params, resolucao: 100 }),
  })

  const predicao = useQuery({
    queryKey: ['floresta', 'predizer', params, consulta],
    queryFn: () =>
      api.floresta.predizer({
        ...params,
        valores: [consulta!.x, consulta!.y],
      }),
    enabled: !!consulta,
  })

  const d = q.data
  const { classes } = usarDataset(String(params.dataset ?? 'v1'))

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        {controles}

        {consulta && (
          <Card
            titulo="votação da floresta"
            acao={
              <Botao
                variante="fantasma"
                tamanho="sm"
                onClick={() => setConsulta(null)}
              >
                limpar
              </Botao>
            }
          >
            {predicao.isPending && <Carregando texto="Votando…" />}
            {predicao.data && (
              <div className="space-y-3">
                <p className="font-mono text-sm text-secondary">
                  ({num(consulta.x, 2)}, {num(consulta.y, 2)})
                </p>
                <div
                  className="rounded-lg px-3 py-2 text-sm font-semibold text-white"
                  style={{ backgroundColor: corDaClasse(predicao.data.classe) }}
                >
                  {cap(predicao.data.classe)}
                </div>
                {classes.map((c) => {
                  const p = predicao.data!.probabilidades[c] ?? 0
                  return (
                    <div key={c}>
                      <div className="mb-0.5 flex justify-between text-xs">
                        <span className="text-secondary">{cap(c)}</span>
                        <span className="tabular text-primary">
                          {predicao.data!.votos[c] ?? 0} votos ({pct(p, 0)})
                        </span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-500/20">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${p * 100}%`,
                            backgroundColor: corDaClasse(c),
                          }}
                        />
                      </div>
                    </div>
                  )
                })}
                <p className="text-xs text-muted">
                  Cada uma das {predicao.data.total_arvores} árvores dá um voto;
                  vence a maioria.
                </p>
              </div>
            )}
          </Card>
        )}
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando texto="Treinando a floresta…" />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metrica
                rotulo="Acerto no teste"
                valor={pct(d.relatorio.acerto_global)}
                detalhe={`${d.n_teste} amostras`}
                destaque={d.relatorio.acerto_global >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica
                rotulo="Acurácia OOB"
                valor={d.oob.acuracia !== null ? pct(d.oob.acuracia) : '—'}
                detalhe="estimativa out-of-bag"
                cor="#16a34a"
              />
              <Metrica
                rotulo="Árvore única"
                valor={pct(d.relatorio_arvore_unica.acerto_global)}
                detalhe="sem ensemble"
                cor="#8b5cf6"
              />
              <Metrica
                rotulo="Kappa"
                valor={num(d.relatorio.kappa, 4)}
                detalhe={`${nArvores} árvores`}
              />
            </div>

            <Card
              titulo="regiões de decisão da floresta"
              acao={
                <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                  <Target size={13} />
                  clique para ver a votação num ponto
                </span>
              }
            >
              {regioes.isPending ? (
                <Carregando texto="Calculando as regiões…" />
              ) : (
                <GraficoDecisao
                  amostras={d.amostras}
                  limites={
                    regioes.data?.limites ?? {
                      x_min: 0,
                      x_max: 8,
                      y_min: 0,
                      y_max: 3,
                    }
                  }
                  eixoX={d.eixo_x}
                  eixoY={d.eixo_y}
                  grade={regioes.data?.grade}
                  classesGrade={regioes.data?.classes}
                  destaque={
                    consulta
                      ? { ...consulta, classe: predicao.data?.classe }
                      : null
                  }
                  onClicar={(x, y) => setConsulta({ x, y })}
                  altura={440}
                />
              )}
              <Nota tom="info" className="mt-3" titulo="Fronteiras em escada">
                Diferente de todos os classificadores anteriores, aqui as
                fronteiras são <strong>escadas alinhadas aos eixos</strong>:
                cada divisão de uma árvore é um corte do tipo{' '}
                <span className="font-mono">atributo ≤ limiar</span>, sempre
                paralelo a um eixo. O ensemble suaviza os degraus, mas a
                geometria continua sendo de retângulos.
              </Nota>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="importância dos atributos">
                <div className="space-y-3">
                  {d.importancias.map((imp) => (
                    <div key={imp.indice}>
                      <div className="mb-1 flex justify-between text-sm">
                        <span className="text-secondary">{imp.nome}</span>
                        <span className="tabular font-semibold text-primary">
                          {pct(imp.importancia)}
                        </span>
                      </div>
                      <div className="h-2.5 overflow-hidden rounded-full bg-sunken">
                        <div
                          className="h-full rounded-full bg-emerald-500"
                          style={{ width: `${imp.importancia * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <BlocoFormula
                  className="mt-4"
                  latex={String.raw`\text{imp}(j) \propto \sum_{\text{nós de } j} n_{\text{nó}} \cdot \text{ganho}`}
                  explicacao="Soma do ganho de impureza que o atributo proporcionou, ponderado pelo número de amostras no nó, sobre todas as árvores."
                />
              </Card>

              <Card titulo="a floresta em números">
                <div className="space-y-2 text-sm">
                  <Linha
                    rotulo="Árvores"
                    valor={String(d.arvores.length)}
                  />
                  <Linha
                    rotulo="Atributos sorteados por nó"
                    valor={`${d.config.n_atributos_por_no} de ${d.dimensoes}`}
                  />
                  <Linha
                    rotulo="Profundidade média"
                    valor={num(
                      d.arvores.reduce((s, a) => s + a.profundidade, 0) /
                        d.arvores.length,
                      1,
                    )}
                  />
                  <Linha
                    rotulo="Folhas por árvore (média)"
                    valor={num(
                      d.arvores.reduce((s, a) => s + a.folhas, 0) /
                        d.arvores.length,
                      1,
                    )}
                  />
                  <Linha
                    rotulo="Amostras únicas no bag"
                    valor={`${pct(
                      d.arvores.reduce((s, a) => s + a.amostras_unicas_bag, 0) /
                        d.arvores.length /
                        d.n_treino,
                      1,
                    )} (teórico 63,2%)`}
                  />
                  <Linha
                    rotulo="Amostras out-of-bag"
                    valor={`${pct(
                      d.arvores.reduce((s, a) => s + a.amostras_oob, 0) /
                        d.arvores.length /
                        d.n_treino,
                      1,
                    )} (teórico 36,8%)`}
                  />
                </div>
                <Nota tom="ok" className="mt-4" titulo="Por que 63,2%?">
                  Cada amostra tem probabilidade{' '}
                  <span className="font-mono">(1 − 1/n)ⁿ → e⁻¹ ≈ 0,368</span> de
                  ficar de fora do sorteio com reposição. As que ficam de fora
                  estimam o erro de generalização de graça.
                </Nota>
              </Card>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="matriz de confusão">
                <MatrizConfusao
                  relatorio={d.relatorio}
                  classes={classesDoRelatorio(d.relatorio)}
                />
              </Card>
              <Card titulo="métricas globais">
                <ResumoGlobal relatorio={d.relatorio} />
              </Card>
            </div>

            <Card titulo="métricas por classe">
              <TabelaPorClasse
                relatorio={d.relatorio}
                classes={classesDoRelatorio(d.relatorio)}
              />
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

function Linha({ rotulo, valor }: { rotulo: string; valor: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-subtle/60 pb-1.5 last:border-0">
      <span className="text-secondary">{rotulo}</span>
      <span className="tabular font-medium text-primary">{valor}</span>
    </div>
  )
}

/* ------------------------------------------------------ arvores individuais */
function PainelArvores({
  params,
  controles,
  criterio,
}: {
  params: ParamsFloresta
  controles: React.ReactNode
  criterio: string
}) {
  const [indice, setIndice] = useState(0)

  const q = useQuery({
    queryKey: ['floresta', 'arvore', indice, params],
    queryFn: () => api.floresta.arvore(indice, params),
  })

  const d = q.data

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        {controles}
        <Card titulo="escolher a árvore">
          <Slider
            rotulo="Índice da árvore"
            valor={indice}
            onChange={setIndice}
            min={0}
            max={Math.max(0, (d?.total_arvores ?? 50) - 1)}
            passo={1}
            formatar={(v) => `nº ${v + 1}`}
          />
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Percorra as árvores e veja como cada uma difere: elas treinaram em
            amostras diferentes (bootstrap) e consideraram atributos
            diferentes em cada nó.
          </p>
          <Nota tom="atencao" className="mt-3">
            Para o diagrama caber na tela, use uma{' '}
            <strong>profundidade máxima</strong> de 3 ou 4 no painel acima.
          </Nota>
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando texto="Montando a árvore…" />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <div className="grid gap-3 sm:grid-cols-4">
              <Metrica rotulo="Profundidade" valor={d.profundidade} />
              <Metrica rotulo="Nós" valor={d.nos} />
              <Metrica rotulo="Folhas" valor={d.folhas} />
              <Metrica
                rotulo="Amostras OOB"
                valor={d.amostras_oob}
                detalhe={`${d.amostras_unicas_bag} únicas no bag`}
              />
            </div>

            <Card titulo={`árvore nº ${d.indice + 1} de ${d.total_arvores}`}>
              <ArvoreDecisao
                arvore={d.arvore}
                nomesFeatures={d.nomes_features}
                criterio={criterio}
              />
            </Card>

            <Card titulo="como ler a árvore">
              <p className="text-sm leading-relaxed text-secondary">
                Cada nó testa uma condição do tipo{' '}
                <span className="font-mono">atributo ≤ limiar</span>. Se for
                verdadeira, a amostra desce pela esquerda; senão, pela direita.
                Ao chegar numa folha, a classe majoritária daquela folha é o
                voto desta árvore.
              </p>
              <BlocoFormula
                className="mt-3"
                titulo="ganho de impureza"
                latex={String.raw`\text{ganho} = I(\text{pai}) - \left[\frac{n_{esq}}{n}I(\text{esq}) + \frac{n_{dir}}{n}I(\text{dir})\right]`}
                explicacao="A divisão escolhida em cada nó é a que maximiza esse ganho, entre os atributos sorteados para aquele nó."
              />
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

/* --------------------------------------------------------------- comparativo */
function PainelComparativo({
  params,
  controles,
}: {
  params: ParamsFloresta
  controles: React.ReactNode
}) {
  const [k, setK] = useState(5)
  const [repeticoes, setRepeticoes] = useState(3)

  const q = useQuery({
    queryKey: ['floresta', 'validacao', params, k, repeticoes],
    queryFn: () => api.floresta.validacaoCruzada({ ...params, k, repeticoes }),
  })

  const d = q.data
  const ordenados = d
    ? Object.entries(d.resultados).sort((a, b) => b[1].media - a[1].media)
    : []
  const melhor = ordenados[0]?.[1].media ?? 1

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        {controles}
        <Card titulo="validação cruzada">
          <div className="space-y-4">
            <Slider
              rotulo="Dobras (k)"
              valor={k}
              onChange={setK}
              min={2}
              max={10}
              passo={1}
            />
            <Slider
              rotulo="Repetições"
              valor={repeticoes}
              onChange={setRepeticoes}
              min={1}
              max={10}
              passo={1}
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Avaliação honesta: toda amostra é testada, e o desvio mostra o
            quanto o número é confiável.
          </p>
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando texto="Rodando a validação cruzada…" />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <Card titulo="floresta × árvore única × demais classificadores">
              <div className="space-y-4">
                {ordenados.map(([chave, r]) => (
                  <div key={chave}>
                    <div className="mb-1 flex items-baseline justify-between gap-3">
                      <span className="flex items-center gap-2 text-sm font-medium text-primary">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{
                            backgroundColor: CORES_MODELO[chave] ?? '#16a34a',
                          }}
                        />
                        {r.nome}
                        {chave === 'floresta' && (
                          <Badge tom="bom">seminário</Badge>
                        )}
                      </span>
                      <span className="tabular text-sm">
                        <span className="font-semibold text-primary">
                          {pct(r.media)}
                        </span>
                        <span className="text-muted"> ± {pct(r.desvio)}</span>
                      </span>
                    </div>
                    <div className="relative h-5 overflow-hidden rounded bg-sunken">
                      <div
                        className="absolute inset-y-0 left-0 opacity-25"
                        style={{
                          width: `${(r.media / melhor) * 100}%`,
                          backgroundColor: CORES_MODELO[chave] ?? '#16a34a',
                        }}
                      />
                      <div
                        className="absolute inset-y-1 rounded-sm"
                        style={{
                          left: `${(r.ic_baixo / melhor) * 100}%`,
                          width: `${((r.ic_alto - r.ic_baixo) / melhor) * 100}%`,
                          backgroundColor: CORES_MODELO[chave] ?? '#16a34a',
                        }}
                      />
                    </div>
                    <div className="mt-0.5 flex justify-between text-[10px] tabular text-muted">
                      <span>
                        IC 95%: [{pct(r.ic_baixo, 1)}, {pct(r.ic_alto, 1)}]
                      </span>
                      <span>
                        min {pct(r.minimo, 1)} · max {pct(r.maximo, 1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs text-muted">
                {d.config.n_amostras} amostras · {d.config.k} dobras ×{' '}
                {d.config.repeticoes} repetições · floresta com{' '}
                {d.config.n_arvores} árvores ({d.config.criterio}).
              </p>
            </Card>

            <Card titulo="teste Z de significância">
              <TabelaTestesZ comparacoes={d.comparacoes} />
            </Card>

            <Card titulo="o que o ensemble compra">
              <Nota tom="ok" titulo="Floresta × árvore única">
                A diferença entre as duas linhas é exatamente o que o ensemble
                acrescenta. Uma árvore sozinha tem variância alta — muda muito
                conforme os dados de treino. Combinando muitas árvores
                descorrelacionadas, a variância cai sem aumentar o viés.
              </Nota>
              <Nota tom="info" className="mt-3" titulo="Quando vale a pena">
                Nas pétalas o problema é fácil demais e quase todos empatam. Nas{' '}
                <strong>sépalas</strong> — onde as classes se sobrepõem — a
                vantagem do ensemble sobre a árvore única costuma aparecer com
                mais clareza. Vale trocar os atributos no painel ao lado
                durante a apresentação.
              </Nota>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
