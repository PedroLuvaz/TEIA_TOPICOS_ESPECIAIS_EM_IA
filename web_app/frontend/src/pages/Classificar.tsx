/**
 * Classificar — escolha do modelo e parametrizacao.
 *
 * E a tela que responde ao enunciado da entrega: o usuario escolhe a base
 * (inclusive um .txt proprio), escolhe o MODELO e ajusta os HIPERPARAMETROS,
 * roda a classificacao e ve todas as metricas de qualidade.
 *
 * Os controles de parametro nao sao escritos a mao: o backend descreve cada
 * hiperparametro em `/api/classificar/modelos` (tipo, faixa, padrao, ajuda) e
 * esta pagina monta o controle correspondente. Acrescentar um modelo novo no
 * catalogo faz a interface crescer sozinha.
 */
import { useQuery } from '@tanstack/react-query'
import { Crosshair, Play, RotateCcw, Timer } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { PainelConfig, usarConfig, usarDataset } from '@/components/Controles'
import { GraficoDecisao } from '@/components/GraficoDecisao'
import { GraficoLinha } from '@/components/GraficoLinha'
import {
  MatrizConfusao,
  ResumoGlobal,
  TabelaPorClasse,
} from '@/components/Metricas'
import {
  Badge,
  Botao,
  Card,
  Carregando,
  ErroBox,
  Metrica,
  Nota,
  Select,
  Slider,
} from '@/components/ui'
import { api } from '@/lib/api'
import type {
  ExtrasModelo,
  ModeloCatalogo,
  ParametroModelo,
  RespostaClassificacao,
  ValorParametro,
} from '@/lib/types'
import { cap, corDaClasse, num, pct } from '@/lib/utils'

export function PaginaClassificar() {
  const { config, set } = usarConfig()
  const [modelo, setModelo] = useState('distancia_minima')
  const [params, setParams] = useState<Record<string, ValorParametro>>({})
  const [valores, setValores] = useState<number[] | null>(null)

  const catalogo = useQuery({
    queryKey: ['classificar', 'modelos'],
    queryFn: api.classificar.modelos,
    staleTime: Infinity,
  })

  const corpo = { ...config, modelo, parametros: params }

  const treino = useQuery({
    queryKey: ['classificar', 'treinar', corpo],
    queryFn: () => api.classificar.treinar(corpo),
  })

  const regioes = useQuery({
    queryKey: ['classificar', 'regioes', corpo],
    queryFn: () => api.classificar.regioes({ ...corpo, resolucao: 90 }),
  })

  const predicao = useQuery({
    queryKey: ['classificar', 'predizer', corpo, valores],
    queryFn: () => api.classificar.predizer({ ...corpo, valores: valores! }),
    enabled: !!valores && valores.length > 0,
  })

  const d = treino.data
  const cfgModelo: ModeloCatalogo | undefined = catalogo.data?.modelos.find(
    (m) => m.id === modelo,
  )

  // Ao trocar de modelo os parametros do anterior nao fazem sentido: zera-se o
  // dicionario e o backend aplica os padroes do modelo novo.
  function trocarModelo(id: string) {
    setModelo(id)
    setParams({})
  }

  // Assim que o treino chega, o formulario de predicao ganha valores iniciais
  // sensatos (a media de cada atributo no conjunto completo).
  const medias = d?.medias
  useEffect(() => {
    if (medias && (!valores || valores.length !== medias.length)) {
      setValores(medias.map((v) => Number(v.toFixed(2))))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [medias, config.dataset, config.atributos])

  return (
    <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
      {/* --------------------------------------------------------- controles */}
      <div className="space-y-5">
        <PainelConfig config={config} set={set} />

        <Card
          titulo="modelo de classificação"
          acao={cfgModelo && <Badge tom="info">{cfgModelo.grupo}</Badge>}
        >
          <div className="space-y-3">
            <Select
              rotulo="Modelo"
              valor={modelo}
              onChange={trocarModelo}
              opcoes={
                catalogo.data?.modelos.map((m) => ({
                  valor: m.id,
                  rotulo: `${m.nome}  ·  ${m.grupo}`,
                })) ?? [{ valor: modelo, rotulo: 'carregando…' }]
              }
            />
            {cfgModelo && (
              <p className="text-xs leading-relaxed text-secondary">
                {cfgModelo.descricao}
              </p>
            )}
          </div>
        </Card>

        <Card
          titulo="parametrização"
          acao={
            !!cfgModelo?.parametros.length && (
              <Botao
                variante="fantasma"
                tamanho="sm"
                onClick={() => setParams({})}
              >
                <RotateCcw size={13} />
                padrões
              </Botao>
            )
          }
        >
          {cfgModelo?.parametros.length ? (
            <div className="space-y-4">
              {cfgModelo.parametros.map((p) => (
                <ControleParametro
                  key={p.id}
                  parametro={p}
                  valor={params[p.id] ?? p.padrao}
                  onChange={(v) => setParams((a) => ({ ...a, [p.id]: v }))}
                />
              ))}
            </div>
          ) : (
            <p className="text-sm leading-relaxed text-secondary">
              Este modelo não tem hiperparâmetros: ele é determinado
              inteiramente pelos dados de treino. O que se pode variar são a
              base, os atributos e a proporção treino/teste, no painel acima.
            </p>
          )}
        </Card>

        {d && (
          <FormularioPredicao
            dataset={config.dataset}
            dados={d}
            valores={valores ?? d.medias}
            onChange={setValores}
            predicao={predicao.data}
            carregando={predicao.isPending && !!valores}
          />
        )}
      </div>

      {/* -------------------------------------------------------- resultados */}
      <div className="space-y-6">
        {treino.isPending && (
          <Card>
            <Carregando texto="Treinando o modelo…" />
          </Card>
        )}
        {treino.error && <ErroBox erro={treino.error} />}

        {d && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metrica
                rotulo="Acerto no teste"
                valor={pct(d.relatorio.acerto_global)}
                detalhe={`${d.n_teste} amostras nunca vistas`}
                destaque={d.relatorio.acerto_global >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica
                rotulo="Acerto no treino"
                valor={pct(d.acerto_treino)}
                detalhe={
                  d.acerto_treino - d.relatorio.acerto_global > 0.15
                    ? 'diferença alta: sobreajuste'
                    : `${d.n_treino} amostras`
                }
                destaque={
                  d.acerto_treino - d.relatorio.acerto_global > 0.15
                    ? 'medio'
                    : undefined
                }
              />
              <Metrica
                rotulo="Kappa"
                valor={num(d.relatorio.kappa, 4)}
                detalhe="concordância corrigida"
              />
              <Metrica
                rotulo="Predição"
                valor={
                  <span className="inline-flex items-center gap-1.5">
                    <Timer size={15} className="text-muted" />
                    {num(d.ms_predicao, 1)} ms
                  </span>
                }
                detalhe={`${d.dimensoes} atributo(s) em uso`}
              />
            </div>

            <Card
              titulo={`regiões de decisão · ${d.modelo.nome}`}
              acao={
                <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                  <Crosshair size={13} />
                  clique para classificar um ponto
                </span>
              }
            >
              {regioes.isPending ? (
                <Carregando texto="Varrendo o plano com o modelo…" />
              ) : (
                <GraficoDecisao
                  amostras={d.amostras}
                  limites={
                    regioes.data?.limites ?? {
                      x_min: 0,
                      x_max: 1,
                      y_min: 0,
                      y_max: 1,
                    }
                  }
                  eixoX={d.eixo_x}
                  eixoY={d.eixo_y}
                  grade={regioes.data?.grade}
                  classesGrade={regioes.data?.classes}
                  destaque={
                    valores && predicao.data
                      ? {
                          x: valores[d.indices.indexOf(d.indices_plot[0])] ?? 0,
                          y: valores[d.indices.indexOf(d.indices_plot[1])] ?? 0,
                          classe: predicao.data.classe,
                        }
                      : null
                  }
                  onClicar={(x, y) => {
                    // O clique altera so as duas dimensoes desenhadas; as
                    // demais continuam com o valor do formulario.
                    const base = [...(valores ?? d.medias)]
                    const px = d.indices.indexOf(d.indices_plot[0])
                    const py = d.indices.indexOf(d.indices_plot[1])
                    if (px >= 0) base[px] = Number(x.toFixed(3))
                    if (py >= 0) base[py] = Number(y.toFixed(3))
                    setValores(base)
                  }}
                  altura={430}
                />
              )}
              {d.dimensoes > 2 && (
                <Nota tom="info" titulo="Projeção em 2D">
                  O modelo usa {d.dimensoes} atributos; o gráfico desenha o
                  plano de {d.eixo_x} × {d.eixo_y} com os demais fixos na média
                  global.
                </Nota>
              )}
            </Card>

            <Card titulo="métricas de qualidade">
              <ResumoGlobal relatorio={d.relatorio} />
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="matriz de confusão">
                <MatrizConfusao relatorio={d.relatorio} classes={d.classes} />
              </Card>
              <Card titulo="métricas por classe">
                <TabelaPorClasse relatorio={d.relatorio} classes={d.classes} />
              </Card>
            </div>

            <PainelExtras
              extras={d.extras}
              modelo={d.modelo.id}
              features={d.features}
              indices={d.indices}
            />
          </>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------ controle generico */
function ControleParametro({
  parametro,
  valor,
  onChange,
}: {
  parametro: ParametroModelo
  valor: ValorParametro
  onChange: (v: ValorParametro) => void
}) {
  if (parametro.tipo === 'opcoes') {
    return (
      <div>
        <Select
          rotulo={parametro.rotulo}
          valor={String(valor)}
          onChange={onChange}
          opcoes={(parametro.opcoes ?? []).map((o) => ({
            valor: o.valor,
            rotulo: o.rotulo,
          }))}
        />
        {parametro.ajuda && (
          <p className="mt-1 text-xs text-muted">{parametro.ajuda}</p>
        )}
      </div>
    )
  }

  const casas = parametro.tipo === 'inteiro' ? 0 : 3
  return (
    <div>
      <Slider
        rotulo={parametro.rotulo}
        valor={Number(valor)}
        onChange={onChange}
        min={parametro.min ?? 0}
        max={parametro.max ?? 100}
        passo={parametro.passo ?? 1}
        formatar={(v) =>
          // 0 significa "sem limite" nos parametros que aceitam essa leitura.
          parametro.id === 'profundidade_max' && v === 0
            ? 'sem limite'
            : num(v, casas)
        }
      />
      {parametro.ajuda && (
        <p className="mt-1 text-xs text-muted">{parametro.ajuda}</p>
      )}
    </div>
  )
}

/* --------------------------------------------------- predicao de amostra */
function FormularioPredicao({
  dataset,
  dados,
  valores,
  onChange,
  predicao,
  carregando,
}: {
  dataset: string
  dados: RespostaClassificacao
  valores: number[]
  onChange: (v: number[]) => void
  predicao?: { classe: string; scores: Record<string, number> }
  carregando: boolean
}) {
  // Bases categoricas guardam codigos 0..k-1; aqui mostramos o rotulo original
  // ('Sol', 'Chuva') ao lado do numero digitado.
  const { rotuloValor, categorico } = usarDataset(dataset)

  return (
    <Card titulo="classificar uma amostra">
      <div className="space-y-3">
        {dados.features.map((nome, k) => (
          <label key={nome} className="block">
            <span className="mb-1 flex items-baseline justify-between">
              <span className="kicker text-muted">{nome}</span>
              {categorico && (
                <span className="text-xs text-secondary">
                  {rotuloValor(dados.indices[k], valores[k] ?? 0)}
                </span>
              )}
            </span>
            <input
              type="number"
              step="0.1"
              value={valores[k] ?? 0}
              onChange={(e) => {
                const copia = [...valores]
                copia[k] = Number(e.target.value)
                onChange(copia)
              }}
              className="tabular w-full rounded-lg border border-strong bg-surface px-3 py-1.5 text-sm text-primary hover:border-accent-500/50 focus:border-accent-500 focus:outline-none"
            />
          </label>
        ))}

        {carregando && <Carregando texto="Classificando…" />}

        {predicao && !carregando && (
          <div className="space-y-2 border-t border-subtle pt-3">
            <div
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold text-white"
              style={{ backgroundColor: corDaClasse(predicao.classe) }}
            >
              <Play size={14} />
              {cap(predicao.classe)}
            </div>
            <p className="kicker text-muted">{dados.modelo.rotulo_score}</p>
            {Object.entries(predicao.scores).map(([c, s]) => (
              <div key={c} className="flex justify-between text-xs">
                <span className="text-secondary">{cap(c)}</span>
                <span className="tabular text-primary">{num(s, 4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  )
}

/* --------------------------------------------------- extras por modelo --- */
function PainelExtras({
  extras,
  modelo,
  features,
  indices,
}: {
  extras: ExtrasModelo
  modelo: string
  features: string[]
  indices: number[]
}) {
  const nomePorIndice = useMemo(() => {
    const mapa: Record<number, string> = {}
    indices.forEach((ind, k) => {
      mapa[ind] = features[k]
    })
    return mapa
  }, [indices, features])

  if (modelo === 'floresta' && extras.importancias) {
    const maior = Math.max(...extras.importancias.map((i) => i.importancia), 1e-9)
    return (
      <div className="grid gap-6 lg:grid-cols-2">
        <Card titulo="erro out-of-bag">
          <div className="space-y-3">
            <Metrica
              rotulo="Acurácia OOB"
              valor={extras.oob?.acuracia != null ? pct(extras.oob.acuracia) : '—'}
              detalhe={`${extras.arvores} árvores · estimativa sem conjunto de validação`}
            />
            <p className="text-xs leading-relaxed text-muted">
              Cada árvore é treinada num bootstrap; as amostras que ficaram de
              fora dela servem para testá-la. A média desses testes é a
              estimativa OOB — validação de graça, embutida no bagging.
            </p>
          </div>
        </Card>
        <Card titulo="importância dos atributos">
          <div className="space-y-2">
            {extras.importancias.map((imp) => (
              <div key={imp.indice}>
                <div className="flex justify-between text-xs">
                  <span className="text-secondary">
                    {nomePorIndice[imp.indice] ?? `Atributo ${imp.indice}`}
                  </span>
                  <span className="tabular text-primary">
                    {num(imp.importancia, 4)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-500/15">
                  <div
                    className="h-full rounded-full bg-accent-500"
                    style={{ width: `${(imp.importancia / maior) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  if (modelo === 'mlp' && extras.curva_erro?.length) {
    return (
      <Card titulo="convergência do backpropagation">
        <GraficoLinha
          dados={extras.curva_erro.map((p) => ({ x: p.epoca, erro: p.erro }))}
          series={[{ chave: 'erro', rotulo: 'Erro médio', cor: '#f59e0b' }]}
          rotuloX="Época"
          rotuloY="Erro quadrático médio"
          altura={260}
        />
        <p className="mt-2 text-xs text-muted">
          Erro final: {num(extras.erro_final ?? 0, 6)}. A curva descendo e
          estabilizando indica que a rede aprendeu; oscilação forte costuma ser
          taxa de aprendizado alta demais.
        </p>
      </Card>
    )
  }

  if ((modelo === 'perceptron_ova' || modelo === 'delta_ova') && extras.curvas) {
    const classes = Object.keys(extras.curvas)
    const maxEpocas = Math.max(...classes.map((c) => extras.curvas![c].length))
    const dados = Array.from({ length: maxEpocas }, (_, i) => {
      const ponto: Record<string, number> = { x: i + 1 }
      for (const c of classes) {
        const v = extras.curvas![c][i]
        if (v !== undefined) ponto[c] = v
      }
      return ponto
    })
    return (
      <Card titulo="convergência por classe (Um-Contra-Todos)">
        <GraficoLinha
          dados={dados}
          series={classes.map((c) => ({
            chave: c,
            rotulo: cap(c),
            cor: corDaClasse(c),
          }))}
          rotuloX="Época"
          rotuloY={
            modelo === 'perceptron_ova'
              ? 'Erros de classificação'
              : 'Erro quadrático médio'
          }
          altura={260}
        />
        {modelo === 'perceptron_ova' && extras.convergiu && (
          <div className="mt-3 flex flex-wrap gap-2">
            {classes.map((c) => (
              <Badge key={c} tom={extras.convergiu![c] ? 'bom' : 'ruim'}>
                {cap(c)}:{' '}
                {extras.convergiu![c]
                  ? `convergiu em ${extras.epocas_por_classe?.[c]} épocas`
                  : 'não convergiu (classes não separáveis)'}
              </Badge>
            ))}
          </div>
        )}
      </Card>
    )
  }

  return null
}
