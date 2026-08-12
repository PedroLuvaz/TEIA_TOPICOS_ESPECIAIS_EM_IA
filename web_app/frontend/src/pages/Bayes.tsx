/** Lab 4 — Bayes Otimo (QDA), Naive Bayes e normalidade multivariada. */
import { useQuery } from '@tanstack/react-query'
import { FileText, Target } from 'lucide-react'
import { useState } from 'react'
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
} from '@/components/ui'
import { api } from '@/lib/api'
import {
  cap,
  classesDoRelatorio,
  corDaClasse,
  num,
  pct,
  sci,
} from '@/lib/utils'

export function PaginaBayes() {
  const { config, set } = usarConfig()
  const [classificador, setClassificador] = useState<'bayes' | 'naive'>('bayes')
  const [consulta, setConsulta] = useState<{ x: number; y: number } | null>(null)
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  const treino = useQuery({
    queryKey: ['bayes', 'treinar', config],
    queryFn: () => api.bayes.treinar(config),
  })

  const regioes = useQuery({
    queryKey: ['bayes', 'regioes', config, classificador],
    queryFn: () =>
      api.bayes.regioes({ ...config, classificador, resolucao: 110 }),
  })

  const normalidade = useQuery({
    queryKey: ['bayes', 'normalidade', config.dataset, config.atributos],
    queryFn: () =>
      api.bayes.normalidade({
        dataset: config.dataset,
        atributos: config.atributos,
      }),
  })

  const predicao = useQuery({
    queryKey: ['bayes', 'predizer', config, classificador, consulta],
    queryFn: () =>
      api.bayes.predizer({
        dataset: config.dataset,
        atributos: config.atributos,
        naive: classificador === 'naive',
        valores:
          config.atributos === 'todas'
            ? [5.8, 3.0, consulta!.x, consulta!.y]
            : [consulta!.x, consulta!.y],
      }),
    enabled: !!consulta,
  })

  const memoria = useQuery({
    queryKey: ['bayes', 'memoria', config, classificador],
    queryFn: () => api.bayes.memoria({ ...config, classificador }),
    enabled: memoriaAberta,
  })

  const d = treino.data
  const atual = d?.[classificador]
  const { classes } = usarDataset(config.dataset)

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set} />

        <Card titulo="classificador">
          <Segmentos
            valor={classificador}
            onChange={setClassificador}
            opcoes={[
              { valor: 'bayes', rotulo: 'Bayes Ótimo' },
              { valor: 'naive', rotulo: 'Naive Bayes' },
            ]}
            className="w-full"
          />
          <p className="mt-3 text-sm leading-relaxed text-secondary">
            {classificador === 'bayes'
              ? 'Usa a matriz de covariância completa de cada classe (QDA) — as fronteiras são quadráticas: parábolas, elipses ou hipérboles.'
              : 'Assume independência entre as features, zerando a covariância fora da diagonal. Fronteiras mais simples, com menos parâmetros a estimar.'}
          </p>
          <BlocoFormula
            className="mt-3"
            titulo="discriminante"
            latex={String.raw`d_j(x) = -\tfrac{1}{2}\ln|\Sigma_j| - \tfrac{1}{2}(x-m_j)^{T}\Sigma_j^{-1}(x-m_j)`}
            explicacao="Máximo a posteriori com prioris iguais. O segundo termo é a distância de Mahalanobis ao quadrado."
          />
        </Card>

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
            Parâmetros estimados, discriminante quadrático passo a passo, por
            que as fronteiras são curvas e o teste Z entre Bayes e Naive.
          </p>
        </Card>

        {consulta && (
          <Card
            titulo="ponto consultado"
            acao={
              <Botao variante="fantasma" tamanho="sm" onClick={() => setConsulta(null)}>
                limpar
              </Botao>
            }
          >
            {predicao.isPending && <Carregando texto="Classificando…" />}
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
                <div className="space-y-1.5">
                  <p className="kicker text-muted">score discriminante</p>
                  {Object.entries(predicao.data.scores).map(([c, s]) => (
                    <div key={c} className="flex justify-between text-xs">
                      <span className="text-secondary">{cap(c)}</span>
                      <span className="tabular text-primary">{num(s, 4)}</span>
                    </div>
                  ))}
                </div>
                {predicao.data.mahalanobis && (
                  <div className="space-y-1.5 border-t border-subtle pt-2">
                    <p className="kicker text-muted">distância de Mahalanobis²</p>
                    {Object.entries(predicao.data.mahalanobis).map(([c, s]) => (
                      <div key={c} className="flex justify-between text-xs">
                        <span className="text-secondary">{cap(c)}</span>
                        <span className="tabular text-secondary">{num(s, 4)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Card>
        )}
      </div>

      <div className="space-y-6">
        {treino.isPending && (
          <Card>
            <Carregando />
          </Card>
        )}
        {treino.error && <ErroBox erro={treino.error} />}

        {d && atual && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metrica
                rotulo="Bayes Ótimo · Ag"
                valor={pct(d.bayes.relatorio.acerto_global)}
                detalhe={`κ = ${num(d.bayes.relatorio.kappa, 4)}`}
                cor="#0ea5e9"
                destaque={d.bayes.relatorio.acerto_global >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica
                rotulo="Naive Bayes · Ag"
                valor={pct(d.naive.relatorio.acerto_global)}
                detalhe={`κ = ${num(d.naive.relatorio.kappa, 4)}`}
                cor="#10b981"
                destaque={d.naive.relatorio.acerto_global >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica
                rotulo="Teste Z (Bayes × Naive)"
                valor={num(d.teste_z.z, 4)}
                detalhe={`p = ${num(d.teste_z.p, 4)}`}
                destaque={d.teste_z.significativo ? 'ruim' : 'bom'}
              />
              <Metrica
                rotulo="Amostras"
                valor={`${d.n_treino} / ${d.n_teste}`}
                detalhe="treino / teste"
              />
            </div>

            <Card
              titulo={`regiões de decisão — ${classificador === 'bayes' ? 'Bayes Ótimo (QDA)' : 'Naive Bayes'}`}
              acao={
                <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                  <Target size={13} />
                  clique para classificar um ponto
                </span>
              }
            >
              {regioes.isPending ? (
                <Carregando texto="Calculando as superfícies quadráticas…" />
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
                  superficies={regioes.data?.superficies}
                  destaque={
                    consulta ? { ...consulta, classe: predicao.data?.classe } : null
                  }
                  onClicar={(x, y) => setConsulta({ x, y })}
                  altura={440}
                />
              )}
              <Nota tom="info" className="mt-3" titulo="Fronteiras curvas">
                Diferente dos classificadores lineares dos labs anteriores, aqui
                as fronteiras são <strong>quadráticas</strong> — traçadas pelo
                nível zero da diferença de scores, o que garante curvas suaves
                em vez de escadinhas.
              </Nota>
            </Card>

            <Card titulo="parâmetros estimados por classe">
              <div className="grid gap-4 lg:grid-cols-3">
                {classes.map((c) => {
                  const p = atual.parametros[c]
                  if (!p) return null
                  return (
                    <div
                      key={c}
                      className="rounded-lg border border-subtle bg-sunken p-4"
                    >
                      <div className="mb-3 flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: corDaClasse(c) }}
                        />
                        <span className="text-sm font-semibold text-primary">
                          {cap(c)}
                        </span>
                      </div>
                      <p className="kicker mb-1 text-muted">vetor médio</p>
                      <p className="mb-3 break-all font-mono text-xs text-secondary">
                        [{p.media.map((v) => num(v, 4)).join(', ')}]
                      </p>
                      <p className="kicker mb-1 text-muted">
                        matriz de covariância
                      </p>
                      <div className="mb-3 space-y-0.5">
                        {p.cov.map((linha, i) => (
                          <p key={i} className="font-mono text-[11px] text-secondary">
                            [{linha.map((v) => num(v, 4)).join('  ')}]
                          </p>
                        ))}
                      </div>
                      <p className="font-mono text-[11px] text-muted">
                        |Σ| = {sci(p.det)}
                      </p>
                    </div>
                  )
                })}
              </div>
              {classificador === 'naive' && (
                <Nota tom="atencao" className="mt-4">
                  No Naive Bayes os termos fora da diagonal são forçados a zero —
                  compare com o Bayes Ótimo para ver o efeito dessa suposição de
                  independência.
                </Nota>
              )}
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="matriz de confusão">
                <MatrizConfusao
                  relatorio={atual.relatorio}
                  classes={classesDoRelatorio(atual.relatorio)}
                />
              </Card>
              <Card titulo="métricas globais">
                <ResumoGlobal relatorio={atual.relatorio} />
              </Card>
            </div>

            <Card titulo="métricas por classe">
              <TabelaPorClasse
                relatorio={atual.relatorio}
                classes={classesDoRelatorio(atual.relatorio)}
              />
            </Card>

            <Card titulo="Bayes Ótimo × Naive Bayes">
              <TabelaTestesZ
                comparacoes={[
                  {
                    nome_a: 'Bayes Ótimo (QDA)',
                    nome_b: 'Naive Bayes',
                    z: d.teste_z.z,
                    p: d.teste_z.p,
                    significativo: d.teste_z.significativo,
                  },
                ]}
              />
            </Card>

            <Card titulo="aderência à normalidade multivariada">
              {normalidade.isPending && <Carregando texto="Calculando MVN…" />}
              {normalidade.error && (
                <Nota tom="atencao">
                  Não foi possível calcular os testes de normalidade:{' '}
                  {String(
                    normalidade.error instanceof Error
                      ? normalidade.error.message
                      : normalidade.error,
                  )}
                </Nota>
              )}
              {normalidade.data && (
                <TabelaNormalidade resultado={normalidade.data.resultado} />
              )}
              <Nota tom="info" className="mt-4" titulo="Por que testar?">
                O Bayes Ótimo assume que cada classe segue uma distribuição
                normal multivariada. Os testes de{' '}
                <strong>Henze-Zirkler</strong> e <strong>Mardia</strong>{' '}
                verificam essa premissa — se ela falhar, o classificador ainda
                funciona, mas perde a garantia de otimalidade.
              </Nota>
            </Card>
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

/** Resultado dos testes de normalidade multivariada, por classe. */
interface ResultadoMvn {
  hz_stat: number
  hz_p: number
  hz_normal: string
  mardia_skew_stat: number
  mardia_skew_p: number
  mardia_skew_normal: string
  mardia_kurt_stat: number
  mardia_kurt_p: number
  mardia_kurt_normal: string
  veredicto: string
}

const TESTES_MVN = [
  { rotulo: 'Henze-Zirkler', stat: 'hz_stat', p: 'hz_p', ok: 'hz_normal' },
  {
    rotulo: 'Mardia · Assimetria',
    stat: 'mardia_skew_stat',
    p: 'mardia_skew_p',
    ok: 'mardia_skew_normal',
  },
  {
    rotulo: 'Mardia · Curtose',
    stat: 'mardia_kurt_stat',
    p: 'mardia_kurt_p',
    ok: 'mardia_kurt_normal',
  },
] as const

function TabelaNormalidade({ resultado }: { resultado: Record<string, unknown> }) {
  const classes = Object.entries(resultado).filter(
    ([, v]) => v && typeof v === 'object' && 'hz_p' in (v as object),
  ) as [string, ResultadoMvn][]

  if (!classes.length) {
    return (
      <pre className="overflow-x-auto rounded-lg bg-sunken p-3 font-mono text-xs text-secondary">
        {JSON.stringify(resultado, null, 2)}
      </pre>
    )
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-subtle">
              <th className="py-2 pr-3 text-left text-[11px] font-semibold text-muted">
                Classe
              </th>
              <th className="px-2 py-2 text-left text-[11px] font-semibold text-muted">
                Teste
              </th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                Estatística
              </th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                p-valor
              </th>
              <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                Normalidade (α = 5%)
              </th>
            </tr>
          </thead>
          <tbody>
            {classes.map(([classe, dados]) =>
              TESTES_MVN.map((teste, i) => {
                const normal = dados[teste.ok] === 'SIM'
                return (
                  <tr
                    key={`${classe}-${teste.rotulo}`}
                    className="border-b border-subtle/60 last:border-0"
                  >
                    <td className="py-2.5 pr-3">
                      {i === 0 && (
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: corDaClasse(classe) }}
                          />
                          <span className="font-medium text-primary">
                            {cap(classe)}
                          </span>
                        </span>
                      )}
                    </td>
                    <td className="px-2 py-2.5 text-secondary">{teste.rotulo}</td>
                    <td className="px-2 py-2.5 text-right tabular text-secondary">
                      {num(dados[teste.stat], 4)}
                    </td>
                    <td className="px-2 py-2.5 text-right tabular text-primary">
                      {num(dados[teste.p], 4)}
                    </td>
                    <td className="px-2 py-2.5 text-right">
                      <Badge tom={normal ? 'bom' : 'ruim'}>
                        {normal ? 'não rejeitada' : 'rejeitada'}
                      </Badge>
                    </td>
                  </tr>
                )
              }),
            )}
          </tbody>
        </table>
      </div>

      <div className="space-y-2">
        {classes.map(([classe, dados]) => (
          <div
            key={classe}
            className="rounded-lg border border-subtle bg-sunken px-3 py-2.5"
          >
            <p className="flex items-start gap-2 text-xs leading-relaxed text-secondary">
              <span
                className="mt-1 h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: corDaClasse(classe) }}
              />
              {dados.veredicto}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
