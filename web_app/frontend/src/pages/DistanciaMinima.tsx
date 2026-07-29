/** Lab 1 — Classificador de Distancia Minima. */
import { useQuery } from '@tanstack/react-query'
import { Crosshair, Target } from 'lucide-react'
import { useState } from 'react'
import { PainelConfig, usarConfig } from '@/components/Controles'
import { BlocoFormula } from '@/components/Formula'
import { GraficoDecisao } from '@/components/GraficoDecisao'
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
} from '@/components/ui'
import { api } from '@/lib/api'
import { cap, corDaClasse, num, pct } from '@/lib/utils'

export function PaginaDistanciaMinima() {
  const { config, set } = usarConfig()
  const [consulta, setConsulta] = useState<{ x: number; y: number } | null>(null)

  const treino = useQuery({
    queryKey: ['dm', 'treinar', config],
    queryFn: () => api.distanciaMinima.treinar(config),
  })

  const regioes = useQuery({
    queryKey: ['dm', 'regioes', config],
    queryFn: () => api.distanciaMinima.regioes({ ...config, resolucao: 110 }),
  })

  const predicao = useQuery({
    queryKey: ['dm', 'predizer', config, consulta],
    queryFn: () =>
      api.distanciaMinima.predizer({
        dataset: config.dataset,
        atributos: config.atributos,
        proporcao: config.proporcao,
        valores:
          config.atributos === 'todas'
            ? [5.8, 3.0, consulta!.x, consulta!.y]
            : [consulta!.x, consulta!.y],
      }),
    enabled: !!consulta,
  })

  const d = treino.data

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      {/* ------------------------------------------------------- controles */}
      <div className="space-y-5">
        <PainelConfig config={config} set={set} />

        <Card titulo="sobre este laboratório">
          <p className="text-sm leading-relaxed text-secondary">
            O classificador calcula o <strong>protótipo</strong> (vetor médio)
            de cada classe e atribui cada amostra à classe cujo protótipo está
            mais próximo — equivalente a maximizar a função discriminante
            linear.
          </p>
          <BlocoFormula
            className="mt-3"
            titulo="protótipo"
            latex={String.raw`m_j = \frac{1}{N_j}\sum_{x \in \omega_j} x`}
          />
          <BlocoFormula
            className="mt-2"
            titulo="função discriminante"
            latex={String.raw`d_j(x) = x^{T} m_j - \tfrac{1}{2}\, m_j^{T} m_j`}
            explicacao="A decisão é argmax_j d_j(x), matematicamente equivalente a argmin_j ‖x − m_j‖."
          />
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
                <div className="flex items-center gap-2">
                  <Crosshair size={15} className="text-muted" />
                  <span className="font-mono text-sm text-secondary">
                    ({num(consulta.x, 2)}, {num(consulta.y, 2)})
                  </span>
                </div>
                <div
                  className="rounded-lg px-3 py-2 text-sm font-semibold text-white"
                  style={{ backgroundColor: corDaClasse(predicao.data.classe) }}
                >
                  {cap(predicao.data.classe)}
                </div>
                <div className="space-y-1.5">
                  {Object.entries(predicao.data.scores).map(([c, s]) => (
                    <div key={c} className="flex justify-between text-xs">
                      <span className="text-secondary">
                        d<sub>{c.slice(0, 3)}</sub>(x)
                      </span>
                      <span className="tabular text-primary">{num(s, 4)}</span>
                    </div>
                  ))}
                </div>
                {predicao.data.distancias && (
                  <div className="space-y-1.5 border-t border-subtle pt-2">
                    {Object.entries(predicao.data.distancias).map(([c, s]) => (
                      <div key={c} className="flex justify-between text-xs">
                        <span className="text-muted">‖x − m_{c.slice(0, 3)}‖</span>
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

      {/* -------------------------------------------------------- resultados */}
      <div className="space-y-6">
        {treino.isPending && (
          <Card>
            <Carregando />
          </Card>
        )}
        {treino.error && <ErroBox erro={treino.error} />}

        {d && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Metrica
                rotulo="Acerto Global"
                valor={pct(d.relatorio.acerto_global)}
                detalhe={`${d.n_teste} amostras de teste`}
                destaque={d.relatorio.acerto_global >= 0.9 ? 'bom' : 'medio'}
              />
              <Metrica
                rotulo="Kappa"
                valor={num(d.relatorio.kappa, 4)}
                detalhe="concordância corrigida"
              />
              <Metrica rotulo="Treino" valor={d.n_treino} detalhe="amostras" />
              <Metrica
                rotulo="Dimensões"
                valor={d.dimensoes}
                detalhe={d.dimensoes === 4 ? 'projetado nas pétalas' : 'plano 2D'}
              />
            </div>

            <Card
              titulo="regiões de decisão"
              acao={
                <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                  <Target size={13} />
                  clique no gráfico para classificar um ponto
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
                  marcadores={Object.entries(d.prototipos_plot).map(([c, p]) => ({
                    x: p.x,
                    y: p.y,
                    classe: c,
                    rotulo: `Protótipo ${cap(c)}`,
                  }))}
                  destaque={
                    consulta
                      ? { ...consulta, classe: predicao.data?.classe }
                      : null
                  }
                  onClicar={(x, y) => setConsulta({ x, y })}
                  altura={440}
                />
              )}
              {d.dimensoes === 4 && (
                <Nota tom="info" titulo="Projeção 4D → 2D">
                  Com as 4 features, o gráfico projeta o plano das pétalas; as
                  sépalas ficam fixas na média global. A classificação usa as 4
                  dimensões.
                </Nota>
              )}
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="protótipos (vetores médios)">
                <div className="space-y-3">
                  {Object.entries(d.prototipos).map(([classe, vetor]) => (
                    <div
                      key={classe}
                      className="flex items-center gap-3 rounded-lg border border-subtle bg-sunken px-3 py-2.5"
                    >
                      <span
                        className="h-8 w-1 shrink-0 rounded-full"
                        style={{ backgroundColor: corDaClasse(classe) }}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-primary">
                          {cap(classe)}
                        </p>
                        <p className="truncate font-mono text-xs text-muted">
                          m = [{vetor.map((v) => num(v, 4)).join(', ')}]
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              <Card titulo="fronteiras de decisão">
                <div className="space-y-3">
                  {d.fronteiras.map((f, i) => (
                    <div key={i} className="rounded-lg border border-subtle bg-sunken p-3">
                      <div className="mb-1.5 flex items-center gap-2">
                        <Badge tom="neutro">
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{ backgroundColor: corDaClasse(f.classe_i) }}
                          />
                          {cap(f.classe_i)}
                        </Badge>
                        <span className="text-xs text-muted">×</span>
                        <Badge tom="neutro">
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{ backgroundColor: corDaClasse(f.classe_j) }}
                          />
                          {cap(f.classe_j)}
                        </Badge>
                      </div>
                      <p className="break-all font-mono text-xs text-secondary">
                        {f.equacao}
                      </p>
                    </div>
                  ))}
                  <p className="text-xs leading-relaxed text-muted">
                    Coeficientes: <span className="font-mono">w = mᵢ − mⱼ</span> e{' '}
                    <span className="font-mono">b = −½(‖mᵢ‖² − ‖mⱼ‖²)</span>.
                  </p>
                </div>
              </Card>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="matriz de confusão">
                <MatrizConfusao
                  relatorio={d.relatorio}
                  classes={['setosa', 'versicolor', 'virginica']}
                />
              </Card>
              <Card titulo="métricas globais">
                <ResumoGlobal relatorio={d.relatorio} />
              </Card>
            </div>

            <Card titulo="métricas por classe">
              <TabelaPorClasse
                relatorio={d.relatorio}
                classes={['setosa', 'versicolor', 'virginica']}
              />
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
