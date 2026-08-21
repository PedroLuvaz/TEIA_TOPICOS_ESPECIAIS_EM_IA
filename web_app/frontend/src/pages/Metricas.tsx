/** Lab 3 — Metricas avancadas: Kappa, Tau, teste Z e matriz editavel. */
import { useQuery } from '@tanstack/react-query'
import { FileText } from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  PainelConfig,
  usarConfig,
  usarDataset,
} from '@/components/Controles'
import { BlocoFormula } from '@/components/Formula'
import { GraficoLinha } from '@/components/GraficoLinha'
import { MemoriaGenerica } from '@/components/MemoriaGenerica'
import {
  MatrizConfusao,
  ResumoGlobal,
  TabelaPorClasse,
  TabelaTestesZ,
} from '@/components/Metricas'
import {
  Botao,
  Card,
  Carregando,
  ErroBox,
  Legenda,
  Metrica,
  Nota,
  Segmentos,
  Slider,
} from '@/components/ui'
import { api } from '@/lib/api'
import { CORES_MODELO, classesDoRelatorio, num, pct } from '@/lib/utils'
import { PaginaSignificancia } from './Significancia'

type Modo = 'validacao' | 'modelos' | 'significancia' | 'editor' | 'curva'

export function PaginaMetricas() {
  const [modo, setModo] = useState<Modo>('validacao')

  return (
    <div className="space-y-6">
      <Segmentos
        valor={modo}
        onChange={setModo}
        opcoes={[
          { valor: 'validacao', rotulo: 'Validação cruzada' },
          { valor: 'modelos', rotulo: 'Split único' },
          { valor: 'significancia', rotulo: 'Significância' },
          { valor: 'editor', rotulo: 'Matriz editável' },
          { valor: 'curva', rotulo: 'Ag × Kappa × Tau' },
        ]}
      />
      {modo === 'validacao' && <PainelValidacao />}
      {modo === 'modelos' && <PainelModelos />}
      {modo === 'significancia' && <PaginaSignificancia />}
      {modo === 'editor' && <PainelEditor />}
      {modo === 'curva' && <PainelCurva />}
    </div>
  )
}

/* -------------------------------------------------------- validacao cruzada */
function PainelValidacao() {
  const { config, set } = usarConfig()
  const { info, categorico } = usarDataset(config.dataset)
  const [k, setK] = useState(5)
  const [repeticoes, setRepeticoes] = useState<number | null>(null)

  // Cada avaliacao treina seis classificadores; com 1000 amostras, 25 avaliacoes
  // levam mais de 30 s. Bases grandes comecam com 1 repeticao, e o slider
  // continua disponivel para quem quiser mais estabilidade.
  const n = info?.n_amostras ?? 0
  const repeticoesEfetivas = repeticoes ?? (n > 400 ? 1 : 5)

  const q = useQuery({
    queryKey: ['metricas', 'validacao', config, k, repeticoesEfetivas],
    queryFn: () =>
      api.metricas.validacaoCruzada({
        ...config,
        k,
        repeticoes: repeticoesEfetivas,
      }),
  })

  const d = q.data
  const ordenados = d
    ? Object.entries(d.resultados).sort((a, b) => b[1].media - a[1].media)
    : []
  const melhor = ordenados[0]?.[1].media ?? 1

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set} mostrarProporcao={false} />

        <Card titulo="parâmetros da validação">
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
              valor={repeticoesEfetivas}
              onChange={setRepeticoes}
              min={1}
              max={20}
              passo={1}
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            {k} dobras × {repeticoesEfetivas}{' '}
            {repeticoesEfetivas === 1 ? 'repetição' : 'repetições'} ={' '}
            <strong>{k * repeticoesEfetivas} avaliações</strong> por
            classificador, sobre {n || '…'} amostras.
          </p>
        </Card>

        <Card titulo="por que validação cruzada?">
          <p className="text-sm leading-relaxed text-secondary">
            Um único split de 70/30 deixa só{' '}
            <strong>{n ? Math.round(n * 0.3) : '…'} amostras</strong> de teste.
            Um erro a mais muda a acurácia em{' '}
            {n ? num(100 / Math.round(n * 0.3), 1) : '…'} pontos, e o resultado
            oscila conforme a semente sorteada.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-secondary">
            Na validação cruzada <strong>toda</strong> amostra é testada
            exatamente uma vez, e o desvio entre as dobras mostra o quanto o
            número é confiável.
          </p>
          <BlocoFormula
            className="mt-3"
            latex={String.raw`\bar{A} \pm z\,\frac{s}{\sqrt{n}}`}
            explicacao="Média das acurácias das dobras, com o intervalo de confiança de 95%."
          />
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
            <Card titulo="acurácia média por classificador">
              <div className="space-y-4">
                {ordenados.map(([chave, r]) => (
                  <div key={chave}>
                    <div className="mb-1 flex items-baseline justify-between gap-3">
                      <span className="flex items-center gap-2 text-sm font-medium text-primary">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: CORES_MODELO[chave] }}
                        />
                        {r.nome}
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
                          backgroundColor: CORES_MODELO[chave],
                        }}
                      />
                      <div
                        className="absolute inset-y-1 rounded-sm"
                        style={{
                          left: `${(r.ic_baixo / melhor) * 100}%`,
                          width: `${((r.ic_alto - r.ic_baixo) / melhor) * 100}%`,
                          backgroundColor: CORES_MODELO[chave],
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

              <Legenda
                className="mt-4"
                itens={[
                  {
                    cor: 'hsl(var(--text-muted))',
                    rotulo: 'faixa clara',
                    descricao: 'acurácia média',
                  },
                  {
                    cor: 'hsl(var(--text-secondary))',
                    rotulo: 'faixa sólida',
                    descricao: 'intervalo de confiança de 95%',
                  },
                ]}
              />

              <p className="mt-3 text-xs text-muted">
                {d.config.n_amostras} amostras · {d.config.k} dobras ×{' '}
                {d.config.repeticoes}{' '}
                {d.config.repeticoes === 1 ? 'repetição' : 'repetições'} ={' '}
                {d.config.n_avaliacoes} avaliações por classificador.
              </p>
            </Card>

            <Card titulo="métricas acumuladas (todas as dobras)">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-subtle">
                      <th className="py-2 pr-3 text-left text-[11px] font-semibold text-muted">
                        Classificador
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        Acerto médio
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        Desvio
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        Kappa
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        Tau
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {ordenados.map(([chave, r]) => (
                      <tr
                        key={chave}
                        className="border-b border-subtle/60 last:border-0"
                      >
                        <td className="py-2.5 pr-3">
                          <span className="inline-flex items-center gap-2">
                            <span
                              className="h-2.5 w-2.5 rounded-full"
                              style={{ backgroundColor: CORES_MODELO[chave] }}
                            />
                            <span className="font-medium text-primary">
                              {r.nome}
                            </span>
                          </span>
                        </td>
                        <td className="px-2 py-2.5 text-right tabular text-primary">
                          {pct(r.media)}
                        </td>
                        <td className="px-2 py-2.5 text-right tabular text-muted">
                          ± {pct(r.desvio)}
                        </td>
                        <td className="px-2 py-2.5 text-right tabular text-secondary">
                          {num(r.relatorio.kappa, 4)}
                        </td>
                        <td className="px-2 py-2.5 text-right tabular text-secondary">
                          {num(r.relatorio.tau, 4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card titulo="teste Z sobre o Kappa acumulado">
              <TabelaTestesZ comparacoes={d.comparacoes} />
            </Card>

            {!categorico && (
            <Card titulo="como ler estes números">
              <Nota tom="atencao" titulo="Por que as pétalas dão quase 100%">
                As pétalas do Iris são quase perfeitamente separáveis — é uma
                propriedade conhecida da base, não um erro de implementação.
                Troque para <strong>Sépalas</strong> no painel ao lado: os
                classificadores caem para a faixa de 76 a 81%, as diferenças
                entre eles ficam visíveis e o teste Z passa a acusar
                significância.
              </Nota>
              <Nota
                tom="info"
                className="mt-3"
                titulo="Split único × validação cruzada"
              >
                Na aba <strong>Split único</strong>, com a semente 42, vários
                classificadores marcam 100% nas pétalas. Aqui a mesma
                configuração dá 96 a 98%: o split único testa 45 amostras uma
                vez, enquanto a validação cruzada testa todas as{' '}
                {d.config.n_amostras}, várias vezes.
              </Nota>
              <Nota
                tom="info"
                className="mt-3"
                titulo="A Regra Delta OvA e o versicolor"
              >
                O Um-Contra-Todos linear fica preso perto de 67% nas pétalas
                porque o <em>versicolor</em> ocupa a faixa central entre as
                outras duas classes — não existe uma única reta que o separe do
                conjunto das outras. É o mesmo limite do XOR reaparecendo, e é
                o que motiva os classificadores não lineares.
              </Nota>
            </Card>
            )}

            {categorico && (
              <Card titulo="como ler estes números">
                <Nota tom="atencao" titulo="Existe um teto teórico aqui">
                  O dataset do seminário tem 8% de ruído de rótulo injetado de
                  propósito, então nenhum classificador pode passar de ~92% —
                  esse é o erro irredutível. Um resultado próximo disso não é
                  overfitting: é o melhor possível.
                </Nota>
                <Nota tom="info" className="mt-3" titulo="Codificação ordinal">
                  Os atributos categóricos viraram inteiros (Sol=0, Vento=1,
                  Chuva=2). Isso inventa uma ordem que não existe e prejudica
                  quem corta por limiar — a floresta ID3 multi-way do script
                  <code> seminario_fim_de_semana.py</code> chega a 94% nos
                  mesmos dados.
                </Nota>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------- comparar modelos */
function PainelModelos() {
  const { config, set } = usarConfig()
  const [selecionado, setSelecionado] = useState('distancia_minima')
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  const q = useQuery({
    queryKey: ['metricas', 'modelos', config],
    queryFn: () => api.metricas.compararModelos(config),
  })

  const rel0 = q.data?.relatorios[selecionado]
  const memoria = useQuery({
    queryKey: ['metricas', 'memoria-modelo', rel0?.nome, rel0?.matriz],
    queryFn: () =>
      api.metricas.memoria({ matriz: rel0!.matriz, nome: rel0!.nome }),
    enabled: memoriaAberta && !!rel0,
  })

  const d = q.data
  const rel = d?.relatorios[selecionado]

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set} />
        <Card titulo="por que kappa?">
          <p className="text-sm leading-relaxed text-secondary">
            O acerto global sozinho é enganoso: um classificador que sempre
            responde a mesma classe já acerta ~33% em 3 classes equilibradas. O
            Kappa desconta esse acerto casual.
          </p>
          <BlocoFormula
            className="mt-3"
            latex={String.raw`\kappa = \frac{A_g - A_c}{1 - A_c}`}
            explicacao="Ac é o acerto esperado ao acaso, calculado dos marginais da matriz."
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
            <Card titulo="ranking dos classificadores">
              <div className="grid gap-3 sm:grid-cols-2">
                {Object.entries(d.relatorios)
                  .sort((a, b) => b[1].kappa - a[1].kappa)
                  .map(([chave, r]) => (
                    <button
                      key={chave}
                      onClick={() => setSelecionado(chave)}
                      className={`rounded-lg border px-4 py-3 text-left transition-all ${
                        selecionado === chave
                          ? 'border-accent-500 bg-accent-500/5 ring-1 ring-accent-500/30'
                          : 'border-subtle bg-sunken hover:border-strong'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: CORES_MODELO[chave] }}
                        />
                        <span className="text-sm font-medium text-primary">
                          {r.nome}
                        </span>
                      </div>
                      <div className="mt-2 flex items-baseline gap-4">
                        <span>
                          <span className="tabular text-xl font-semibold text-primary">
                            {pct(r.acerto_global)}
                          </span>
                          <span className="ml-1 text-[10px] text-muted">Ag</span>
                        </span>
                        <span>
                          <span className="tabular text-sm font-medium text-secondary">
                            {num(r.kappa, 4)}
                          </span>
                          <span className="ml-1 text-[10px] text-muted">κ</span>
                        </span>
                      </div>
                    </button>
                  ))}
              </div>
              <p className="mt-3 text-xs text-muted">
                Todos avaliados no mesmo split ({d.n_teste} amostras de teste).
                Clique num modelo para ver os detalhes abaixo.
              </p>
            </Card>

            {rel && (
              <>
                <Card
                  titulo={`detalhes — ${rel.nome}`}
                  acao={
                    <Botao tamanho="sm" onClick={() => setMemoriaAberta(true)}>
                      <FileText size={13} />
                      Ver cálculos
                    </Botao>
                  }
                >
                  <ResumoGlobal relatorio={rel} />
                </Card>

                <div className="grid gap-6 lg:grid-cols-2">
                  <Card titulo="matriz de confusão">
                    <MatrizConfusao relatorio={rel} classes={classesDoRelatorio(rel)} />
                  </Card>
                  <Card titulo="métricas por classe">
                    <TabelaPorClasse relatorio={rel} classes={classesDoRelatorio(rel)} />
                  </Card>
                </div>
              </>
            )}

            <Card titulo="teste Z de significância entre os pares">
              <TabelaTestesZ comparacoes={d.comparacoes} />
              <Nota tom="info" className="mt-4">
                Um p-valor alto significa que a diferença observada entre os dois
                classificadores <strong>pode ser explicada pelo acaso</strong> —
                acurácias diferentes não implicam modelos estatisticamente
                diferentes.
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

/* ------------------------------------------------------------ matriz editavel */
function matrizInicial(): Record<string, Record<string, number>> {
  return {
    setosa: { setosa: 15, versicolor: 0, virginica: 0 },
    versicolor: { setosa: 0, versicolor: 14, virginica: 1 },
    virginica: { setosa: 0, versicolor: 1, virginica: 14 },
  }
}

function PainelEditor() {
  const [matriz, setMatriz] = useState(matrizInicial)
  const [debounced, setDebounced] = useState(matriz)
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(matriz), 250)
    return () => clearTimeout(t)
  }, [matriz])

  const q = useQuery({
    queryKey: ['metricas', 'avaliar', debounced],
    queryFn: () =>
      api.metricas.avaliar({ matriz: debounced, nome: 'Matriz personalizada' }),
  })

  const memoria = useQuery({
    queryKey: ['metricas', 'memoria', debounced],
    queryFn: () =>
      api.metricas.memoria({ matriz: debounced, nome: 'Matriz personalizada' }),
    enabled: memoriaAberta,
  })

  const editar = (predito: string, real: string, valor: number) =>
    setMatriz((m) => ({ ...m, [predito]: { ...m[predito], [real]: valor } }))

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Card
        titulo="matriz de confusão editável"
        acao={
          <span className="flex gap-2">
            <Botao tamanho="sm" onClick={() => setMemoriaAberta(true)}>
              <FileText size={13} />
              Ver cálculos
            </Botao>
            <Botao
              variante="fantasma"
              tamanho="sm"
              onClick={() => setMatriz(matrizInicial())}
            >
              restaurar
            </Botao>
          </span>
        }
      >
        {q.data && (
          <MatrizConfusao
            relatorio={q.data.relatorio}
            classes={classesDoRelatorio(q.data.relatorio)}
            editavel
            onEditar={editar}
          />
        )}
        <Nota tom="info" className="mt-4">
          Edite qualquer célula e observe como o Kappa reage. Concentre os erros
          numa única classe e compare com erros distribuídos: o acerto global
          pode ser idêntico enquanto o Kappa muda.
        </Nota>
      </Card>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}
        {q.data && (
          <>
            <Card titulo="métricas resultantes">
              <ResumoGlobal relatorio={q.data.relatorio} />
              <p className="mt-3 text-xs text-muted">
                Total de {q.data.total_amostras} amostras na matriz.
              </p>
            </Card>
            <Card titulo="métricas por classe">
              <TabelaPorClasse
                relatorio={q.data.relatorio}
                classes={classesDoRelatorio(q.data.relatorio)}
              />
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

/* --------------------------------------------------------------- curva Kappa */
function PainelCurva() {
  const [n, setN] = useState(15)

  const q = useQuery({
    queryKey: ['metricas', 'curva', n],
    queryFn: () => api.metricas.curvaKappa({ n_por_classe: n, passos: 41 }),
  })

  const dados = q.data?.pontos.map((p) => ({
    x: Number((p.acerto_global * 100).toFixed(2)),
    kappa: p.kappa,
    tau: p.tau,
    ag: p.acerto_global,
  }))

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <Card titulo="parâmetros">
          <Slider
            rotulo="Amostras por classe"
            valor={n}
            onChange={setN}
            min={5}
            max={100}
            passo={5}
          />
        </Card>
        <Card titulo="kappa × tau">
          <p className="text-sm leading-relaxed text-secondary">
            Ambos corrigem o acerto casual, mas de formas diferentes: o{' '}
            <strong>Kappa</strong> usa os marginais observados na matriz,
            enquanto o <strong>Tau</strong> assume classes equiprováveis.
          </p>
          <BlocoFormula
            className="mt-3"
            latex={String.raw`\tau = \frac{A_g - 1/M}{1 - 1/M}`}
            explicacao="M é o número de classes — aqui, 1/M = 1/3 ≈ 0,333."
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
        {dados && (
          <>
            <Card titulo="como Kappa e Tau reagem ao acerto global">
              <GraficoLinha
                dados={dados}
                series={[
                  { chave: 'ag', rotulo: 'Acerto Global', cor: '#94a3b8', tracejada: true },
                  { chave: 'kappa', rotulo: 'Kappa (κ)', cor: '#f59e0b' },
                  { chave: 'tau', rotulo: 'Tau (τ)', cor: '#0ea5e9' },
                ]}
                rotuloX="acerto global (%)"
                rotuloY="valor do coeficiente"
                altura={340}
              />
              <Legenda
                className="mt-2"
                itens={[
                  {
                    cor: '#94a3b8',
                    forma: 'linha-tracejada',
                    rotulo: 'Acerto Global',
                    descricao: 'referência (proporção bruta de acertos)',
                  },
                  {
                    cor: '#f59e0b',
                    forma: 'linha',
                    rotulo: 'Kappa',
                    descricao: 'desconta o acerto casual dos marginais',
                  },
                  {
                    cor: '#0ea5e9',
                    forma: 'linha',
                    rotulo: 'Tau',
                    descricao: 'desconta assumindo classes equiprováveis',
                  },
                ]}
              />
            </Card>

            <Card titulo="leitura do gráfico">
              <div className="grid gap-3 sm:grid-cols-3">
                <Metrica
                  rotulo="Ag = 33%"
                  valor={num(
                    dados.find((d) => Math.abs(d.x - 33.33) < 4)?.kappa ?? 0,
                    3,
                  )}
                  detalhe="κ ≈ 0 — equivale ao acaso"
                  destaque="ruim"
                />
                <Metrica
                  rotulo="Ag = 70%"
                  valor={num(
                    dados.find((d) => Math.abs(d.x - 70) < 4)?.kappa ?? 0,
                    3,
                  )}
                  detalhe="κ bem abaixo do Ag"
                  destaque="medio"
                />
                <Metrica
                  rotulo="Ag = 100%"
                  valor={num(dados.at(-1)?.kappa ?? 0, 3)}
                  detalhe="κ = 1 — concordância perfeita"
                  destaque="bom"
                />
              </div>
              <Nota tom="atencao" className="mt-4" titulo="Por que isso importa">
                Note que o Kappa fica sempre <strong>abaixo</strong> do acerto
                global. Reportar apenas o Ag superestima a qualidade do
                classificador — por isso os laboratórios exigem Kappa, Tau e o
                teste Z.
              </Nota>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
