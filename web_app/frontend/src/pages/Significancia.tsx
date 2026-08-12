/**
 * Lab 3 — Testes de significancia estatistica entre classificadores.
 *
 * Responde a observacao do professor: apos calcular o coeficiente de
 * Matthews (e as demais metricas), verificar se a diferenca entre dois
 * classificadores e realmente significativa.
 *
 * O teste Z de Kappa dos laboratorios soma as variancias — ou seja, assume
 * que as duas avaliacoes sao INDEPENDENTES. Aqui os classificadores sao
 * avaliados no MESMO conjunto de teste, entao as estimativas sao pareadas.
 * Os tres testes desta aba (McNemar, bootstrap pareado e permutacao) sao os
 * corretos para esse cenario.
 */
import { useQuery } from '@tanstack/react-query'
import { FileText, Minus, TrendingDown, TrendingUp } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  PainelConfig,
  usarConfig,
  usarDataset,
} from '@/components/Controles'
import { BlocoFormula } from '@/components/Formula'
import { MemoriaGenerica } from '@/components/MemoriaGenerica'
import {
  Badge,
  Botao,
  Card,
  Carregando,
  ErroBox,
  Legenda,
  Nota,
  Segmentos,
  Select,
  Slider,
} from '@/components/ui'
import { api } from '@/lib/api'
import type { McNemar, ParSignificancia } from '@/lib/types'
import { CORES_MODELO, num } from '@/lib/utils'

const COR_A = '#0ea5e9'
const COR_B = '#f59e0b'
const COR_SIM = '#10b981'
const COR_NAO = '#94a3b8'

type Aba = 'par' | 'matriz'

/** Abreviacoes para caber nos cabecalhos de tabela. */
const ABREV_METRICA: Record<string, string> = {
  mcc: 'MCC',
  kappa: 'κ',
  acerto_global: 'Ag',
  f1: 'F1',
  precisao: 'Precisão',
  revocacao: 'Revocação',
  especificidade: 'Especif.',
}

/** p-valores muito pequenos viram "< 0,000001" em vez de "0,000000". */
function pval(p: number, casas = 6): string {
  const piso = 10 ** -casas
  return p < piso ? `< ${num(piso, casas)}` : num(p, casas)
}

/** "p = 0,0123" ou "p < 0,00001" — evita o "p = < ..." do formato ingenuo. */
function rotuloP(p: number, casas = 6): string {
  const texto = pval(p, casas)
  return texto.startsWith('<') ? `p ${texto}` : `p = ${texto}`
}

export function PaginaSignificancia() {
  const [aba, setAba] = useState<Aba>('par')

  return (
    <div className="space-y-5">
      <Nota tom="atencao" titulo="Por que esta aba existe">
        O teste Z do Kappa soma as variâncias dos dois classificadores, o que
        só vale se as avaliações forem <strong>independentes</strong>. Como os
        dois são avaliados no <strong>mesmo</strong> conjunto de teste — errando
        as mesmas amostras difíceis —, as estimativas são{' '}
        <strong>pareadas</strong>. Os três testes abaixo levam esse pareamento
        em conta e valem para qualquer métrica, inclusive o MCC.
      </Nota>

      <Segmentos
        valor={aba}
        onChange={setAba}
        opcoes={[
          { valor: 'par', rotulo: 'Testar um par' },
          { valor: 'matriz', rotulo: 'Todos os pares' },
        ]}
      />

      {aba === 'par' ? <PainelPar /> : <PainelMatriz />}
    </div>
  )
}

/* ==================================================== teste de um par ===== */
function PainelPar() {
  const { config, set } = usarConfig()
  const { atributos: atributosDoDataset } = usarDataset(config.dataset)
  const [modeloA, setModeloA] = useState('bayes')
  const [modeloB, setModeloB] = useState('delta_ova')
  const [metrica, setMetrica] = useState('mcc')
  const [reamostragens, setReamostragens] = useState(2000)
  const [permutacoes, setPermutacoes] = useState(2000)
  const [memoriaAberta, setMemoriaAberta] = useState(false)

  const lista = useQuery({
    queryKey: ['metricas', 'classificadores'],
    queryFn: () => api.metricas.classificadores(),
    staleTime: Infinity,
  })

  const params = {
    ...config,
    modelo_a: modeloA,
    modelo_b: modeloB,
    metrica,
    n_reamostragens: reamostragens,
    n_permutacoes: permutacoes,
  }

  const q = useQuery({
    queryKey: ['metricas', 'significancia', params],
    queryFn: () => api.metricas.significancia(params),
    enabled: modeloA !== modeloB,
  })

  const memoria = useQuery({
    queryKey: ['metricas', 'significancia-memoria', params],
    queryFn: () => api.metricas.memoriaSignificancia(params),
    enabled: memoriaAberta && modeloA !== modeloB,
  })

  const d = q.data
  const nomeAtributos =
    atributosDoDataset.find((a) => a.id === config.atributos)?.nome ??
    String(config.atributos)
  const opcoesModelo =
    lista.data?.classificadores.map((c) => ({
      valor: c.id,
      rotulo: c.nome,
    })) ?? []
  const opcoesMetrica =
    lista.data?.metricas.map((m) => ({ valor: m.id, rotulo: m.nome })) ?? []

  const histograma = useMemo(() => {
    if (!d) return []
    const { faixas, contagens } = d.bootstrap.distribuicao
    return contagens.map((c, i) => ({
      x: Number(((faixas[i] + faixas[i + 1]) / 2).toFixed(4)),
      n: c,
    }))
  }, [d])

  const quantosSignificativos = d
    ? [
        d.mcnemar.significativo,
        d.bootstrap.significativo,
        d.permutacao.significativo,
      ].filter(Boolean).length
    : 0

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set} />

        <Card titulo="par a comparar">
          <div className="space-y-3">
            <Select
              rotulo="Classificador A"
              valor={modeloA}
              onChange={setModeloA}
              opcoes={opcoesModelo.map((o) => ({
                ...o,
                desabilitado: o.valor === modeloB,
              }))}
            />
            <Select
              rotulo="Classificador B"
              valor={modeloB}
              onChange={setModeloB}
              opcoes={opcoesModelo.map((o) => ({
                ...o,
                desabilitado: o.valor === modeloA,
              }))}
            />
            <Select
              rotulo="Métrica testada"
              valor={metrica}
              onChange={setMetrica}
              opcoes={opcoesMetrica}
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            A métrica escolhida vale para o bootstrap e a permutação. O McNemar
            compara apenas <em>acertos</em> e por isso independe dela.
          </p>
        </Card>

        <Card titulo="esforço computacional">
          <div className="space-y-4">
            <Slider
              rotulo="Reamostragens (bootstrap)"
              valor={reamostragens}
              onChange={setReamostragens}
              min={200}
              max={5000}
              passo={200}
            />
            <Slider
              rotulo="Permutações"
              valor={permutacoes}
              onChange={setPermutacoes}
              min={200}
              max={5000}
              passo={200}
            />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            Mais repetições = intervalos mais estáveis. Com 2000 o IC já varia
            menos de 0,01 entre execuções.
          </p>
        </Card>

        <Card titulo="os três testes">
          <ul className="space-y-3 text-sm leading-relaxed text-secondary">
            <li>
              <strong className="text-primary">McNemar</strong> — olha só as
              amostras em que os dois discordam. Exato (binomial) quando há
              menos de 25 discordantes.
            </li>
            <li>
              <strong className="text-primary">Bootstrap pareado</strong> —
              reamostra o teste com reposição levando sempre o par de predições
              junto, e devolve o IC 95% da diferença.
            </li>
            <li>
              <strong className="text-primary">Permutação</strong> — troca as
              predições de A e B ao acaso para construir a distribuição sob H₀.
            </li>
          </ul>
        </Card>
      </div>

      <div className="space-y-6">
        {modeloA === modeloB && (
          <Card>
            <Nota tom="atencao">
              Escolha dois classificadores <strong>diferentes</strong> — não faz
              sentido testar um modelo contra ele mesmo.
            </Nota>
          </Card>
        )}
        {q.isPending && modeloA !== modeloB && (
          <Card>
            <Carregando texto="Rodando McNemar, bootstrap e permutação…" />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            {/* -------------------------------------------------- veredito */}
            <Card
              titulo="veredito"
              acao={
                <Botao tamanho="sm" onClick={() => setMemoriaAberta(true)}>
                  <FileText size={13} />
                  Ver cálculos
                </Botao>
              }
            >
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <div>
                  <p className="text-sm text-secondary">
                    <span
                      className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full align-middle"
                      style={{
                        backgroundColor: CORES_MODELO[d.modelo_a.id] ?? COR_A,
                      }}
                    />
                    <strong className="text-primary">{d.modelo_a.nome}</strong>{' '}
                    <span className="text-muted">×</span>{' '}
                    <span
                      className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full align-middle"
                      style={{
                        backgroundColor: CORES_MODELO[d.modelo_b.id] ?? COR_B,
                      }}
                    />
                    <strong className="text-primary">{d.modelo_b.nome}</strong>
                  </p>
                  <p className="mt-1 text-xs text-muted">
                    {d.nome_metrica} · {d.n_amostras} amostras de teste ·{' '}
                    {nomeAtributos}
                  </p>
                </div>
                <Diferenca
                  valor={d.bootstrap.diferenca}
                  significativa={quantosSignificativos >= 2}
                />
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-3">
                <CartaoTeste
                  nome="McNemar"
                  detalhe={d.mcnemar.metodo}
                  p={d.mcnemar.p_valor}
                  significativo={d.mcnemar.significativo}
                />
                <CartaoTeste
                  nome="Bootstrap pareado"
                  detalhe={`IC 95% [${d.bootstrap.ic_baixo >= 0 ? '+' : ''}${num(
                    d.bootstrap.ic_baixo,
                    3,
                  )}, ${d.bootstrap.ic_alto >= 0 ? '+' : ''}${num(
                    d.bootstrap.ic_alto,
                    3,
                  )}]`}
                  significativo={d.bootstrap.significativo}
                />
                <CartaoTeste
                  nome="Permutação"
                  detalhe={`${d.permutacao.extremos} de ${d.permutacao.n_permutacoes} extremas`}
                  p={d.permutacao.p_valor}
                  significativo={d.permutacao.significativo}
                />
              </div>

              <Nota
                tom={quantosSignificativos >= 2 ? 'ok' : 'info'}
                className="mt-4"
                titulo={
                  quantosSignificativos >= 2
                    ? 'A diferença é estatisticamente significativa'
                    : 'Não há evidência de diferença real'
                }
              >
                {quantosSignificativos >= 2 ? (
                  <>
                    {quantosSignificativos} dos 3 testes rejeitam H₀ a 5%. A
                    vantagem de{' '}
                    <strong>
                      {d.bootstrap.diferenca > 0
                        ? d.modelo_a.nome
                        : d.modelo_b.nome}
                    </strong>{' '}
                    em {d.nome_metrica} dificilmente é fruto do acaso amostral.
                  </>
                ) : (
                  <>
                    Nenhum (ou apenas um) dos testes rejeita H₀ a 5%. A
                    diferença de {num(Math.abs(d.bootstrap.diferenca), 4)} em{' '}
                    {d.nome_metrica} é compatível com o que se obteria sorteando
                    outro conjunto de teste — não se pode afirmar que um
                    classificador é melhor.
                  </>
                )}
              </Nota>
            </Card>

            {/* --------------------------------------- contingência McNemar */}
            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="teste 1 — tabela de McNemar">
                <TabelaMcNemar
                  m={d.mcnemar}
                  nomeA={d.modelo_a.nome}
                  nomeB={d.modelo_b.nome}
                />
                <BlocoFormula
                  className="mt-4"
                  latex={String.raw`\chi^2 = \frac{(|b - c| - 1)^2}{b + c}`}
                  explicacao="Com b + c < 25 usa-se o binomial exato: p = 2·P(X ≤ min(b,c)), X ~ Bin(b+c, 0,5)."
                />
                <p className="mt-3 text-xs leading-relaxed text-muted">
                  {d.mcnemar.observacao}
                </p>
              </Card>

              <Card titulo="métricas lado a lado">
                <TabelaMetricas
                  metricas={d.metricas}
                  nomeA={d.modelo_a.nome}
                  nomeB={d.modelo_b.nome}
                  destaque={d.metrica}
                />
                <p className="mt-3 text-xs leading-relaxed text-muted">
                  MCC multiclasse (Gorodkin):{' '}
                  <strong className="text-secondary">
                    {num(d.mcc_multiclasse.a, 4)}
                  </strong>{' '}
                  ×{' '}
                  <strong className="text-secondary">
                    {num(d.mcc_multiclasse.b, 4)}
                  </strong>{' '}
                  — a versão macro da tabela é a média dos MCC um-contra-resto.
                </p>
              </Card>
            </div>

            {/* --------------------------------------------- bootstrap hist */}
            <Card titulo={`teste 2 — bootstrap da diferença em ${d.nome_metrica}`}>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={histograma}
                  margin={{ top: 8, right: 16, bottom: 26, left: 4 }}
                >
                  <CartesianGrid
                    stroke="hsl(var(--grid-line))"
                    strokeDasharray="3 3"
                  />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tick={{ fontSize: 11, fill: 'hsl(var(--text-muted))' }}
                    stroke="hsl(var(--border-strong))"
                    tickFormatter={(v: number) => num(v, 2)}
                    label={{
                      value: `diferença ${d.modelo_a.nome} − ${d.modelo_b.nome}`,
                      position: 'insideBottom',
                      offset: -16,
                      style: {
                        fontSize: 11,
                        fill: 'hsl(var(--text-secondary))',
                        textAnchor: 'middle',
                      },
                    }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: 'hsl(var(--text-muted))' }}
                    stroke="hsl(var(--border-strong))"
                    label={{
                      value: 'reamostragens',
                      angle: -90,
                      position: 'insideLeft',
                      style: {
                        fontSize: 11,
                        fill: 'hsl(var(--text-secondary))',
                        textAnchor: 'middle',
                      },
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'hsl(var(--bg-surface))',
                      border: '1px solid hsl(var(--border-strong))',
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number) => [v, 'reamostragens']}
                    labelFormatter={(v: number) => `diferença ≈ ${num(v, 4)}`}
                  />
                  <Bar
                    dataKey="n"
                    fill={d.bootstrap.significativo ? COR_SIM : COR_NAO}
                    fillOpacity={0.55}
                  />
                  <ReferenceLine
                    x={0}
                    stroke="#ef4444"
                    strokeDasharray="4 3"
                    label={{
                      value: 'H₀: sem diferença',
                      position: 'top',
                      style: { fontSize: 10, fill: '#ef4444' },
                    }}
                  />
                  <ReferenceLine
                    x={d.bootstrap.ic_baixo}
                    stroke="hsl(var(--text-secondary))"
                    strokeDasharray="2 2"
                  />
                  <ReferenceLine
                    x={d.bootstrap.ic_alto}
                    stroke="hsl(var(--text-secondary))"
                    strokeDasharray="2 2"
                  />
                  <ReferenceLine
                    x={d.bootstrap.diferenca}
                    stroke={d.bootstrap.significativo ? COR_SIM : COR_NAO}
                    strokeWidth={2}
                  />
                </BarChart>
              </ResponsiveContainer>

              <Legenda
                className="mt-2"
                itens={[
                  {
                    cor: d.bootstrap.significativo ? COR_SIM : COR_NAO,
                    rotulo: 'barras',
                    descricao: `distribuição das ${d.bootstrap.n_reamostragens} diferenças reamostradas`,
                  },
                  {
                    cor: d.bootstrap.significativo ? COR_SIM : COR_NAO,
                    forma: 'linha',
                    rotulo: 'linha sólida',
                    descricao: `diferença observada (${d.bootstrap.diferenca >= 0 ? '+' : ''}${num(d.bootstrap.diferenca, 4)})`,
                  },
                  {
                    cor: 'hsl(var(--text-secondary))',
                    forma: 'linha-tracejada',
                    rotulo: 'tracejado cinza',
                    descricao: 'limites do intervalo de confiança de 95%',
                  },
                  {
                    cor: '#ef4444',
                    forma: 'linha-tracejada',
                    rotulo: 'tracejado vermelho',
                    descricao: 'zero — hipótese nula de que os dois empatam',
                  },
                ]}
              />

              <div className="mt-4 grid gap-3 text-sm sm:grid-cols-4">
                <ItemNumero
                  rotulo={d.modelo_a.nome}
                  valor={num(d.bootstrap.metrica_a, 4)}
                  cor={CORES_MODELO[d.modelo_a.id] ?? COR_A}
                />
                <ItemNumero
                  rotulo={d.modelo_b.nome}
                  valor={num(d.bootstrap.metrica_b, 4)}
                  cor={CORES_MODELO[d.modelo_b.id] ?? COR_B}
                />
                <ItemNumero
                  rotulo="Erro padrão"
                  valor={num(d.bootstrap.erro_padrao, 4)}
                />
                <ItemNumero
                  rotulo="IC contém zero?"
                  valor={d.bootstrap.contem_zero ? 'sim' : 'não'}
                  cor={d.bootstrap.contem_zero ? COR_NAO : COR_SIM}
                />
              </div>

              <Nota tom="info" className="mt-4">
                Se a linha vermelha (zero) cai <strong>dentro</strong> do
                intervalo tracejado, a diferença observada é compatível com o
                acaso. Se cai fora, a vantagem se mantém em praticamente toda
                reamostragem do conjunto de teste.
              </Nota>
            </Card>

            {/* ------------------------------------ permutação + Z clássico */}
            <div className="grid gap-6 lg:grid-cols-2">
              <Card titulo="teste 3 — permutação">
                <BlocoFormula
                  latex={String.raw`p = \frac{1 + \#\{|\Delta^{*}| \geq |\Delta_{obs}|\}}{1 + B}`}
                  explicacao="Fração de permutações tão extremas quanto a diferença observada."
                />
                <div className="mt-4 space-y-2 text-sm">
                  <LinhaValor
                    rotulo="Diferença observada"
                    valor={`${d.permutacao.diferenca_observada >= 0 ? '+' : ''}${num(d.permutacao.diferenca_observada, 4)}`}
                  />
                  <LinhaValor
                    rotulo="Permutações extremas"
                    valor={`${d.permutacao.extremos} de ${d.permutacao.n_permutacoes}`}
                  />
                  <LinhaValor
                    rotulo="p-valor"
                    valor={pval(d.permutacao.p_valor)}
                    destaque={d.permutacao.significativo}
                  />
                </div>
              </Card>

              <Card titulo="contraste com o teste Z de Kappa">
                <div className="space-y-2 text-sm">
                  <LinhaValor
                    rotulo="Z (assume independência)"
                    valor={num(d.teste_z_kappa.z, 4)}
                  />
                  <LinhaValor
                    rotulo="p-valor do teste Z"
                    valor={pval(d.teste_z_kappa.p)}
                    destaque={d.teste_z_kappa.significativo}
                  />
                  <LinhaValor
                    rotulo="p-valor do McNemar"
                    valor={pval(d.mcnemar.p_valor)}
                    destaque={d.mcnemar.significativo}
                  />
                </div>
                <Nota tom="atencao" className="mt-4">
                  O teste Z soma as variâncias como se os dois classificadores
                  tivessem sido avaliados em conjuntos <em>separados</em>. Como
                  eles compartilham o mesmo teste, existe covariância positiva
                  entre as estimativas — ignorá-la infla o denominador e{' '}
                  <strong>subestima</strong> a significância.
                </Nota>
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

/* ==================================================== todos os pares ====== */
function PainelMatriz() {
  const { config, set } = usarConfig()
  const { atributos: atributosDoDataset } = usarDataset(config.dataset)
  const [metrica, setMetrica] = useState('mcc')
  const [reamostragens, setReamostragens] = useState(600)

  const lista = useQuery({
    queryKey: ['metricas', 'classificadores'],
    queryFn: () => api.metricas.classificadores(),
    staleTime: Infinity,
  })

  const q = useQuery({
    queryKey: ['metricas', 'matriz-significancia', config, metrica, reamostragens],
    queryFn: () =>
      api.metricas.matrizSignificancia({
        ...config,
        metrica,
        n_reamostragens: reamostragens,
      }),
  })

  const d = q.data
  const melhor = d?.modelos[0]

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set} />
        <Card titulo="parâmetros">
          <Select
            rotulo="Métrica"
            valor={metrica}
            onChange={setMetrica}
            opcoes={
              lista.data?.metricas.map((m) => ({
                valor: m.id,
                rotulo: m.nome,
              })) ?? []
            }
          />
          <Slider
            className="mt-4"
            rotulo="Reamostragens por par"
            valor={reamostragens}
            onChange={setReamostragens}
            min={200}
            max={2000}
            passo={200}
          />
          <p className="mt-3 text-xs leading-relaxed text-muted">
            São 10 pares testados de uma vez — mantenha um valor moderado para
            a resposta ficar rápida na apresentação.
          </p>
        </Card>
        <Card titulo="como ler a tabela">
          <p className="text-sm leading-relaxed text-secondary">
            Cada linha é um par. Os três selos mostram o veredito de cada teste
            a 5%. Quando os três concordam, a conclusão é sólida; quando
            divergem, a diferença está no limiar e vale reportar o intervalo de
            confiança em vez de um sim/não.
          </p>
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando texto="Testando todos os pares…" />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <Card titulo={`ranking por ${d.nome_metrica}`}>
              <div className="space-y-3">
                {d.modelos.map((m) => (
                  <div key={m.id}>
                    <div className="mb-1 flex items-baseline justify-between gap-3">
                      <span className="flex items-center gap-2 text-sm font-medium text-primary">
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: CORES_MODELO[m.id] }}
                        />
                        {m.nome}
                      </span>
                      <span className="tabular text-sm font-semibold text-primary">
                        {num(m.valor, 4)}
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded bg-sunken">
                      <div
                        className="h-full rounded"
                        style={{
                          width: `${Math.max(0, (m.valor / (melhor?.valor || 1)) * 100)}%`,
                          backgroundColor: CORES_MODELO[m.id],
                          opacity: 0.7,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs text-muted">
                Mesmo split de {d.config.n_teste} amostras de teste ·{' '}
                {atributosDoDataset.find((a) => a.id === config.atributos)
                  ?.nome ?? String(config.atributos)}.
              </p>
            </Card>

            <Card titulo="significância par a par">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-subtle">
                      <th className="py-2 pr-3 text-left text-[11px] font-semibold text-muted">
                        Par
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        Δ {ABREV_METRICA[d.metrica] ?? d.nome_metrica}
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        IC 95%
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        p McNemar
                      </th>
                      <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
                        p Permut.
                      </th>
                      <th className="px-2 py-2 text-center text-[11px] font-semibold text-muted">
                        Veredito
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...d.pares]
                      .sort((a, b) => a.p_mcnemar - b.p_mcnemar)
                      .map((p) => (
                        <LinhaPar key={`${p.a}-${p.b}`} p={p} />
                      ))}
                  </tbody>
                </table>
              </div>

              <Legenda
                className="mt-4"
                itens={[
                  {
                    cor: COR_SIM,
                    rotulo: 'M / B / P verdes',
                    descricao:
                      'McNemar / Bootstrap / Permutação acusaram diferença (p < 0,05)',
                  },
                  {
                    cor: COR_NAO,
                    rotulo: 'selos cinzas',
                    descricao: 'o teste não rejeitou a hipótese de empate',
                  },
                ]}
              />

              <Nota tom="info" className="mt-4">
                Troque os atributos para <strong>Sépalas</strong> no painel ao
                lado: os classificadores ficam mais próximos entre si e vários
                pares deixam de ser significativos — é exatamente o cenário em
                que reportar só a acurácia levaria a uma conclusão errada.
              </Nota>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

/* ========================================================= auxiliares ===== */
function LinhaPar({ p }: { p: ParSignificancia }) {
  const n = [
    p.significativo_mcnemar,
    p.significativo_bootstrap,
    p.significativo_permutacao,
  ].filter(Boolean).length

  return (
    <tr className="border-b border-subtle/60 last:border-0">
      <td className="py-2.5 pr-3">
        <span className="flex items-center gap-1.5">
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: CORES_MODELO[p.a] }}
          />
          <span className="text-primary">{p.nome_a}</span>
          <span className="text-muted">×</span>
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: CORES_MODELO[p.b] }}
          />
          <span className="text-primary">{p.nome_b}</span>
        </span>
      </td>
      <td className="px-2 py-2.5 text-right tabular font-medium text-primary">
        {p.diferenca >= 0 ? '+' : ''}
        {num(p.diferenca, 4)}
      </td>
      <td className="px-2 py-2.5 text-right tabular text-xs text-muted">
        [{p.ic_baixo >= 0 ? '+' : ''}
        {num(p.ic_baixo, 3)}, {p.ic_alto >= 0 ? '+' : ''}
        {num(p.ic_alto, 3)}]
      </td>
      <td
        className={`px-2 py-2.5 text-right tabular ${
          p.significativo_mcnemar ? 'font-semibold text-primary' : 'text-muted'
        }`}
      >
        {pval(p.p_mcnemar, 4)}
      </td>
      <td
        className={`px-2 py-2.5 text-right tabular ${
          p.significativo_permutacao ? 'font-semibold text-primary' : 'text-muted'
        }`}
      >
        {pval(p.p_permutacao, 4)}
      </td>
      <td className="px-2 py-2.5 text-center">
        <span className="inline-flex gap-1">
          <Selo rotulo="M" ativo={p.significativo_mcnemar} />
          <Selo rotulo="B" ativo={p.significativo_bootstrap} />
          <Selo rotulo="P" ativo={p.significativo_permutacao} />
        </span>
        <span className="ml-2 text-[10px] text-muted">{n}/3</span>
      </td>
    </tr>
  )
}

function Selo({ rotulo, ativo }: { rotulo: string; ativo: boolean }) {
  return (
    <span
      title={ativo ? 'p < 0,05' : 'p ≥ 0,05'}
      className={`inline-flex h-5 w-5 items-center justify-center rounded text-[10px] font-bold ${
        ativo
          ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
          : 'bg-sunken text-muted'
      }`}
    >
      {rotulo}
    </span>
  )
}

function CartaoTeste({
  nome,
  detalhe,
  p,
  significativo,
}: {
  nome: string
  detalhe: string
  p?: number
  significativo: boolean
}) {
  return (
    <div
      className={`rounded-lg border px-4 py-3 ${
        significativo
          ? 'border-emerald-500/40 bg-emerald-500/5'
          : 'border-subtle bg-sunken'
      }`}
    >
      <p className="kicker text-muted">{nome}</p>
      {p !== undefined ? (
        <>
          <p className="mt-1 tabular text-lg font-semibold text-primary">
            {rotuloP(p, 5)}
          </p>
          <p className="mt-0.5 text-[11px] text-muted">{detalhe}</p>
        </>
      ) : (
        <p className="mt-1 tabular text-sm font-semibold text-primary">
          {detalhe}
        </p>
      )}
      <Badge tom={significativo ? 'bom' : 'neutro'} className="mt-2">
        {significativo ? 'significativo' : 'não significativo'}
      </Badge>
    </div>
  )
}

function Diferenca({
  valor,
  significativa,
}: {
  valor: number
  significativa: boolean
}) {
  const Icone = valor > 0 ? TrendingUp : valor < 0 ? TrendingDown : Minus
  return (
    <div className="text-right">
      <span className="kicker block text-muted">diferença A − B</span>
      <span
        className={`inline-flex items-center gap-1.5 tabular text-2xl font-semibold ${
          significativa ? 'text-emerald-600 dark:text-emerald-400' : 'text-primary'
        }`}
      >
        <Icone size={18} />
        {valor >= 0 ? '+' : ''}
        {num(valor, 4)}
      </span>
    </div>
  )
}

function TabelaMcNemar({
  m,
  nomeA,
  nomeB,
}: {
  m: McNemar
  nomeA: string
  nomeB: string
}) {
  const celula = (v: number, destaque: boolean) => (
    <td
      className={`border border-subtle px-3 py-3 text-center tabular text-lg ${
        destaque
          ? 'bg-accent-500/10 font-semibold text-primary'
          : 'bg-sunken text-muted'
      }`}
    >
      {v}
    </td>
  )

  return (
    <>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            <th className="px-2 py-2" />
            <th className="px-2 py-2 text-center text-[11px] font-semibold text-muted">
              {nomeB} acertou
            </th>
            <th className="px-2 py-2 text-center text-[11px] font-semibold text-muted">
              {nomeB} errou
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
              {nomeA} acertou
            </th>
            {celula(m.a, false)}
            {celula(m.b, true)}
          </tr>
          <tr>
            <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
              {nomeA} errou
            </th>
            {celula(m.c, true)}
            {celula(m.d, false)}
          </tr>
        </tbody>
      </table>
      <Legenda
        className="mt-3"
        itens={[
          {
            cor: 'hsl(var(--accent-500))',
            rotulo: 'células destacadas (b e c)',
            descricao: `os ${m.discordantes} discordantes — são o único insumo do teste`,
          },
          {
            cor: 'hsl(var(--text-muted))',
            rotulo: 'células cinzas (a e d)',
            descricao: 'ambos acertaram ou ambos erraram — não informam nada',
          },
        ]}
      />
    </>
  )
}

function TabelaMetricas({
  metricas,
  nomeA,
  nomeB,
  destaque,
}: {
  metricas: Record<string, { nome: string; a: number; b: number }>
  nomeA: string
  nomeB: string
  destaque: string
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-subtle">
            <th className="py-2 pr-3 text-left text-[11px] font-semibold text-muted">
              Métrica
            </th>
            <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
              {nomeA}
            </th>
            <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
              {nomeB}
            </th>
            <th className="px-2 py-2 text-right text-[11px] font-semibold text-muted">
              Δ
            </th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metricas).map(([chave, m]) => {
            const dif = m.a - m.b
            const eh = chave === destaque
            return (
              <tr
                key={chave}
                className={`border-b border-subtle/60 last:border-0 ${
                  eh ? 'bg-accent-500/5' : ''
                }`}
              >
                <td className="py-2 pr-3">
                  <span
                    className={eh ? 'font-semibold text-primary' : 'text-secondary'}
                  >
                    {m.nome}
                  </span>
                  {eh && (
                    <span className="ml-2 text-[10px] text-accent-700 dark:text-accent-400">
                      testada
                    </span>
                  )}
                </td>
                <td className="px-2 py-2 text-right tabular text-primary">
                  {num(m.a, 4)}
                </td>
                <td className="px-2 py-2 text-right tabular text-primary">
                  {num(m.b, 4)}
                </td>
                <td
                  className={`px-2 py-2 text-right tabular ${
                    Math.abs(dif) < 1e-9 ? 'text-muted' : 'text-secondary'
                  }`}
                >
                  {dif >= 0 ? '+' : ''}
                  {num(dif, 4)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ItemNumero({
  rotulo,
  valor,
  cor,
}: {
  rotulo: string
  valor: string
  cor?: string
}) {
  return (
    <div className="rounded-lg border border-subtle bg-sunken px-3 py-2">
      <span className="kicker block truncate text-muted">{rotulo}</span>
      <span
        className="tabular text-base font-semibold"
        style={{ color: cor ?? 'hsl(var(--text-primary))' }}
      >
        {valor}
      </span>
    </div>
  )
}

function LinhaValor({
  rotulo,
  valor,
  destaque,
}: {
  rotulo: string
  valor: string
  destaque?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-subtle/60 py-1.5 last:border-0">
      <span className="text-secondary">{rotulo}</span>
      <span
        className={`tabular ${
          destaque
            ? 'font-semibold text-emerald-600 dark:text-emerald-400'
            : 'text-primary'
        }`}
      >
        {valor}
      </span>
    </div>
  )
}
