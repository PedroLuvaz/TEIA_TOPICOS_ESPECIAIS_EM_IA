/**
 * Lab 5.1 — Feedforward (MLP): itens (i) e (ii) do enunciado + slide 34.
 *
 * · item (i)  memoria de calculo da rede 2-2-2 "galinha vs homem"
 * · bonus     canvas 8x8 pintavel, classificado ao vivo por uma rede 64-10-1
 * · item (ii) comparativo MLP (sklearn) x Bayes Otimo x Naive Bayes no Iris
 */
import { useMutation, useQuery } from '@tanstack/react-query'
import { Eraser, FileText } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { PainelConfig, usarConfig } from '@/components/Controles'
import { MemoriaCalculo } from '@/components/MemoriaCalculo'
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
  Nota,
  Segmentos,
  Slider,
} from '@/components/ui'
import { api } from '@/lib/api'
import type { PredicaoImagem } from '@/lib/types'
import { CORES_MODELO, cn, num, pct, tomCinza } from '@/lib/utils'

const CLASSES = ['setosa', 'versicolor', 'virginica']
type Modo = 'item-i' | 'item-ii'

export function PaginaLab51() {
  const [modo, setModo] = useState<Modo>('item-i')

  return (
    <div className="space-y-6">
      <Segmentos
        valor={modo}
        onChange={setModo}
        opcoes={[
          { valor: 'item-i', rotulo: 'Item (i) · Galinha vs Homem' },
          { valor: 'item-ii', rotulo: 'Item (ii) · Comparativo no Iris' },
        ]}
      />
      {modo === 'item-i' ? <PainelItemI /> : <PainelItemII />}
    </div>
  )
}

/* ------------------------------------------------------------- item (i) --- */
function PainelItemI() {
  const [exercicio, setExercicio] = useState<string | null>(null)

  const memoria = useQuery({
    queryKey: ['lab5', 'memoria', exercicio],
    queryFn: () => api.lab5.memoria(exercicio!),
    enabled: !!exercicio,
  })

  return (
    <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-5">
        <Card titulo="item (i) do enunciado">
          <p className="text-sm leading-relaxed text-secondary">
            Rede totalmente conectada <strong>2-2-2</strong> em Python puro (sem
            bibliotecas de ML), com os pesos iniciais do slide,{' '}
            <span className="font-mono">η = 0,05</span> e saída desejada 0 para
            o homem e 1 para a galinha.
          </p>
          <Nota tom="ok" className="mt-3" titulo="Conferido com o slide">
            A alimentação adiante reproduz exatamente os valores da aula:
            out_b₁ = 0,7020 · out_b₂ = 0,5841 · out_c₁ = 0,5934 · out_c₂ =
            0,7353 · E = 0,21108.
          </Nota>
        </Card>

        <Card titulo="memórias de cálculo">
          <div className="space-y-2">
            <Botao
              variante="primario"
              className="w-full"
              onClick={() => setExercicio('galinha-homem')}
            >
              <FileText size={15} />
              Item (i) — Galinha vs Homem
            </Botao>
            <Botao
              variante="secundario"
              className="w-full"
              onClick={() => setExercicio('fig-1232')}
            >
              <FileText size={15} />
              Exercício extra — Fig. 12.32 (slide 34)
            </Botao>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-muted">
            O exercício do slide 34 usa uma rede maior (3-2-2) e demonstra que o
            mesmo motor de backpropagation generaliza para outras arquiteturas
            sem alterar uma linha de código.
          </p>
        </Card>
      </div>

      <div className="space-y-6">
        <CanvasPixels />
      </div>

      {exercicio && (
        <MemoriaCalculo
          traco={memoria.data}
          carregando={memoria.isPending}
          erro={memoria.error}
          onFechar={() => setExercicio(null)}
        />
      )}
    </div>
  )
}

/* ------------------------------------------- bonus: reconhecimento 8x8 ---- */
const GRADE_VAZIA = () =>
  Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => 0))

function CanvasPixels() {
  const [pixels, setPixels] = useState<number[][]>(GRADE_VAZIA)
  const [intensidade, setIntensidade] = useState(0.85)
  const [pintando, setPintando] = useState<number | null>(null)
  const [predicao, setPredicao] = useState<PredicaoImagem | null>(null)
  const debounce = useRef<number | null>(null)

  const padroes = useQuery({
    queryKey: ['lab5', 'imagem', 'padroes'],
    queryFn: api.lab5.padroesImagem,
  })

  const prever = useMutation({
    mutationFn: (grade: number[][]) => api.lab5.preverImagem(grade),
    onSuccess: setPredicao,
  })

  // Reavalia com um pequeno atraso para nao disparar uma chamada por pixel
  useEffect(() => {
    if (debounce.current) window.clearTimeout(debounce.current)
    debounce.current = window.setTimeout(() => prever.mutate(pixels), 120)
    return () => {
      if (debounce.current) window.clearTimeout(debounce.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pixels])

  const pintar = useCallback(
    (linha: number, coluna: number, valor: number) => {
      setPixels((p) => {
        if (p[linha][coluna] === valor) return p
        const novo = p.map((l) => [...l])
        novo[linha][coluna] = valor
        return novo
      })
    },
    [],
  )

  useEffect(() => {
    const soltar = () => setPintando(null)
    window.addEventListener('mouseup', soltar)
    return () => window.removeEventListener('mouseup', soltar)
  }, [])

  const out = predicao?.saida ?? 0
  const corSaida =
    predicao?.classe === 'homem'
      ? '#2563eb'
      : predicao?.classe === 'galinha'
        ? '#d97706'
        : 'hsl(var(--text-muted))'

  return (
    <Card titulo="bônus interativo · reconhecimento de imagem 8×8">
      <p className="mb-4 text-sm leading-relaxed text-secondary">
        O slide ilustra o problema original como o reconhecimento de uma imagem
        de <strong>8×8 pixels</strong> (64 entradas). Aqui uma segunda rede{' '}
        <strong>64 → 10 → 1</strong>, treinada do zero em Python puro apenas com
        os dois padrões de referência, classifica seu desenho em tempo real.
      </p>

      <div className="flex flex-wrap gap-6">
        {/* ---------------------------------------------------- canvas 8x8 */}
        <div className="shrink-0 space-y-3">
          <div
            className="grid select-none gap-[2px] rounded-lg border border-strong bg-white p-1.5"
            style={{ gridTemplateColumns: 'repeat(8, 34px)' }}
            onContextMenu={(e) => e.preventDefault()}
          >
            {pixels.map((linha, i) =>
              linha.map((valor, j) => (
                <button
                  key={`${i}-${j}`}
                  className="h-[34px] w-[34px] rounded-[3px] border border-zinc-200 transition-colors"
                  style={{ backgroundColor: tomCinza(valor) }}
                  onMouseDown={(e) => {
                    e.preventDefault()
                    const v = e.button === 2 ? 0 : intensidade
                    setPintando(v)
                    pintar(i, j, v)
                  }}
                  onMouseEnter={() => {
                    if (pintando !== null) pintar(i, j, pintando)
                  }}
                  aria-label={`Pixel linha ${i + 1}, coluna ${j + 1}`}
                />
              )),
            )}
          </div>

          <p className="text-center text-xs text-muted">
            botão esquerdo pinta · botão direito apaga
          </p>

          <div>
            <Slider
              rotulo="Intensidade do traço"
              valor={intensidade}
              onChange={setIntensidade}
              min={0.05}
              max={1}
              passo={0.05}
              formatar={(v) => `${Math.round(v * 100)}%`}
            />
            <div className="mt-1.5 flex items-center gap-2">
              <span
                className="h-4 w-4 rounded border border-strong"
                style={{ backgroundColor: tomCinza(intensidade) }}
              />
              <span className="text-xs text-muted">
                tom aplicado ao pintar
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <Botao
              tamanho="sm"
              onClick={() =>
                padroes.data && setPixels(padroes.data.homem.pixels.map((l) => [...l]))
              }
              disabled={!padroes.data}
            >
              Homem
            </Botao>
            <Botao
              tamanho="sm"
              onClick={() =>
                padroes.data &&
                setPixels(padroes.data.galinha.pixels.map((l) => [...l]))
              }
              disabled={!padroes.data}
            >
              Galinha
            </Botao>
            <Botao tamanho="sm" onClick={() => setPixels(GRADE_VAZIA())}>
              <Eraser size={13} />
              Limpar
            </Botao>
          </div>
        </div>

        {/* ------------------------------------------------ saída da rede */}
        <div className="min-w-[280px] flex-1 space-y-4">
          <div>
            <p className="kicker mb-2">
              saída da rede (0 = homem · 1 = galinha)
            </p>
            <div className="h-7 overflow-hidden rounded-lg border border-subtle bg-sunken">
              <div
                className="h-full transition-all duration-300"
                style={{
                  width: `${Math.max(1, out * 100)}%`,
                  backgroundColor: corSaida,
                }}
              />
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="tabular text-2xl font-semibold text-primary">
                {predicao?.vazio ? '—' : num(out, 4)}
              </span>
              <span
                className="text-sm font-semibold"
                style={{ color: corSaida }}
              >
                {predicao?.rotulo ?? 'Desenhe algo'}
              </span>
            </div>
          </div>

          <div>
            <p className="kicker mb-2">
              camada oculta · 10 neurônios (ativação sigmoide)
            </p>
            <div className="flex flex-wrap gap-1.5">
              {(predicao?.ativacoes_ocultas ?? Array(10).fill(0)).map((a, i) => (
                <div
                  key={i}
                  className="flex h-11 w-11 items-center justify-center rounded border border-subtle text-[9px] tabular"
                  style={{
                    backgroundColor: tomCinza(a),
                    color: a > 0.55 ? '#fff' : 'hsl(var(--text-muted))',
                  }}
                  title={`h${i + 1} = ${num(a, 4)}`}
                >
                  {num(a, 2)}
                </div>
              ))}
            </div>
            <Legenda
              className="mt-2"
              itens={[
                {
                  cor: tomCinza(0.1),
                  forma: 'quadrado',
                  rotulo: 'claro',
                  descricao: 'neurônio pouco ativo',
                },
                {
                  cor: tomCinza(0.9),
                  forma: 'quadrado',
                  rotulo: 'escuro',
                  descricao: 'neurônio muito ativo',
                },
              ]}
            />
          </div>

          {padroes.data && (
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-subtle bg-sunken px-3 py-2">
                <p className="text-xs text-muted">Padrão “Homem”</p>
                <p className="tabular text-sm font-semibold text-primary">
                  {num(padroes.data.homem.saida, 4)}
                </p>
              </div>
              <div className="rounded-lg border border-subtle bg-sunken px-3 py-2">
                <p className="text-xs text-muted">Padrão “Galinha”</p>
                <p className="tabular text-sm font-semibold text-primary">
                  {num(padroes.data.galinha.saida, 4)}
                </p>
              </div>
            </div>
          )}

          <Nota tom="info" titulo="Objetivo pedagógico">
            Como a rede foi treinada com apenas 2 exemplos, ela não generaliza no
            sentido estatístico — a ideia é deixar visível, em tempo real, como
            pequenas mudanças na entrada se propagam pela rede e alteram a saída.
          </Nota>
        </div>
      </div>
    </Card>
  )
}

/* ------------------------------------------------------------ item (ii) --- */
function PainelItemII() {
  const { config, set } = usarConfig({ atributos: 'todas' })
  const [camada, setCamada] = useState(8)
  const [selecionado, setSelecionado] = useState('mlp')

  const q = useQuery({
    queryKey: ['lab5', 'iris', config, camada],
    queryFn: () =>
      api.lab5.compararIris({ ...config, camada_oculta: camada, max_iter: 3000 }),
  })

  const d = q.data
  const rel = d?.relatorios[selecionado]

  return (
    <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
      <div className="space-y-5">
        <PainelConfig config={config} set={set}>
          <Slider
            rotulo="Neurônios na camada oculta"
            valor={camada}
            onChange={setCamada}
            min={2}
            max={32}
            passo={1}
          />
        </PainelConfig>

        <Card titulo="item (ii) do enunciado">
          <p className="text-sm leading-relaxed text-secondary">
            Classificar as 3 espécies do Iris com uma rede feedforward e comparar
            com o Classificador Ótimo de Bayes e o Naive Bayes, avaliando todas
            as métricas de qualidade.
          </p>
          <Nota tom="atencao" className="mt-3" titulo="Única exceção do projeto">
            O enunciado permite bibliotecas de ML <em>apenas</em> neste item. A
            rede usa <span className="font-mono">MLPClassifier</span> do
            scikit-learn, isolado em{' '}
            <span className="font-mono">models/mlp_sklearn.py</span> — todo o
            resto do projeto continua 100% Python puro.
          </Nota>
        </Card>
      </div>

      <div className="space-y-6">
        {q.isPending && (
          <Card>
            <Carregando texto="Treinando a rede e os classificadores bayesianos…" />
          </Card>
        )}
        {q.error && <ErroBox erro={q.error} />}

        {d && (
          <>
            <Card titulo="comparativo dos três modelos">
              <div className="grid gap-3 sm:grid-cols-3">
                {Object.entries(d.relatorios).map(([chave, r]) => (
                  <button
                    key={chave}
                    onClick={() => setSelecionado(chave)}
                    className={cn(
                      'rounded-lg border px-4 py-3 text-left transition-all',
                      selecionado === chave
                        ? 'border-accent-500 bg-accent-500/5 ring-1 ring-accent-500/30'
                        : 'border-subtle bg-sunken hover:border-strong',
                    )}
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
                        <span className="tabular text-sm text-secondary">
                          {num(r.kappa, 4)}
                        </span>
                        <span className="ml-1 text-[10px] text-muted">κ</span>
                      </span>
                    </div>
                  </button>
                ))}
              </div>
              <p className="mt-3 text-xs text-muted">
                Split: {d.n_treino} treino / {d.n_teste} teste · camada oculta com{' '}
                {d.config?.camada_oculta} neurônios · atributos:{' '}
                {d.config?.atributos}.
              </p>
            </Card>

            {rel && (
              <>
                <Card titulo={`métricas globais — ${rel.nome}`}>
                  <ResumoGlobal relatorio={rel} />
                </Card>

                <div className="grid gap-6 lg:grid-cols-2">
                  <Card titulo="matriz de confusão">
                    <MatrizConfusao relatorio={rel} classes={CLASSES} />
                  </Card>
                  <Card titulo="métricas por classe">
                    <TabelaPorClasse relatorio={rel} classes={CLASSES} />
                  </Card>
                </div>
              </>
            )}

            <Card titulo="teste Z de significância entre os pares">
              <TabelaTestesZ comparacoes={d.comparacoes} />
              <Nota tom="info" className="mt-4" titulo="Leitura do resultado">
                Mesmo quando a rede acerta mais que os classificadores
                bayesianos, o teste Z costuma indicar{' '}
                <strong>ausência de diferença significativa</strong> — o conjunto
                de teste é pequeno demais para distinguir com confiança um erro a
                mais ou a menos. É a lição central do Lab 3 aplicada aqui.
              </Nota>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
