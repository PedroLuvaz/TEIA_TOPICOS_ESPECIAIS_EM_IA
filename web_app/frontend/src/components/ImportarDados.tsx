/**
 * Importacao da base de dados do usuario (.txt).
 *
 * Fluxo em tres passos, todos na mesma tela:
 *   1. escolher o arquivo — o conteudo e lido no navegador e enviado como
 *      texto para `POST /api/dataset/analisar`;
 *   2. conferir a leitura — delimitador, cabecalho, coluna de classe e quais
 *      colunas ignorar, com a previa das primeiras linhas ao lado;
 *   3. importar — a base entra no seletor "Base de dados" de TODAS as telas.
 *
 * O arquivo nunca sai do computador do usuario para lugar nenhum alem do
 * proprio backend local do aplicativo.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, FileText, FileUp, Trash2, TriangleAlert } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api, ErroApi } from '@/lib/api'
import type { AnaliseArquivo, ExemploTxt } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Badge, Botao, Carregando, Nota, Select } from './ui'

const EXTENSOES = '.txt,.csv,.tsv,.data,text/plain'

interface Props {
  /** Chamado com o id da base recem-importada, para o painel ja seleciona-la. */
  aoImportar: (id: string) => void
}

export function ImportarDados({ aoImportar }: Props) {
  const qc = useQueryClient()
  const inputArquivo = useRef<HTMLInputElement>(null)

  const [conteudo, setConteudo] = useState('')
  const [nomeArquivo, setNomeArquivo] = useState('')
  const [nome, setNome] = useState('')
  const [delimitador, setDelimitador] = useState('auto')
  const [cabecalho, setCabecalho] = useState('auto')
  const [colunaClasse, setColunaClasse] = useState<number | null>(null)
  const [ignoradas, setIgnoradas] = useState<number[]>([])
  const [erroArquivo, setErroArquivo] = useState('')
  // Nome do arquivo de exemplo em uso — vazio quando o arquivo veio do
  // computador do usuario. So serve para destacar o botao correspondente.
  const [exemploAtivo, setExemploAtivo] = useState('')

  const { data: opcoes } = useQuery({
    queryKey: ['opcoes-leitura'],
    queryFn: api.dataset.opcoesLeitura,
    staleTime: Infinity,
  })

  const { data: enviados } = useQuery({
    queryKey: ['datasets-enviados'],
    queryFn: api.dataset.enviados,
  })

  const { data: exemplos } = useQuery({
    queryKey: ['exemplos-txt'],
    queryFn: api.dataset.exemplos,
    staleTime: Infinity,
  })

  /** Analise: roda ao escolher o arquivo e a cada troca de delimitador/cabecalho. */
  const analise = useMutation({
    mutationFn: (p: { conteudo: string; delimitador: string; cabecalho: string }) =>
      api.dataset.analisar(p),
    onSuccess: (dados: AnaliseArquivo) => {
      // A sugestao so vale como ponto de partida: se o usuario ja escolheu uma
      // coluna, a escolha dele manda — salvo quando ela deixou de existir.
      setColunaClasse((atual) =>
        atual === null || atual >= dados.n_colunas
          ? dados.coluna_classe_sugerida
          : atual,
      )
    },
  })

  const importar = useMutation({
    mutationFn: () =>
      api.dataset.importar({
        conteudo,
        nome: nome.trim() || nomeArquivo || 'Base do usuário',
        arquivo_original: nomeArquivo,
        delimitador,
        cabecalho,
        coluna_classe: colunaClasse ?? undefined,
        colunas_ignoradas: ignoradas,
      }),
    onSuccess: async (res) => {
      // O seletor de datasets so enxerga a base nova depois que o metadata
      // volta do servidor — por isso o await antes de selecionar.
      await qc.invalidateQueries({ queryKey: ['metadata'] })
      await qc.invalidateQueries({ queryKey: ['datasets-enviados'] })
      aoImportar(res.id)
      limpar()
    },
  })

  const remover = useMutation({
    mutationFn: (id: string) => api.dataset.remover(id),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['metadata'] })
      await qc.invalidateQueries({ queryKey: ['datasets-enviados'] })
    },
  })

  function limpar() {
    setConteudo('')
    setNomeArquivo('')
    setNome('')
    setDelimitador('auto')
    setCabecalho('auto')
    setColunaClasse(null)
    setIgnoradas([])
    setErroArquivo('')
    setExemploAtivo('')
    analise.reset()
    if (inputArquivo.current) inputArquivo.current.value = ''
  }

  /**
   * Ponto unico por onde todo texto entra na tela — venha do computador do
   * usuario ou de um exemplo. Manter os dois no mesmo caminho garante que a
   * previa do exemplo seja exatamente a de um arquivo importado de verdade.
   */
  function usarConteudo(texto: string, arquivo: string, doExemplo: string) {
    setConteudo(texto)
    setNomeArquivo(arquivo)
    setNome(arquivo.replace(/\.[^.]+$/, '') + (doExemplo ? ' (exemplo)' : ''))
    setExemploAtivo(doExemplo)
    setDelimitador('auto')
    setCabecalho('auto')
    setColunaClasse(null)
    setIgnoradas([])
    setErroArquivo('')
    analise.mutate({ conteudo: texto, delimitador: 'auto', cabecalho: 'auto' })
  }

  const exemplo = useMutation({
    mutationFn: (arquivo: string) => api.dataset.exemplo(arquivo),
    onSuccess: (res) => usarConteudo(res.conteudo, res.arquivo, res.arquivo),
    onError: () =>
      setErroArquivo('Não foi possível carregar o exemplo. Confira se a pasta '
        + 'data/exemplos/ continua no lugar.'),
  })

  // A tela abre ja com um exemplo lido, para quem chega nela ver na hora o
  // formato de entrada esperado. Roda uma unica vez: depois de "Cancelar" ou
  // de escolher um arquivo, a decisao passa a ser do usuario.
  const jaAbriuComExemplo = useRef(false)
  useEffect(() => {
    if (jaAbriuComExemplo.current || conteudo) return
    const padrao = exemplos?.exemplos.find((e) => e.padrao)
    if (!padrao) return
    jaAbriuComExemplo.current = true
    exemplo.mutate(padrao.arquivo)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exemplos])

  async function aoEscolherArquivo(arquivo: File | undefined) {
    if (!arquivo) return
    setErroArquivo('')
    let texto: string
    try {
      texto = await arquivo.text()
    } catch {
      setErroArquivo('Não foi possível ler o arquivo escolhido.')
      return
    }
    if (!texto.trim()) {
      setErroArquivo('O arquivo está vazio.')
      return
    }
    // Um arquivo escolhido a mao substitui o exemplo que abriu a tela.
    jaAbriuComExemplo.current = true
    usarConteudo(texto, arquivo.name, '')
  }

  function reanalisar(novoDelimitador: string, novoCabecalho: string) {
    setDelimitador(novoDelimitador)
    setCabecalho(novoCabecalho)
    if (conteudo) {
      analise.mutate({
        conteudo,
        delimitador: novoDelimitador,
        cabecalho: novoCabecalho,
      })
    }
  }

  function alternarIgnorada(indice: number) {
    setIgnoradas((atual) =>
      atual.includes(indice)
        ? atual.filter((i) => i !== indice)
        : [...atual, indice],
    )
  }

  const dados = analise.data
  const nAtributos = dados
    ? dados.n_colunas - 1 - ignoradas.filter((i) => i !== colunaClasse).length
    : 0

  return (
    <div className="space-y-4">
      <Nota tom="info" titulo="Base de dados do usuário">
        Envie um arquivo <strong>.txt</strong> (ou .csv/.tsv) com uma amostra
        por linha: as colunas de atributos e uma coluna com a classe. O
        delimitador e o cabeçalho são detectados automaticamente, e você
        confere tudo antes de importar. Exemplos prontos em{' '}
        <code className="tabular text-xs">data/exemplos/</code>.
      </Nota>

      {/* ------------------------------------------------ 1. escolher arquivo */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={inputArquivo}
          type="file"
          accept={EXTENSOES}
          className="hidden"
          onChange={(e) => aoEscolherArquivo(e.target.files?.[0])}
        />
        <Botao
          variante="primario"
          tamanho="sm"
          onClick={() => inputArquivo.current?.click()}
        >
          <FileUp size={15} />
          Escolher arquivo
        </Botao>
        {nomeArquivo && (
          <>
            <span className="tabular text-xs text-secondary">{nomeArquivo}</span>
            <Botao tamanho="sm" variante="fantasma" onClick={limpar}>
              Cancelar
            </Botao>
          </>
        )}
      </div>

      {/* ------------------------------------------- 1b. exemplos prontos */}
      {!!exemplos?.exemplos.length && (
        <Exemplos
          lista={exemplos.exemplos}
          ativo={exemploAtivo}
          carregando={exemplo.isPending ? exemplo.variables ?? '' : ''}
          aoEscolher={(arquivo) => {
            jaAbriuComExemplo.current = true
            exemplo.mutate(arquivo)
          }}
        />
      )}

      {erroArquivo && <Nota tom="atencao">{erroArquivo}</Nota>}
      {analise.isPending && <Carregando texto="Analisando o arquivo…" />}
      {analise.isError && (
        <Nota tom="atencao" titulo="Não foi possível ler o arquivo">
          {analise.error instanceof ErroApi
            ? analise.error.message
            : 'Erro desconhecido.'}
        </Nota>
      )}

      {/* -------------------------------------------------- 2. conferir leitura */}
      {dados && !analise.isPending && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <Select
              rotulo="Delimitador"
              valor={delimitador}
              onChange={(v) => reanalisar(v, cabecalho)}
              opcoes={[
                {
                  valor: 'auto',
                  rotulo: `Detectar (${dados.delimitador_rotulo})`,
                },
                ...(opcoes?.delimitadores.map((d) => ({
                  valor: d.id,
                  rotulo: d.nome,
                })) ?? []),
              ]}
            />
            <Select
              rotulo="Primeira linha"
              valor={cabecalho}
              onChange={(v) => reanalisar(delimitador, v)}
              opcoes={[
                {
                  valor: 'auto',
                  rotulo: `Detectar (${dados.cabecalho ? 'cabeçalho' : 'dados'})`,
                },
                { valor: 'sim', rotulo: 'É cabeçalho' },
                { valor: 'nao', rotulo: 'Já são dados' },
              ]}
            />
            <Select
              rotulo="Coluna da classe"
              valor={String(colunaClasse ?? dados.coluna_classe_sugerida)}
              onChange={(v) => setColunaClasse(Number(v))}
              opcoes={dados.colunas.map((c) => ({
                valor: String(c.indice),
                rotulo: `${c.nome} (${c.n_distintos} valores)`,
              }))}
            />
          </div>

          <label className="block">
            <span className="kicker mb-1.5 block text-muted">
              Nome da base no aplicativo
            </span>
            <input
              value={nome}
              onChange={(e) => setNome(e.target.value)}
              placeholder="Ex.: Vinhos (UCI)"
              className={cn(
                'w-full rounded-lg border border-strong bg-surface px-3 py-2',
                'text-sm text-primary transition-colors',
                'hover:border-accent-500/50 focus:border-accent-500 focus:outline-none',
              )}
            />
          </label>

          <TabelaColunas
            dados={dados}
            colunaClasse={colunaClasse ?? dados.coluna_classe_sugerida}
            ignoradas={ignoradas}
            alternar={alternarIgnorada}
          />

          <Previa dados={dados} colunaClasse={colunaClasse ?? dados.coluna_classe_sugerida} ignoradas={ignoradas} />

          {nAtributos < 2 && (
            <Nota tom="atencao" titulo="Atributos insuficientes">
              Restaram {nAtributos} atributo(s). A aplicação precisa de pelo
              menos dois — os gráficos de dispersão e as superfícies de decisão
              são desenhados em 2D.
            </Nota>
          )}

          {importar.isError && (
            <Nota tom="atencao" titulo="A importação falhou">
              {importar.error instanceof ErroApi
                ? importar.error.message
                : 'Erro desconhecido.'}
            </Nota>
          )}

          <div className="flex items-center gap-3">
            <Botao
              variante="primario"
              onClick={() => importar.mutate()}
              disabled={importar.isPending || nAtributos < 2}
            >
              <Check size={15} />
              {importar.isPending ? 'Importando…' : 'Importar esta base'}
            </Botao>
            <span className="text-xs text-muted">
              {dados.n_linhas} linhas · {nAtributos} atributos ·{' '}
              {dados.colunas[colunaClasse ?? dados.coluna_classe_sugerida]
                ?.n_distintos ?? 0}{' '}
              classes
            </span>
          </div>
        </div>
      )}

      {/* ------------------------------------------------- 3. bases ja enviadas */}
      {!!enviados?.datasets.length && (
        <div className="border-t border-subtle pt-4">
          <p className="kicker mb-2 text-muted">Bases já importadas</p>
          <ul className="space-y-2">
            {enviados.datasets.map((d) => (
              <li
                key={d.id}
                className="flex items-start justify-between gap-3 rounded-lg border border-subtle px-3 py-2"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-primary">
                    {d.nome}
                  </p>
                  <p className="text-xs text-muted">
                    {d.n_amostras} amostras · {d.features.length} atributos ·{' '}
                    {d.classes.length} classes · classe em “{d.coluna_classe}”
                  </p>
                  {!!d.avisos.length && (
                    <p className="mt-1 flex items-start gap-1 text-xs text-accent-700 dark:text-accent-400">
                      <TriangleAlert size={13} className="mt-0.5 shrink-0" />
                      {d.avisos.join(' ')}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => remover.mutate(d.id)}
                  disabled={remover.isPending}
                  aria-label={`Remover ${d.nome}`}
                  className="shrink-0 rounded-md p-1.5 text-muted transition-colors hover:bg-rose-500/10 hover:text-rose-500"
                >
                  <Trash2 size={15} />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------- exemplos */
/**
 * Exemplos prontos: um clique carrega o arquivo no mesmo fluxo de analise, e
 * o texto cru das primeiras linhas fica a vista para o formato de entrada
 * ficar obvio antes de o usuario procurar a base dele.
 */
function Exemplos({
  lista,
  ativo,
  carregando,
  aoEscolher,
}: {
  lista: ExemploTxt[]
  ativo: string
  carregando: string
  aoEscolher: (arquivo: string) => void
}) {
  const emFoco = lista.find((e) => e.arquivo === ativo) ?? lista[0]
  const restantes = emFoco ? emFoco.n_linhas - emFoco.n_amostra : 0

  return (
    <div className="rounded-lg border border-subtle bg-zinc-500/5 p-3">
      <p className="kicker mb-2 text-muted">
        Ou carregue um exemplo — para ver o formato de entrada
      </p>
      <div className="flex flex-wrap gap-2">
        {lista.map((e) => (
          <Botao
            key={e.arquivo}
            tamanho="sm"
            variante={e.arquivo === ativo ? 'secundario' : 'fantasma'}
            disabled={!!carregando}
            onClick={() => aoEscolher(e.arquivo)}
          >
            <FileText size={14} />
            {carregando === e.arquivo ? 'Carregando…' : e.nome}
          </Botao>
        ))}
      </div>

      {emFoco && (
        <div className="mt-3">
          <p className="text-xs text-secondary">{emFoco.descricao}</p>
          <div className="mt-2 overflow-x-auto rounded-md border border-subtle bg-surface">
            <pre className="tabular px-3 py-2 text-xs whitespace-pre text-secondary">
              {emFoco.amostra}
            </pre>
            {restantes > 0 && (
              <p className="border-t border-subtle px-3 py-1 text-xs text-muted">
                … e mais {restantes} linha(s)
              </p>
            )}
          </div>
          <p className="mt-1.5 text-xs text-muted">
            <code className="tabular">data/exemplos/{emFoco.arquivo}</code> ·{' '}
            {emFoco.n_linhas} linhas
          </p>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ colunas */
function TabelaColunas({
  dados,
  colunaClasse,
  ignoradas,
  alternar,
}: {
  dados: AnaliseArquivo
  colunaClasse: number
  ignoradas: number[]
  alternar: (i: number) => void
}) {
  return (
    <div>
      <p className="kicker mb-2 text-muted">
        Colunas — desmarque as que não devem virar atributo (ids, códigos
        redundantes)
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        {dados.colunas.map((c) => {
          const eClasse = c.indice === colunaClasse
          const ignorada = ignoradas.includes(c.indice)
          return (
            <label
              key={c.indice}
              className={cn(
                'flex items-start gap-2 rounded-lg border px-3 py-2 text-sm',
                eClasse
                  ? 'border-accent-500/40 bg-accent-500/5'
                  : 'border-subtle',
                !eClasse && 'cursor-pointer hover:border-accent-500/40',
              )}
            >
              <input
                type="checkbox"
                className="mt-1 accent-accent-500"
                checked={eClasse ? true : !ignorada}
                disabled={eClasse}
                onChange={() => alternar(c.indice)}
              />
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="truncate font-medium text-primary">
                    {c.nome}
                  </span>
                  {eClasse ? (
                    <Badge tom="medio">classe</Badge>
                  ) : (
                    <Badge tom={c.tipo === 'numerico' ? 'info' : 'neutro'}>
                      {c.tipo === 'numerico' ? 'numérico' : 'categórico'}
                    </Badge>
                  )}
                </span>
                <span className="mt-0.5 block truncate text-xs text-muted">
                  {c.n_distintos} valores
                  {c.n_ausentes > 0 && ` · ${c.n_ausentes} ausentes`}
                  {c.exemplos.length > 0 && ` · ex.: ${c.exemplos.slice(0, 3).join(', ')}`}
                </span>
              </span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------- previa */
function Previa({
  dados,
  colunaClasse,
  ignoradas,
}: {
  dados: AnaliseArquivo
  colunaClasse: number
  ignoradas: number[]
}) {
  return (
    <div>
      <p className="kicker mb-2 text-muted">
        Prévia — primeiras {dados.previa.length} linhas de dados
      </p>
      <div className="overflow-x-auto rounded-lg border border-subtle">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-subtle bg-zinc-500/5">
              {dados.colunas.map((c) => (
                <th
                  key={c.indice}
                  className={cn(
                    'px-2 py-1.5 text-left font-semibold whitespace-nowrap',
                    c.indice === colunaClasse
                      ? 'text-accent-700 dark:text-accent-400'
                      : ignoradas.includes(c.indice)
                        ? 'text-muted line-through'
                        : 'text-secondary',
                  )}
                >
                  {c.nome}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dados.previa.map((linha, i) => (
              <tr key={i} className="border-b border-subtle last:border-0">
                {linha.map((campo, j) => (
                  <td
                    key={j}
                    className={cn(
                      'tabular px-2 py-1 whitespace-nowrap',
                      j === colunaClasse
                        ? 'font-medium text-accent-700 dark:text-accent-400'
                        : ignoradas.includes(j)
                          ? 'text-muted line-through'
                          : 'text-secondary',
                    )}
                  >
                    {campo}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
