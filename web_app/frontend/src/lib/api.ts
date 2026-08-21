/** Cliente HTTP tipado da API do projeto. */
import type { TracoGenerico } from '@/components/MemoriaGenerica'
import type {
  AnaliseArquivo,
  ComparacaoModelos,
  DatasetEnviado,
  Estatisticas,
  ModeloCatalogo,
  OpcoesLeitura,
  RespostaClassificacao,
  RespostaPredicaoModelo,
  ResultadoImportacao,
  ValorParametro,
  EstadoXor,
  ListaExercicios,
  Metadata,
  PadroesImagem,
  PontoCurvaKappa,
  PredicaoImagem,
  Relatorio,
  RespostaAmostras,
  RespostaBayes,
  RespostaBinario,
  RespostaDistanciaMinima,
  RespostaNormalidade,
  RespostaOva,
  RespostaPredicao,
  RespostaRegioes,
  RespostaXorDelta,
  ListaClassificadores,
  MatrizSignificancia,
  PredicaoFloresta,
  RespostaArvore,
  RespostaFloresta,
  RespostaSignificancia,
  Traco,
  Trajetoria,
  ValidacaoCruzada,
  ValidacaoFloresta,
} from './types'

export class ErroApi extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ErroApi'
  }
}

type Params = Record<string, string | number | boolean | undefined>

function querystring(params?: Params): string {
  if (!params) return ''
  const sp = new URLSearchParams()
  for (const [chave, valor] of Object.entries(params)) {
    if (valor !== undefined) sp.set(chave, String(valor))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

async function tratar<T>(resposta: Response): Promise<T> {
  if (!resposta.ok) {
    let detalhe = `Erro ${resposta.status}`
    try {
      const corpo = await resposta.json()
      if (corpo?.detail) {
        detalhe =
          typeof corpo.detail === 'string'
            ? corpo.detail
            : JSON.stringify(corpo.detail)
      }
    } catch {
      /* resposta sem corpo JSON — mantem a mensagem padrao */
    }
    throw new ErroApi(detalhe, resposta.status)
  }
  return resposta.json() as Promise<T>
}

async function get<T>(caminho: string, params?: Params): Promise<T> {
  return tratar<T>(await fetch(`/api${caminho}${querystring(params)}`))
}

async function corpoJson<T>(
  metodo: 'POST' | 'PATCH',
  caminho: string,
  corpo: unknown,
): Promise<T> {
  return tratar<T>(
    await fetch(`/api${caminho}`, {
      method: metodo,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(corpo),
    }),
  )
}

async function post<T>(caminho: string, corpo: unknown): Promise<T> {
  return corpoJson<T>('POST', caminho, corpo)
}

async function patch<T>(caminho: string, corpo: unknown): Promise<T> {
  return corpoJson<T>('PATCH', caminho, corpo)
}

async function del<T>(caminho: string): Promise<T> {
  return tratar<T>(await fetch(`/api${caminho}`, { method: 'DELETE' }))
}

/** Parametros comuns a quase todos os experimentos. */
export interface ParamsBase extends Params {
  dataset?: string
  atributos?: string
  proporcao?: number
}

/** Parametros da floresta aleatoria. */
export interface ParamsFloresta extends ParamsBase {
  n_arvores?: number
  criterio?: string
  profundidade_max?: number
  max_atributos?: string
  min_amostras_folha?: number
}

/** Corpo comum das rotas de classificacao (modelo + hiperparametros). */
export interface CorpoClassificar {
  dataset: string
  atributos: string
  proporcao: number
  modelo: string
  parametros: Record<string, ValorParametro>
}

/** Rede montada na interface do construtor. */
export interface RedeCustom {
  pesos_oculta: number[][]
  bias_oculta: number[]
  pesos_saida: number[][]
  bias_saida: number[]
  padroes: { entrada: number[]; alvo: number[] }[]
  taxa: number
  epocas: number
  n_snapshots?: number
  rotulos_entrada?: string[]
  rotulos_ocultos?: string[]
  rotulos_saida?: string[]
}

export const api = {
  health: () => get<{ status: string; sklearn: boolean }>('/health'),

  dataset: {
    metadata: () => get<Metadata>('/dataset/metadata'),
    amostras: (p: ParamsBase) => get<RespostaAmostras>('/dataset/amostras', p),
    estatisticas: (p: { dataset?: string }) =>
      get<Estatisticas>('/dataset/estatisticas', p),

    // --- importacao da base do usuario (.txt) ---
    opcoesLeitura: () => get<OpcoesLeitura>('/dataset/opcoes-leitura'),
    /** Pre-visualiza o arquivo sem importa-lo. */
    analisar: (corpo: {
      conteudo: string
      delimitador?: string
      cabecalho?: string
    }) => post<AnaliseArquivo>('/dataset/analisar', corpo),
    importar: (corpo: {
      conteudo: string
      nome: string
      arquivo_original?: string
      delimitador?: string
      cabecalho?: string
      coluna_classe?: number
      colunas_ignoradas?: number[]
    }) => post<ResultadoImportacao>('/dataset/importar', corpo),
    enviados: () => get<{ datasets: DatasetEnviado[] }>('/dataset/enviados'),
    renomear: (id: string, nome: string) =>
      patch<{ id: string; nome: string }>(`/dataset/enviados/${id}`, { nome }),
    remover: (id: string) =>
      del<{ removido: string }>(`/dataset/enviados/${id}`),
  },

  /**
   * Tela de classificacao: escolha do modelo + parametrizacao.
   * O corpo e sempre JSON porque `parametros` muda de forma conforme o modelo.
   */
  classificar: {
    modelos: () =>
      get<{ modelos: ModeloCatalogo[]; ordem: string[] }>(
        '/classificar/modelos',
      ),
    treinar: (corpo: CorpoClassificar) =>
      post<RespostaClassificacao>('/classificar/treinar', corpo),
    regioes: (corpo: CorpoClassificar & { resolucao?: number }) =>
      post<RespostaRegioes>('/classificar/regioes', corpo),
    predizer: (corpo: CorpoClassificar & { valores: number[] }) =>
      post<RespostaPredicaoModelo>('/classificar/predizer', corpo),
  },

  distanciaMinima: {
    treinar: (p: ParamsBase) =>
      get<RespostaDistanciaMinima>('/distancia-minima/treinar', p),
    memoria: (p: ParamsBase & { x1?: number; x2?: number }) =>
      get<TracoGenerico>('/distancia-minima/memoria', p),
    regioes: (p: ParamsBase & { resolucao?: number }) =>
      get<RespostaRegioes>('/distancia-minima/regioes', p),
    predizer: (corpo: {
      dataset: string
      atributos: string
      proporcao?: number
      valores: number[]
    }) => post<RespostaPredicao>('/distancia-minima/predizer', corpo),
  },

  perceptronDelta: {
    binario: (
      p: ParamsBase & {
        algoritmo: 'perceptron' | 'delta'
        classe_pos?: string
        classe_neg?: string
        taxa?: number
        max_epocas?: number
      },
    ) => get<RespostaBinario>('/perceptron-delta/binario', p),
    ova: (p: ParamsBase & { taxa?: number; max_epocas?: number }) =>
      get<RespostaOva>('/perceptron-delta/ova', p),
    xor: (p: { taxa?: number; max_epocas?: number }) =>
      get<RespostaXorDelta>('/perceptron-delta/xor', p),
    memoria: (
      p: ParamsBase & {
        algoritmo: 'perceptron' | 'delta'
        classe_pos?: string
        classe_neg?: string
        taxa?: number
        max_epocas?: number
      },
    ) => get<TracoGenerico>('/perceptron-delta/memoria', p),
  },

  metricas: {
    compararModelos: (p: ParamsBase) =>
      get<ComparacaoModelos>('/metricas/comparar-modelos', p),
    avaliar: (corpo: {
      matriz: Record<string, Record<string, number>>
      classes?: string[]
      nome?: string
    }) =>
      post<{
        relatorio: Relatorio
        total_amostras: number
        classes: string[]
      }>('/metricas/avaliar', corpo),
    compararMatrizes: (corpo: {
      matriz_a: Record<string, Record<string, number>>
      matriz_b: Record<string, Record<string, number>>
      nome_a?: string
      nome_b?: string
    }) =>
      post<{
        a: Relatorio
        b: Relatorio
        kappa: { z: number; p: number; significativo: boolean }
        tau: { z: number; p: number; significativo: boolean }
      }>('/metricas/comparar-matrizes', corpo),
    memoria: (corpo: {
      matriz: Record<string, Record<string, number>>
      classes?: string[]
      nome?: string
    }) => post<TracoGenerico>('/metricas/memoria', corpo),
    simular: (p: { acerto?: number; n_por_classe?: number }) =>
      get<{ relatorio: Relatorio; acerto_alvo: number; n_por_classe: number }>(
        '/metricas/simular',
        p,
      ),
    validacaoCruzada: (
      p: ParamsBase & { k?: number; repeticoes?: number },
    ) => get<ValidacaoCruzada>('/metricas/validacao-cruzada', p),
    classificadores: () => get<ListaClassificadores>('/metricas/classificadores'),
    significancia: (
      p: ParamsBase & {
        modelo_a: string
        modelo_b: string
        metrica: string
        n_reamostragens?: number
        n_permutacoes?: number
      },
    ) => get<RespostaSignificancia>('/metricas/significancia', p),
    matrizSignificancia: (
      p: ParamsBase & { metrica: string; n_reamostragens?: number },
    ) => get<MatrizSignificancia>('/metricas/significancia/matriz', p),
    memoriaSignificancia: (
      p: ParamsBase & {
        modelo_a: string
        modelo_b: string
        metrica: string
        n_reamostragens?: number
        n_permutacoes?: number
      },
    ) => get<TracoGenerico>('/metricas/significancia/memoria', p),
    curvaKappa: (p: { n_por_classe?: number; passos?: number }) =>
      get<{ pontos: PontoCurvaKappa[] }>('/metricas/curva-kappa', p),
  },

  bayes: {
    treinar: (p: ParamsBase) => get<RespostaBayes>('/bayes/treinar', p),
    regioes: (
      p: ParamsBase & { classificador?: 'bayes' | 'naive'; resolucao?: number },
    ) => get<RespostaRegioes>('/bayes/regioes', p),
    memoria: (p: ParamsBase & { classificador?: 'bayes' | 'naive' }) =>
      get<TracoGenerico>('/bayes/memoria', p),
    normalidade: (p: { dataset?: string; atributos?: string }) =>
      get<RespostaNormalidade>('/bayes/normalidade', p),
    predizer: (corpo: {
      dataset: string
      atributos: string
      naive: boolean
      valores: number[]
    }) => post<RespostaPredicao>('/bayes/predizer', corpo),
  },

  lab5: {
    exercicios: () => get<ListaExercicios>('/lab5/exercicios'),
    memoria: (id: string) => get<Traco>(`/lab5/memoria/${id}`),
    xorInicial: (p: { resolucao?: number }) =>
      get<EstadoXor>('/lab5/xor/inicial', p),
    xorTreinar: (corpo: {
      epocas: number
      taxa: number
      resolucao?: number
      pesos_oculta?: number[][]
      bias_oculta?: number[]
      pesos_saida?: number[][]
      bias_saida?: number[]
    }) => post<EstadoXor>('/lab5/xor/treinar', corpo),
    trajetoria: (
      exercicio: string,
      corpo: {
        epocas: number
        taxa?: number
        n_snapshots?: number
        pesos_oculta?: number[][]
        bias_oculta?: number[]
        pesos_saida?: number[][]
        bias_saida?: number[]
      },
    ) => post<Trajetoria>(`/lab5/trajetoria/${exercicio}`, corpo),
    redeTrajetoria: (corpo: RedeCustom) =>
      post<Trajetoria>('/lab5/rede/trajetoria', corpo),
    redeMemoria: (corpo: RedeCustom) => post<Traco>('/lab5/rede/memoria', corpo),
    padroesImagem: () => get<PadroesImagem>('/lab5/imagem/padroes'),
    preverImagem: (pixels: number[][]) =>
      post<PredicaoImagem>('/lab5/imagem/prever', { pixels }),
    compararIris: (
      p: ParamsBase & { camada_oculta?: number; max_iter?: number },
    ) => get<ComparacaoModelos>('/lab5/iris/comparar', p),
  },

  floresta: {
    treinar: (p: ParamsFloresta) => get<RespostaFloresta>('/floresta/treinar', p),
    arvore: (indice: number, p: ParamsFloresta) =>
      get<RespostaArvore>(`/floresta/arvore/${indice}`, p),
    regioes: (p: ParamsFloresta & { resolucao?: number }) =>
      get<RespostaRegioes & { confianca: number[][] }>('/floresta/regioes', p),
    // POST: corpo JSON, sem a assinatura de indice dos query params
    predizer: (corpo: {
      dataset?: string
      atributos?: string
      proporcao?: number
      n_arvores?: number
      criterio?: string
      profundidade_max?: number
      max_atributos?: string
      valores: number[]
    }) => post<PredicaoFloresta>('/floresta/predizer', corpo),
    validacaoCruzada: (p: ParamsFloresta & { k?: number; repeticoes?: number }) =>
      get<ValidacaoFloresta>('/floresta/validacao-cruzada', p),
    memoria: (p: ParamsFloresta) => get<TracoGenerico>('/floresta/memoria', p),
  },
}
