/** Tipos espelhando os payloads da API FastAPI (`web_app/backend`). */

/**
 * Nome de uma classe. Era uma uniao literal das 3 do Iris; agora e livre,
 * porque cada dataset declara as suas (o do seminario tem 4, em portugues).
 * O alias foi mantido para nao renomear o tipo em dezenas de arquivos.
 */
export type ClasseIris = string

/** Chave de um conjunto de atributos — tambem definida por dataset. */
export type ChaveAtributos = string

/* ------------------------------------------------------------------ dataset */
export interface ConfigAtributo {
  id: ChaveAtributos
  nome: string
  indices: number[]
  eixo_x: string
  eixo_y: string
}

export interface ParClasses {
  pos: ClasseIris
  neg: ClasseIris
}

export interface DatasetInfo {
  id: string
  nome: string
  descricao: string
  tipo: 'continuo' | 'categorico'
  disponivel: boolean
  n_amostras: number
  classes: ClasseIris[]
  features: string[]
  atributos: ConfigAtributo[]
  atributos_padrao: ChaveAtributos
  pares: ParClasses[]
  /** Rotulos dos valores categoricos por indice de atributo (null se continuo). */
  valores: Record<string, string[]> | null
}

export interface Metadata {
  datasets: DatasetInfo[]
  dataset_padrao: string
  /** Atalhos do dataset padrao — usados onde nao ha selecao (Lab 5). */
  atributos: ConfigAtributo[]
  classes: ClasseIris[]
  features: string[]
  pares: ParClasses[]
}

export interface Amostra {
  x: number
  y: number
  classe: ClasseIris
  treino: boolean
  atributos: number[]
}

export interface RespostaAmostras {
  amostras: Amostra[]
  total: number
  n_treino: number
  n_teste: number
  eixo_x: string
  eixo_y: string
}

export interface Estatisticas {
  por_classe: Record<
    string,
    {
      n: number
      features: {
        feature: string
        media: number
        desvio: number
        minimo: number
        maximo: number
      }[]
    }
  >
  features: string[]
}

/* ------------------------------------------------------------------ metricas */
export interface MetricasClasse {
  acuracia_produtor: number
  acuracia_usuario: number
  sensibilidade: number
  especificidade: number
  precisao: number
  f1: number
  f2: number
  mcc: number
  vp: number
  fp: number
  fn: number
  vn: number
}

export interface Relatorio {
  nome: string
  matriz: Record<string, Record<string, number>>
  acerto_global: number
  kappa: number
  variancia_kappa: number
  tau: number
  variancia_tau: number
  por_classe: Record<string, MetricasClasse>
}

export interface Comparacao {
  a: string
  b: string
  nome_a: string
  nome_b: string
  z: number
  p: number
  significativo: boolean
}

export interface Limites {
  x_min: number
  x_max: number
  y_min: number
  y_max: number
}

/* --------------------------------------------------------- distancia minima */
export interface Fronteira {
  classe_i: ClasseIris
  classe_j: ClasseIris
  w: number[]
  b: number
  equacao: string
}

export interface RespostaDistanciaMinima {
  prototipos: Record<string, number[]>
  prototipos_plot: Record<string, { x: number; y: number }>
  relatorio: Relatorio
  fronteiras: Fronteira[]
  amostras: Amostra[]
  n_treino: number
  n_teste: number
  eixo_x: string
  eixo_y: string
  dimensoes: number
}

export interface RespostaRegioes {
  grade: number[][]
  eixo_x: number[]
  eixo_y: number[]
  limites: Limites
  classes: ClasseIris[]
  superficies?: Record<string, number[][]>
}

export interface RespostaPredicao {
  classe: ClasseIris
  scores: Record<string, number>
  distancias?: Record<string, number>
  mahalanobis?: Record<string, number>
  valores: number[]
}

/* --------------------------------------------------------- perceptron/delta */
export interface RespostaBinario {
  algoritmo: 'perceptron' | 'delta'
  pesos: number[]
  historico: number[]
  rotulo_historico: 'erros' | 'mse'
  epocas: number
  convergiu: boolean
  acuracia_treino: number
  acuracia_teste: number
  relatorio: Relatorio
  amostras: Amostra[]
  limites: Limites
  classe_pos: ClasseIris
  classe_neg: ClasseIris
  eixo_x: string
  eixo_y: string
  bidimensional: boolean
}

export interface RespostaOva {
  pesos: Record<string, number[]>
  historico: Record<string, number[]>
  epocas: number
  relatorio: Relatorio
  amostras: Amostra[]
  limites: Limites
  eixo_x: string
  eixo_y: string
  bidimensional: boolean
}

export interface RespostaXorDelta {
  pesos: number[]
  historico: number[]
  mse_final: number | null
  mse_teorico: number
  padroes: {
    x1: number
    x2: number
    alvo: number
    net: number
    previsto: number
    correto: boolean
  }[]
  acertos: number
}

/* ------------------------------------------------------------------- bayes */
export interface ParametrosBayes {
  media: number[]
  cov: number[][]
  det: number
  inv_cov: number[][]
}

export interface RespostaBayes {
  bayes: { relatorio: Relatorio; parametros: Record<string, ParametrosBayes> }
  naive: { relatorio: Relatorio; parametros: Record<string, ParametrosBayes> }
  teste_z: { z: number; p: number; significativo: boolean }
  amostras: Amostra[]
  n_treino: number
  n_teste: number
  eixo_x: string
  eixo_y: string
  dimensoes: number
}

export interface RespostaNormalidade {
  resultado: Record<string, unknown>
  atributos: string
  indices: number[]
  n_features: number
}

/* -------------------------------------------------------------------- lab 5 */
export interface Arquitetura {
  rotulos_entrada: string[]
  rotulos_ocultos: string[]
  rotulos_saida: string[]
  pesos_oculta: number[][]
  bias_oculta: number[]
  pesos_saida: number[][]
  bias_saida: number[]
  bias_compartilhado: boolean
  texto: string
}

export interface ResumoExercicio {
  id: string
  titulo: string
  subtitulo: string
  slide: number | null
  taxa: number
  arquitetura: string
  bias_compartilhado: boolean
}

export interface ListaExercicios {
  lab_5_0: ResumoExercicio[]
  lab_5_1: ResumoExercicio[]
}

export interface TermoForward {
  entrada: number
  peso: number
  produto: number
  origem?: string
}

export interface NeuronioForward {
  nome: string
  termos: TermoForward[]
  bias: number
  net: number
  out: number
}

export interface ConfigExercicio {
  id: string
  titulo: string
  subtitulo: string
  slide: number | null
  taxa: number
  nota: string
  bias_compartilhado: boolean
}

export interface TracoPassoUnico {
  tipo: 'passo-unico'
  config: ConfigExercicio
  arquitetura: Arquitetura
  arquitetura_depois: Arquitetura
  entradas: number[]
  alvo: number[]
  forward_oculta: NeuronioForward[]
  forward_saida: NeuronioForward[]
  erro: {
    por_saida: { nome: string; alvo: number; saida: number; erro: number }[]
    total: number
  }
  deltas_saida: { nome: string; saida: number; alvo: number; delta: number }[]
  deltas_oculta: {
    nome: string
    out: number
    delta: number
    contribuicoes: {
      origem: string
      delta: number
      peso: number
      produto: number
    }[]
  }[]
  atualizacao: {
    saida: AtualizacaoPeso[]
    oculta: AtualizacaoPeso[]
    bias: AtualizacaoBias[]
  }
  nova_predicao: {
    saidas: { nome: string; antes: number; depois: number; alvo: number }[]
    erro_antes: number
    erro_depois: number
    reduziu: boolean
  }
}

export interface AtualizacaoPeso {
  destino: string
  origem: string
  antes: number
  depois: number
  delta: number
  entrada: number
}

export interface AtualizacaoBias {
  camada: 'oculta' | 'saida'
  nome: string
  antes: number
  depois: number
  deltas: number[]
  soma_deltas: number
}

export interface TracoEpoca {
  tipo: 'epoca'
  config: ConfigExercicio
  arquitetura: Arquitetura
  arquitetura_depois: Arquitetura
  padroes: { entrada: number[]; alvo: number }[]
  passos: {
    indice: number
    entrada: number[]
    alvo: number
    saida_oculta: number[]
    saida: number
    erro: number
    delta_saida: number[]
    delta_oculta: number[]
    pesos_oculta: number[][]
    bias_oculta: number[]
    pesos_saida: number[][]
    bias_saida: number[]
  }[]
  erro_medio: number
  resultados: {
    entrada: number[]
    alvo: number
    antes: number
    depois: number
    previsto: number
    correto: boolean
  }[]
  acertos: number
  total: number
}

export type Traco = TracoPassoUnico | TracoEpoca

export interface EstadoXor {
  historico: number[]
  erro_medio: number | null
  resultados: {
    entrada: number[]
    alvo: number
    saida: number
    previsto: number
    correto: boolean
  }[]
  acertos: number
  superficie: number[][]
  eixo: number[]
  limites: { min: number; max: number }
  pesos: {
    oculta: number[][]
    bias_oculta: number[]
    saida: number[][]
    bias_saida: number[]
  }
  arquitetura: Arquitetura
}

export interface SnapshotRede {
  epoca: number
  erro: number | null
  pesos_oculta: number[][]
  bias_oculta: number[]
  pesos_saida: number[][]
  bias_saida: number[]
  saidas: number[][]
}

export interface Trajetoria {
  exercicio: string
  tipo: 'passo-unico' | 'epoca'
  taxa: number
  epocas: number
  historico: number[]
  snapshots: SnapshotRede[]
  padroes: { entrada: number[]; alvo: number[] }[]
  alvos: number[][]
  arquitetura: Arquitetura
  config: ConfigExercicio
}

export interface PadroesImagem {
  homem: { pixels: number[][]; saida: number }
  galinha: { pixels: number[][]; saida: number }
  arquitetura: { entradas: number; ocultos: number; saidas: number }
}

export interface PredicaoImagem {
  saida: number
  ativacoes_ocultas: number[]
  rotulo: string
  classe: 'homem' | 'galinha' | 'ambiguo' | 'vazio'
  vazio: boolean
}

export interface ComparacaoModelos {
  relatorios: Record<string, Relatorio>
  comparacoes: Comparacao[]
  n_teste: number
  classes?: ClasseIris[]
  n_treino?: number
  config?: { camada_oculta: number; max_iter: number; atributos: string }
}

export interface ResultadoValidacao {
  nome: string
  media: number
  desvio: number
  minimo: number
  maximo: number
  ic_baixo: number
  ic_alto: number
  acuracias: number[]
  n_avaliacoes: number
  relatorio: Relatorio
}

export interface ValidacaoCruzada {
  resultados: Record<string, ResultadoValidacao>
  comparacoes: Comparacao[]
  config: {
    k: number
    repeticoes: number
    atributos: string
    dataset: string
    n_amostras: number
    n_avaliacoes: number
  }
  classes: ClasseIris[]
}

export interface PontoCurvaKappa {
  acerto_alvo: number
  acerto_global: number
  kappa: number
  tau: number
  var_kappa: number
  var_tau: number
}

/* --------------------------------------------------------------- floresta */
export interface NoArvoreApi {
  folha: boolean
  n_amostras: number
  impureza: number
  distribuicao: Record<string, number>
  profundidade: number
  classe?: string
  atributo?: number
  limiar?: number
  ganho?: number
  esquerda?: NoArvoreApi
  direita?: NoArvoreApi
}

export interface ResumoArvore {
  indice: number
  profundidade: number
  nos: number
  folhas: number
  amostras_unicas_bag: number
  amostras_oob: number
}

export interface RespostaFloresta {
  relatorio: Relatorio
  relatorio_arvore_unica: Relatorio
  oob: { acuracia: number | null; erro: number | null }
  importancias: { indice: number; nome: string; importancia: number }[]
  arvores: ResumoArvore[]
  config: {
    n_arvores: number
    criterio: string
    profundidade_max: number | null
    max_atributos: string | number | null
    atributos: string
    n_atributos_por_no: number
  }
  amostras: Amostra[]
  n_treino: number
  n_teste: number
  eixo_x: string
  eixo_y: string
  dimensoes: number
}

export interface RespostaArvore {
  indice: number
  arvore: NoArvoreApi
  profundidade: number
  nos: number
  folhas: number
  amostras_oob: number
  amostras_unicas_bag: number
  nomes_features: string[]
  total_arvores: number
}

export interface PredicaoFloresta {
  classe: ClasseIris
  votos: Record<string, number>
  probabilidades: Record<string, number>
  total_arvores: number
  valores: number[]
}

export interface ValidacaoFloresta {
  resultados: Record<string, ResultadoValidacao>
  comparacoes: Comparacao[]
  config: {
    k: number
    repeticoes: number
    n_avaliacoes: number
    n_amostras: number
    atributos: string
    n_arvores: number
    criterio: string
  }
  classes: ClasseIris[]
}

/* --------------------------------------------------- testes de significancia */
export interface McNemar {
  a: number
  b: number
  c: number
  d: number
  discordantes: number
  metodo: string
  estatistica: number | null
  p_valor: number
  significativo: boolean
  observacao: string
}

export interface BootstrapDiferenca {
  metrica_a: number
  metrica_b: number
  diferenca: number
  ic_baixo: number
  ic_alto: number
  erro_padrao: number
  contem_zero: boolean
  significativo: boolean
  n_reamostragens: number
  confianca: number
  distribuicao: { faixas: number[]; contagens: number[] }
}

export interface TestePermutacao {
  diferenca_observada: number
  p_valor: number
  significativo: boolean
  n_permutacoes: number
  extremos: number
}

export interface RespostaSignificancia {
  metrica: string
  nome_metrica: string
  n_amostras: number
  mcnemar: McNemar
  bootstrap: BootstrapDiferenca
  permutacao: TestePermutacao
  modelo_a: { id: string; nome: string }
  modelo_b: { id: string; nome: string }
  metricas: Record<string, { nome: string; a: number; b: number }>
  teste_z_kappa: { z: number; p: number; significativo: boolean }
  mcc_multiclasse: { a: number; b: number }
  config: {
    dataset: string
    atributos: string
    proporcao: number
    n_teste: number
  }
}

export interface ParSignificancia {
  a: string
  b: string
  nome_a: string
  nome_b: string
  valor_a: number
  valor_b: number
  diferenca: number
  ic_baixo: number
  ic_alto: number
  p_mcnemar: number
  p_permutacao: number
  significativo_mcnemar: boolean
  significativo_bootstrap: boolean
  significativo_permutacao: boolean
  discordantes: number
}

export interface MatrizSignificancia {
  metrica: string
  nome_metrica: string
  modelos: { id: string; nome: string; valor: number }[]
  pares: ParSignificancia[]
  config: { dataset: string; atributos: string; n_teste: number }
}

export interface ListaClassificadores {
  classificadores: { id: string; nome: string }[]
  metricas: { id: string; nome: string }[]
}
