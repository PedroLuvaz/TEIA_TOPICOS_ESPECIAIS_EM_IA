import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/* ----------------------------------------------------------- formatacao ---- */

/** Numero com N casas, em notacao brasileira (virgula decimal). */
export function num(valor: number | null | undefined, casas = 4): string {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return '—'
  return valor.toLocaleString('pt-BR', {
    minimumFractionDigits: casas,
    maximumFractionDigits: casas,
  })
}

/** Percentual, ex.: 0.9778 -> "97,78%" */
export function pct(valor: number | null | undefined, casas = 2): string {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return '—'
  return `${(valor * 100).toLocaleString('pt-BR', {
    minimumFractionDigits: casas,
    maximumFractionDigits: casas,
  })}%`
}

/** Notacao cientifica quando o valor e muito pequeno. */
export function sci(valor: number | null | undefined, casas = 4): string {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return '—'
  if (valor !== 0 && Math.abs(valor) < 1e-4) return valor.toExponential(2)
  return num(valor, casas)
}

/* --------------------------------------------------------------- cores ----- */

/**
 * Cores por classe.
 *
 * As 3 do Iris tem cor fixa (a interface inteira ja e reconhecida por elas).
 * Classes de outros datasets recebem cores da paleta abaixo na ordem em que
 * o dataset as declara — ver `registrarCoresClasses`.
 */
export const CORES_CLASSE: Record<string, string> = {
  setosa: '#0ea5e9',
  versicolor: '#10b981',
  virginica: '#f43f5e',
}

const PALETA_CLASSES = [
  '#0ea5e9', '#f59e0b', '#8b5cf6', '#10b981',
  '#f43f5e', '#14b8a6', '#ec4899', '#84cc16',
]

/**
 * Garante uma cor para cada classe de um dataset. Idempotente: chamada a cada
 * render pelo hook de metadata, nunca reatribui uma classe ja registrada —
 * assim a cor de uma classe nao muda ao trocar de aba.
 */
export function registrarCoresClasses(classes: string[]): void {
  classes.forEach((c, i) => {
    if (!(c in CORES_CLASSE)) {
      CORES_CLASSE[c] = PALETA_CLASSES[i % PALETA_CLASSES.length]
    }
  })
}

export const CORES_MODELO: Record<string, string> = {
  distancia_minima: '#8b5cf6',
  delta_ova: '#f59e0b',
  bayes: '#0ea5e9',
  naive: '#10b981',
  floresta: '#14b8a6',
  mlp: '#f43f5e',
  perceptron: '#8b5cf6',
  delta: '#f59e0b',
}

export function corDaClasse(classe: string): string {
  return CORES_CLASSE[classe] ?? '#94a3b8'
}

/**
 * Classes de um relatorio, na ordem da matriz de confusao.
 *
 * Evita depender de uma lista fixa de classes no frontend: o proprio
 * relatorio ja carrega as classes com que foi calculado.
 */
export function classesDoRelatorio(
  relatorio: { matriz: Record<string, Record<string, number>> } | undefined,
): string[] {
  return relatorio ? Object.keys(relatorio.matriz) : []
}

/** Capitaliza a primeira letra (nomes de classe vem em minusculas). */
export function cap(texto: string): string {
  return texto.charAt(0).toUpperCase() + texto.slice(1)
}

/**
 * Interpretacao qualitativa do Kappa (escala de Landis & Koch),
 * usada nos laboratorios para julgar a concordancia.
 */
export function interpretarKappa(k: number): {
  rotulo: string
  tom: 'bom' | 'medio' | 'ruim'
} {
  if (k > 0.8) return { rotulo: 'Excelente', tom: 'bom' }
  if (k > 0.6) return { rotulo: 'Substancial', tom: 'bom' }
  if (k > 0.4) return { rotulo: 'Moderada', tom: 'medio' }
  if (k > 0.2) return { rotulo: 'Razoavel', tom: 'medio' }
  return { rotulo: 'Fraca', tom: 'ruim' }
}

/** Mistura duas cores hex por um fator t em [0,1]. */
export function misturar(hexA: string, hexB: string, t: number): string {
  const a = parseInt(hexA.slice(1), 16)
  const b = parseInt(hexB.slice(1), 16)
  const f = Math.max(0, Math.min(1, t))
  const r = Math.round(((a >> 16) & 255) * (1 - f) + ((b >> 16) & 255) * f)
  const g = Math.round(((a >> 8) & 255) * (1 - f) + ((b >> 8) & 255) * f)
  const bl = Math.round((a & 255) * (1 - f) + (b & 255) * f)
  return `#${((r << 16) | (g << 8) | bl).toString(16).padStart(6, '0')}`
}

/** Escala divergente azul -> branco -> vermelho (saida 0..1 do XOR). */
export function escalaDivergente(t: number): string {
  const v = Math.max(0, Math.min(1, t))
  return v < 0.5
    ? misturar('#0ea5e9', '#ffffff', v * 2)
    : misturar('#ffffff', '#f43f5e', (v - 0.5) * 2)
}

/** Tom de cinza para intensidade de pixel (0 = claro, 1 = escuro). */
export function tomCinza(v: number): string {
  const t = Math.round(255 - Math.max(0, Math.min(1, v)) * 220)
  const h = t.toString(16).padStart(2, '0')
  return `#${h}${h}${h}`
}

/* ------------------------------------------------ rede feedforward local --- */

export interface PesosRede {
  pesos_oculta: number[][]
  bias_oculta: number[]
  pesos_saida: number[][]
  bias_saida: number[]
}

export function sigmoide(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

/**
 * Alimentacao adiante da MLP, replicando `models/mlp_backprop.py`.
 *
 * Roda no cliente para que arrastar o slider de epocas recalcule a superficie
 * de decisao instantaneamente, sem uma chamada de rede por quadro. Os pesos
 * vem sempre do backend — aqui so ha a propagacao, nunca treino.
 */
export function forward(
  entradas: number[],
  pesos: PesosRede,
): { ocultas: number[]; saidas: number[] } {
  const ocultas = pesos.bias_oculta.map((bias, i) =>
    sigmoide(
      bias +
        pesos.pesos_oculta[i].reduce((s, w, j) => s + w * entradas[j], 0),
    ),
  )
  const saidas = pesos.bias_saida.map((bias, i) =>
    sigmoide(
      bias + pesos.pesos_saida[i].reduce((s, w, j) => s + w * ocultas[j], 0),
    ),
  )
  return { ocultas, saidas }
}

/** Amostra a saida da rede numa grade quadrada — usado no mapa do XOR. */
export function superficieDeSaida(
  pesos: PesosRede,
  resolucao: number,
  min = -0.35,
  max = 1.35,
): number[][] {
  const passo = (max - min) / (resolucao - 1)
  const eixo = Array.from({ length: resolucao }, (_, k) => min + k * passo)
  return eixo.map((y) => eixo.map((x) => forward([x, y], pesos).saidas[0]))
}
