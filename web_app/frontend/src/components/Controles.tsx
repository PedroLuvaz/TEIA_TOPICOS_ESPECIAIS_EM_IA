/** Painel de controles comum aos experimentos (dataset, atributos, split). */
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useState } from 'react'
import { api } from '@/lib/api'
import type { ChaveAtributos, DatasetInfo, Metadata } from '@/lib/types'
import { registrarCoresClasses } from '@/lib/utils'
import { Card, Nota, Select, Slider } from './ui'

export interface ConfigExperimento {
  // A assinatura de indice permite passar a config direto como query params.
  [chave: string]: string | number | boolean | undefined
  dataset: string
  atributos: ChaveAtributos
  proporcao: number
}

export function usarConfig(inicial?: Partial<ConfigExperimento>) {
  const [config, setConfig] = useState<ConfigExperimento>({
    dataset: 'v1',
    atributos: 'petalas',
    proporcao: 0.7,
    ...inicial,
  })
  const qc = useQueryClient()

  /**
   * Troca uma chave da config.
   *
   * Ao trocar de `dataset`, o conjunto de atributos tambem e ajustado NA MESMA
   * atualizacao de estado: os datasets tem chaves proprias ('petalas' no Iris,
   * 'clima_pais' no seminario), e corrigir depois, num efeito, faria a pagina
   * disparar uma requisicao invalida antes de se acertar.
   */
  const set = useCallback(
    <K extends keyof ConfigExperimento>(
      chave: K,
      valor: ConfigExperimento[K],
    ) => {
      setConfig((c) => {
        if (chave !== 'dataset' || valor === c.dataset) {
          return { ...c, [chave]: valor }
        }
        const meta = qc.getQueryData<Metadata>(['metadata'])
        const alvo = meta?.datasets.find((d) => d.id === valor)
        const atributosValidos =
          !alvo || alvo.atributos.some((atr) => atr.id === c.atributos)
        return {
          ...c,
          dataset: String(valor),
          atributos: atributosValidos
            ? c.atributos
            : (alvo?.atributos_padrao ?? c.atributos),
        }
      })
    },
    [qc],
  )

  return { config, set }
}

export function usarMetadata() {
  const q = useQuery({
    queryKey: ['metadata'],
    queryFn: api.dataset.metadata,
    staleTime: Infinity,
  })

  // Registra as cores de todas as classes de todos os datasets assim que o
  // metadata chega — idempotente, entao rodar a cada render nao custa nada.
  if (q.data) {
    for (const d of q.data.datasets) registrarCoresClasses(d.classes)
  }
  return q
}

/**
 * Informacoes do dataset selecionado.
 *
 * E o que substitui as listas fixas de classes espalhadas pelas paginas: cada
 * tela pergunta ao dataset quais sao as classes, features e pares validos.
 */
export function usarDataset(dataset: string) {
  const { data: meta, isPending } = usarMetadata()
  const info: DatasetInfo | undefined = meta?.datasets.find(
    (d) => d.id === dataset,
  )
  return {
    meta,
    info,
    carregando: isPending,
    classes: info?.classes ?? [],
    features: info?.features ?? [],
    atributos: info?.atributos ?? [],
    pares: info?.pares ?? [],
    categorico: info?.tipo === 'categorico',
    /** Rotulo de um valor categorico, ex.: (0, 1) -> 'Vento'. */
    rotuloValor: (indiceAtributo: number, valor: number) => {
      const tabela = info?.valores?.[String(indiceAtributo)]
      const i = Math.round(valor)
      return tabela && i >= 0 && i < tabela.length
        ? tabela[i]
        : valor.toFixed(2)
    },
  }
}

export function PainelConfig({
  config,
  set,
  children,
  mostrarProporcao = true,
}: {
  config: ConfigExperimento
  set: <K extends keyof ConfigExperimento>(
    chave: K,
    valor: ConfigExperimento[K],
  ) => void
  children?: React.ReactNode
  mostrarProporcao?: boolean
}) {
  const { meta, atributos, categorico } = usarDataset(config.dataset)

  return (
    <Card titulo="configuração do experimento">
      <div className="space-y-4">
        <Select
          rotulo="Base de dados"
          valor={config.dataset}
          onChange={(v) => set('dataset', v)}
          opcoes={
            meta?.datasets.map((d) => ({
              valor: d.id,
              rotulo: d.nome,
              desabilitado: !d.disponivel,
            })) ?? [{ valor: 'v1', rotulo: 'Iris Original' }]
          }
        />

        <Select
          rotulo="Atributos"
          valor={config.atributos}
          onChange={(v) => set('atributos', v)}
          opcoes={
            atributos.length
              ? atributos.map((a) => ({ valor: a.id, rotulo: a.nome }))
              : [{ valor: config.atributos, rotulo: 'carregando…' }]
          }
        />

        {mostrarProporcao && (
          <Slider
            rotulo="Proporção de treino"
            valor={config.proporcao}
            onChange={(v) => set('proporcao', v)}
            min={0.3}
            max={0.9}
            passo={0.05}
            formatar={(v) => `${Math.round(v * 100)}% / ${Math.round((1 - v) * 100)}%`}
          />
        )}

        {children}

        {categorico && (
          <Nota tom="atencao" titulo="Dataset categórico">
            Os atributos são categorias codificadas em inteiros. No gráfico os
            pontos levam um pequeno deslocamento aleatório — sem ele todas as
            amostras cairiam umas sobre as outras em poucas posições da grade.
          </Nota>
        )}
      </div>
    </Card>
  )
}
