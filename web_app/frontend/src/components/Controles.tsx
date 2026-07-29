/** Painel de controles comum aos experimentos (dataset, atributos, split). */
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '@/lib/api'
import type { ChaveAtributos } from '@/lib/types'
import { Card, Select, Slider } from './ui'

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
  const set = <K extends keyof ConfigExperimento>(
    chave: K,
    valor: ConfigExperimento[K],
  ) => setConfig((c) => ({ ...c, [chave]: valor }))
  return { config, set }
}

export function usarMetadata() {
  return useQuery({
    queryKey: ['metadata'],
    queryFn: api.dataset.metadata,
    staleTime: Infinity,
  })
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
  const { data: meta } = usarMetadata()

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
            meta?.atributos.map((a) => ({
              valor: a.id,
              rotulo: a.nome,
            })) ?? [{ valor: 'petalas' as ChaveAtributos, rotulo: 'Pétalas' }]
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
      </div>
    </Card>
  )
}
