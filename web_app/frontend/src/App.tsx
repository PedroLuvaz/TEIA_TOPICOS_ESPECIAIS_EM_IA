import { Suspense, lazy } from 'react'
import { Layout, usarRota } from './components/Layout'
import { Carregando } from './components/ui'

// Cada laboratorio vira um chunk proprio — o carregamento inicial nao paga
// pelas bibliotecas de grafico e LaTeX antes de a pagina precisar delas.
const PaginaClassificar = lazy(() =>
  import('./pages/Classificar').then((m) => ({ default: m.PaginaClassificar })),
)
const PaginaDistanciaMinima = lazy(() =>
  import('./pages/DistanciaMinima').then((m) => ({
    default: m.PaginaDistanciaMinima,
  })),
)
const PaginaPerceptronDelta = lazy(() =>
  import('./pages/PerceptronDelta').then((m) => ({
    default: m.PaginaPerceptronDelta,
  })),
)
const PaginaMetricas = lazy(() =>
  import('./pages/Metricas').then((m) => ({ default: m.PaginaMetricas })),
)
const PaginaBayes = lazy(() =>
  import('./pages/Bayes').then((m) => ({ default: m.PaginaBayes })),
)
const PaginaLab50 = lazy(() =>
  import('./pages/Lab50').then((m) => ({ default: m.PaginaLab50 })),
)
const PaginaLab51 = lazy(() =>
  import('./pages/Lab51').then((m) => ({ default: m.PaginaLab51 })),
)
const PaginaFloresta = lazy(() =>
  import('./pages/Floresta').then((m) => ({ default: m.PaginaFloresta })),
)
const PaginaConstrutor = lazy(() =>
  import('./pages/Construtor').then((m) => ({ default: m.PaginaConstrutor })),
)

export function App() {
  const [rota, navegar] = usarRota()

  return (
    <Layout rota={rota} navegar={navegar}>
      <Suspense fallback={<Carregando texto="Carregando laboratório…" />}>
        {rota === 'classificar' && <PaginaClassificar />}
        {rota === 'distancia-minima' && <PaginaDistanciaMinima />}
        {rota === 'perceptron-delta' && <PaginaPerceptronDelta />}
        {rota === 'metricas' && <PaginaMetricas />}
        {rota === 'bayes' && <PaginaBayes />}
        {rota === 'lab-5-0' && <PaginaLab50 />}
        {rota === 'lab-5-1' && <PaginaLab51 />}
        {rota === 'floresta' && <PaginaFloresta />}
        {rota === 'construtor' && <PaginaConstrutor />}
      </Suspense>
    </Layout>
  )
}
