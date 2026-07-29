/** Shell da aplicacao: barra lateral de navegacao, cabecalho e tema. */
import { motion } from 'motion/react'
import {
  Blocks,
  Boxes,
  BrainCircuit,
  GitCompareArrows,
  Menu,
  Moon,
  Network,
  Ruler,
  Sigma,
  Sun,
  X,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

export interface Aba {
  id: string
  rotulo: string
  descricao: string
  icone: ReactNode
  grupo: string
}

export const ABAS: Aba[] = [
  {
    id: 'distancia-minima',
    rotulo: 'Distância Mínima',
    descricao: 'Protótipos e discriminante linear',
    icone: <Ruler size={17} />,
    grupo: 'Laboratórios',
  },
  {
    id: 'perceptron-delta',
    rotulo: 'Perceptron & Delta',
    descricao: 'Rosenblatt, Widrow-Hoff e XOR',
    icone: <GitCompareArrows size={17} />,
    grupo: 'Laboratórios',
  },
  {
    id: 'metricas',
    rotulo: 'Métricas Avançadas',
    descricao: 'Kappa, Tau e teste Z',
    icone: <Sigma size={17} />,
    grupo: 'Laboratórios',
  },
  {
    id: 'bayes',
    rotulo: 'Bayes & Normalidade',
    descricao: 'QDA, Naive Bayes e MVN',
    icone: <Boxes size={17} />,
    grupo: 'Laboratórios',
  },
  {
    id: 'lab-5-0',
    rotulo: 'Lab 5.0 · XOR (MLP)',
    descricao: 'Backpropagation e o XOR',
    icone: <Network size={17} />,
    grupo: 'Lab 5 — Redes Neurais',
  },
  {
    id: 'lab-5-1',
    rotulo: 'Lab 5.1 · Feedforward',
    descricao: 'Galinha vs Homem e Iris',
    icone: <BrainCircuit size={17} />,
    grupo: 'Lab 5 — Redes Neurais',
  },
  {
    id: 'construtor',
    rotulo: 'Construtor de Rede',
    descricao: 'Monte a MLP e treine do zero',
    icone: <Blocks size={17} />,
    grupo: 'Lab 5 — Redes Neurais',
  },
]

export function usarRota(): [string, (id: string) => void] {
  const ler = () => window.location.hash.replace(/^#\/?/, '') || ABAS[0].id
  const [rota, setRota] = useState(ler)

  useEffect(() => {
    const aoMudar = () => setRota(ler())
    window.addEventListener('hashchange', aoMudar)
    if (!window.location.hash) window.location.hash = `#/${ABAS[0].id}`
    return () => window.removeEventListener('hashchange', aoMudar)
  }, [])

  const navegar = (id: string) => {
    window.location.hash = `#/${id}`
  }
  return [rota, navegar]
}

function usarTema(): [boolean, () => void] {
  const [escuro, setEscuro] = useState(() => {
    const salvo = localStorage.getItem('teia-tema')
    if (salvo) return salvo === 'escuro'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', escuro)
    localStorage.setItem('teia-tema', escuro ? 'escuro' : 'claro')
  }, [escuro])

  return [escuro, () => setEscuro((v) => !v)]
}

export function Layout({
  rota,
  navegar,
  children,
}: {
  rota: string
  navegar: (id: string) => void
  children: ReactNode
}) {
  const [escuro, alternarTema] = usarTema()
  const [menuAberto, setMenuAberto] = useState(false)
  const atual = ABAS.find((a) => a.id === rota) ?? ABAS[0]

  const grupos = [...new Set(ABAS.map((a) => a.grupo))]

  useEffect(() => {
    setMenuAberto(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [rota])

  return (
    <div className="flex min-h-screen">
      {/* Overlay do menu no mobile */}
      {menuAberto && (
        <div
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm lg:hidden"
          onClick={() => setMenuAberto(false)}
          aria-hidden
        />
      )}

      {/* Barra lateral */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-[268px] flex-col border-r border-subtle bg-surface',
          'transition-transform duration-300 lg:translate-x-0',
          menuAberto ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex items-center gap-3 border-b border-subtle px-5 py-4">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-accent-400 to-accent-600 shadow-sm shadow-accent-600/25">
            <BrainCircuit size={19} className="text-white" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-primary">
              Reconhecimento de Padrões
            </p>
            <p className="truncate text-[11px] text-muted">TEIA · UEPB 2026</p>
          </div>
          <button
            className="ml-auto text-muted hover:text-primary lg:hidden"
            onClick={() => setMenuAberto(false)}
            aria-label="Fechar menu"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="flex-1 space-y-5 overflow-y-auto px-3 py-4">
          {grupos.map((grupo) => (
            <div key={grupo}>
              <p className="kicker mb-2 px-2 text-muted">{grupo}</p>
              <ul className="space-y-0.5">
                {ABAS.filter((a) => a.grupo === grupo).map((aba) => {
                  const ativo = aba.id === rota
                  return (
                    <li key={aba.id}>
                      <button
                        onClick={() => navegar(aba.id)}
                        aria-current={ativo ? 'page' : undefined}
                        className={cn(
                          'group relative flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left',
                          'transition-colors',
                          ativo
                            ? 'text-accent-700 dark:text-accent-400'
                            : 'text-secondary hover:bg-raised hover:text-primary',
                        )}
                      >
                        {ativo && (
                          <motion.span
                            layoutId="nav-ativo"
                            className="absolute inset-0 rounded-lg bg-accent-500/10 ring-1 ring-accent-500/25"
                            transition={{ type: 'spring', duration: 0.4, bounce: 0.15 }}
                          />
                        )}
                        <span className="relative z-10 shrink-0">{aba.icone}</span>
                        <span className="relative z-10 min-w-0">
                          <span className="block truncate text-[13px] font-medium">
                            {aba.rotulo}
                          </span>
                          <span className="block truncate text-[11px] text-muted">
                            {aba.descricao}
                          </span>
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </nav>

        <div className="border-t border-subtle px-3 py-3">
          <button
            onClick={alternarTema}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-[13px] text-secondary transition-colors hover:bg-raised hover:text-primary"
          >
            {escuro ? <Sun size={16} /> : <Moon size={16} />}
            Tema {escuro ? 'claro' : 'escuro'}
          </button>
        </div>
      </aside>

      {/* Conteudo */}
      <div className="flex min-w-0 flex-1 flex-col lg:pl-[268px]">
        <header className="sticky top-0 z-20 border-b border-subtle bg-surface/85 backdrop-blur-md">
          <div className="flex items-center gap-3 px-5 py-3.5 sm:px-7">
            <button
              className="text-secondary hover:text-primary lg:hidden"
              onClick={() => setMenuAberto(true)}
              aria-label="Abrir menu"
            >
              <Menu size={20} />
            </button>
            <div className="min-w-0">
              <p className="kicker">{atual.grupo}</p>
              <h1 className="truncate text-base font-semibold text-primary sm:text-lg">
                {atual.rotulo}
              </h1>
            </div>
          </div>
        </header>

        <main className="flex-1 px-5 py-6 sm:px-7 sm:py-8">
          <motion.div
            key={rota}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="mx-auto max-w-[1400px]"
          >
            {children}
          </motion.div>
        </main>

        <footer className="border-t border-subtle px-5 py-5 sm:px-7">
          <p className="text-xs text-muted">
            Tópicos Especiais em Inteligência Artificial · UEPB 2026 — Erick
            Nathan · Laura Barbosa · Pedro Lucas. Toda a matemática roda em
            Python puro, sem bibliotecas de ML (exceto o item (ii) do Lab 5.1).
          </p>
        </footer>
      </div>
    </div>
  )
}
