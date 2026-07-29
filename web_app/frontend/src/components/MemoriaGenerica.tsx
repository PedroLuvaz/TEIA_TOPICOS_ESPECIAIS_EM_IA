/**
 * Renderizador do formato generico de memoria de calculo.
 *
 * Recebe o traco montado por `web_app/backend/traco.py` (secoes numeradas,
 * cada uma com blocos tipados) e desenha a janela — a mesma para todos os
 * laboratorios. Adicionar a memoria de um lab novo nao exige tela nova.
 */
import { motion } from 'motion/react'
import { X } from 'lucide-react'
import { useEffect } from 'react'
import { cn } from '@/lib/utils'
import { BlocoFormula } from './Formula'
import { Botao, Carregando, ErroBox, Nota } from './ui'

/* --------------------------------------------------------------- tipos --- */
export interface BlocoTexto {
  tipo: 'texto'
  conteudo: string
}
export interface BlocoFormulaT {
  tipo: 'formula'
  latex: string
  titulo: string | null
  explicacao: string | null
}
export interface BlocoPassos {
  tipo: 'passos'
  itens: string[]
  titulo: string | null
}
export interface BlocoTabela {
  tipo: 'tabela'
  colunas: string[]
  linhas: (string | number)[][]
  titulo: string | null
  alinhamento: ('esq' | 'dir')[]
}
export interface BlocoResultado {
  tipo: 'resultado'
  conteudo: string
  tom: 'destaque' | 'bom' | 'medio' | 'ruim'
}
export interface BlocoNota {
  tipo: 'nota'
  conteudo: string
  tom: 'info' | 'atencao' | 'ok'
  titulo: string | null
}
export interface BlocoRef {
  tipo: 'ref'
  arquivo: string
  linha: number
  nome: string
}

export type Bloco =
  | BlocoTexto
  | BlocoFormulaT
  | BlocoPassos
  | BlocoTabela
  | BlocoResultado
  | BlocoNota
  | BlocoRef

export interface SecaoTraco {
  titulo: string
  blocos: Bloco[]
}

export interface TracoGenerico {
  formato: 'generico'
  titulo: string
  subtitulo: string
  cabecalho: { rotulo: string; valor: string }[]
  secoes: SecaoTraco[]
}

/* ---------------------------------------------------------------- modal --- */
export function MemoriaGenerica({
  traco,
  carregando,
  erro,
  onFechar,
}: {
  traco?: TracoGenerico
  carregando?: boolean
  erro?: unknown
  onFechar: () => void
}) {
  useEffect(() => {
    const aoTeclar = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onFechar()
    }
    document.addEventListener('keydown', aoTeclar)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', aoTeclar)
      document.body.style.overflow = ''
    }
  }, [onFechar])

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/55 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onFechar} aria-hidden />
      <motion.div
        initial={{ opacity: 0, y: 16, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full max-w-5xl rounded-2xl border border-subtle bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Memória de cálculo"
      >
        <header className="sticky top-0 z-10 rounded-t-2xl border-b border-subtle bg-surface/95 px-6 py-4 backdrop-blur">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="kicker">Memória de cálculo</p>
              <h2 className="mt-0.5 text-lg font-semibold text-primary">
                {traco?.titulo ?? 'Carregando…'}
              </h2>
              {traco && (
                <p className="mt-0.5 text-sm text-muted">{traco.subtitulo}</p>
              )}
            </div>
            <Botao variante="fantasma" tamanho="sm" onClick={onFechar}>
              <X size={16} />
              Fechar
            </Botao>
          </div>

          {!!traco?.cabecalho.length && (
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5">
              {traco.cabecalho.map((c, i) => (
                <span key={i} className="text-xs">
                  <span className="text-muted">{c.rotulo}: </span>
                  <span className="font-medium text-secondary">{c.valor}</span>
                </span>
              ))}
            </div>
          )}
        </header>

        <div className="space-y-7 px-6 py-6">
          {carregando && <Carregando texto="Montando a memória de cálculo…" />}
          {erro ? <ErroBox erro={erro} /> : null}
          {traco?.secoes.map((s, i) => (
            <Secao key={i} numero={i + 1} titulo={s.titulo} blocos={s.blocos} />
          ))}
        </div>
      </motion.div>
    </div>
  )
}

/* --------------------------------------------------------------- secao ---- */
function Secao({
  numero,
  titulo,
  blocos,
}: {
  numero: number
  titulo: string
  blocos: Bloco[]
}) {
  return (
    <section className="space-y-3">
      <h3 className="flex items-center gap-2.5 text-sm font-semibold text-primary">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-accent-500/15 text-xs font-bold text-accent-700 dark:text-accent-400">
          {numero}
        </span>
        {titulo}
      </h3>
      <div className="space-y-3 pl-8.5">
        {blocos.map((b, i) => (
          <RenderBloco key={i} bloco={b} />
        ))}
      </div>
    </section>
  )
}

function RenderBloco({ bloco }: { bloco: Bloco }) {
  switch (bloco.tipo) {
    case 'texto':
      return (
        <p className="text-sm leading-relaxed text-secondary">
          {bloco.conteudo}
        </p>
      )

    case 'formula':
      return (
        <BlocoFormula
          latex={bloco.latex}
          titulo={bloco.titulo ?? undefined}
          explicacao={bloco.explicacao ?? undefined}
        />
      )

    case 'passos':
      return (
        <div>
          {bloco.titulo && (
            <p className="kicker mb-1.5 text-muted">{bloco.titulo}</p>
          )}
          <div className="space-y-0.5 rounded-md bg-sunken px-3 py-2.5">
            {bloco.itens.map((linha, i) => (
              <p
                key={i}
                className="whitespace-pre-wrap font-mono text-[12.5px] leading-relaxed text-secondary"
              >
                {linha}
              </p>
            ))}
          </div>
        </div>
      )

    case 'tabela':
      return (
        <div>
          {bloco.titulo && (
            <p className="kicker mb-1.5 text-muted">{bloco.titulo}</p>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-subtle">
                  {bloco.colunas.map((c, i) => (
                    <th
                      key={i}
                      className={cn(
                        'py-2 text-[11px] font-semibold text-muted',
                        bloco.alinhamento[i] === 'dir'
                          ? 'px-2 text-right'
                          : 'pr-3 text-left',
                      )}
                    >
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bloco.linhas.map((linha, i) => (
                  <tr
                    key={i}
                    className="border-b border-subtle/60 last:border-0"
                  >
                    {linha.map((celula, j) => (
                      <td
                        key={j}
                        className={cn(
                          'py-2',
                          bloco.alinhamento[j] === 'dir'
                            ? 'px-2 text-right tabular text-primary'
                            : 'pr-3 text-secondary',
                        )}
                      >
                        {celula}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )

    case 'resultado': {
      const tons = {
        destaque:
          'bg-accent-500/10 text-accent-700 dark:text-accent-400',
        bom: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
        medio: 'bg-accent-500/10 text-accent-700 dark:text-accent-400',
        ruim: 'bg-rose-500/10 text-rose-700 dark:text-rose-400',
      }
      return (
        <div
          className={cn(
            'rounded-md px-3 py-2.5 font-mono text-sm font-semibold',
            tons[bloco.tom],
          )}
        >
          {bloco.conteudo}
        </div>
      )
    }

    case 'nota':
      return (
        <Nota tom={bloco.tom} titulo={bloco.titulo ?? undefined}>
          {bloco.conteudo}
        </Nota>
      )

    case 'ref':
      return (
        <p className="font-mono text-[10.5px] text-muted">
          → {bloco.arquivo} : linha {bloco.linha}
          {bloco.nome !== '?' && `  (${bloco.nome})`}
        </p>
      )
  }
}
