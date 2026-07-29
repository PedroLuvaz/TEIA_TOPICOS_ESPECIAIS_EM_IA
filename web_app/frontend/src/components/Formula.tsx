/** Renderizacao de formulas LaTeX com KaTeX. */
import katex from 'katex'
import { useMemo } from 'react'
import { cn } from '@/lib/utils'

export function Formula({
  children,
  display = false,
  className,
}: {
  children: string
  display?: boolean
  className?: string
}) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(children, {
        displayMode: display,
        throwOnError: false,
        strict: false,
        trust: false,
      })
    } catch {
      return `<code>${children}</code>`
    }
  }, [children, display])

  return (
    <span
      className={cn(display && 'block overflow-x-auto', className)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

/** Bloco de formula com titulo e explicacao opcional. */
export function BlocoFormula({
  titulo,
  latex,
  explicacao,
  className,
}: {
  titulo?: string
  latex: string
  explicacao?: string
  className?: string
}) {
  return (
    <div
      className={cn(
        'rounded-lg border border-subtle bg-sunken px-4 py-3',
        className,
      )}
    >
      {titulo && <div className="kicker mb-2">{titulo}</div>}
      <Formula display>{latex}</Formula>
      {explicacao && (
        <p className="mt-2 text-xs leading-relaxed text-muted">{explicacao}</p>
      )}
    </div>
  )
}
