/** Componentes base do design system. */
import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/* ------------------------------------------------------------------ Card --- */
export function Card({
  titulo,
  acao,
  children,
  className,
  padded = true,
}: {
  titulo?: string
  acao?: ReactNode
  children: ReactNode
  className?: string
  padded?: boolean
}) {
  return (
    <section
      className={cn(
        'bg-surface border border-subtle rounded-[--radius-card] shadow-sm',
        'transition-colors',
        className,
      )}
    >
      {titulo && (
        <header className="flex items-center justify-between gap-3 border-b border-subtle px-5 py-3">
          <h2 className="kicker">{titulo}</h2>
          {acao}
        </header>
      )}
      <div className={cn(padded && 'p-5')}>{children}</div>
    </section>
  )
}

/* ------------------------------------------------------------ MetricBlock -- */
export function Metrica({
  rotulo,
  valor,
  detalhe,
  cor,
  destaque,
}: {
  rotulo: string
  valor: ReactNode
  detalhe?: ReactNode
  cor?: string
  destaque?: 'bom' | 'medio' | 'ruim'
}) {
  const tomFundo =
    destaque === 'bom'
      ? 'from-emerald-500/10'
      : destaque === 'ruim'
        ? 'from-rose-500/10'
        : destaque === 'medio'
          ? 'from-accent-500/10'
          : 'from-transparent'

  return (
    <div className="relative overflow-hidden rounded-[--radius-card] border border-subtle bg-surface">
      <div
        className={cn('absolute inset-0 bg-gradient-to-br to-transparent', tomFundo)}
        aria-hidden
      />
      <div
        className="absolute left-0 top-0 h-full w-[3px]"
        style={{ backgroundColor: cor ?? 'var(--color-accent-500)' }}
        aria-hidden
      />
      <div className="relative px-4 py-3.5">
        <div className="kicker mb-1 !text-[10px] text-muted">{rotulo}</div>
        <div className="tabular text-2xl font-semibold leading-tight text-primary">
          {valor}
        </div>
        {detalhe && (
          <div className="mt-0.5 text-xs text-muted tabular">{detalhe}</div>
        )}
      </div>
    </div>
  )
}

/* ----------------------------------------------------------------- Badge --- */
export function Badge({
  children,
  tom = 'neutro',
  className,
}: {
  children: ReactNode
  tom?: 'neutro' | 'bom' | 'medio' | 'ruim' | 'info'
  className?: string
}) {
  const tons = {
    neutro: 'bg-zinc-500/10 text-secondary border-zinc-500/20',
    bom: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/25',
    medio: 'bg-accent-500/10 text-accent-700 dark:text-accent-400 border-accent-500/25',
    ruim: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/25',
    info: 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/25',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5',
        'text-xs font-medium whitespace-nowrap',
        tons[tom],
        className,
      )}
    >
      {children}
    </span>
  )
}

/* ---------------------------------------------------------------- Botao ---- */
export function Botao({
  children,
  onClick,
  variante = 'secundario',
  tamanho = 'md',
  disabled,
  className,
  type = 'button',
}: {
  children: ReactNode
  onClick?: () => void
  variante?: 'primario' | 'secundario' | 'fantasma'
  tamanho?: 'sm' | 'md'
  disabled?: boolean
  className?: string
  type?: 'button' | 'submit'
}) {
  const variantes = {
    primario:
      'bg-accent-600 text-white hover:bg-accent-500 active:bg-accent-700 shadow-sm shadow-accent-600/20 border-transparent',
    secundario:
      'bg-surface text-primary hover:bg-raised border-strong hover:border-accent-500/50',
    fantasma:
      'bg-transparent text-secondary hover:text-primary hover:bg-raised border-transparent',
  }
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg border font-medium',
        'transition-all duration-150 select-none',
        'disabled:cursor-not-allowed disabled:opacity-45',
        tamanho === 'sm' ? 'px-2.5 py-1.5 text-xs' : 'px-4 py-2 text-sm',
        variantes[variante],
        className,
      )}
    >
      {children}
    </button>
  )
}

/* ---------------------------------------------------------------- Select --- */
export function Select<T extends string>({
  rotulo,
  valor,
  onChange,
  opcoes,
  className,
}: {
  rotulo?: string
  valor: T
  onChange: (v: T) => void
  opcoes: { valor: T; rotulo: string; desabilitado?: boolean }[]
  className?: string
}) {
  return (
    <label className={cn('block', className)}>
      {rotulo && <span className="kicker mb-1.5 block text-muted">{rotulo}</span>}
      <select
        value={valor}
        onChange={(e) => onChange(e.target.value as T)}
        className={cn(
          'w-full rounded-lg border border-strong bg-surface px-3 py-2 text-sm',
          'text-primary transition-colors cursor-pointer',
          'hover:border-accent-500/50 focus:border-accent-500 focus:outline-none',
        )}
      >
        {opcoes.map((o) => (
          <option key={o.valor} value={o.valor} disabled={o.desabilitado}>
            {o.rotulo}
          </option>
        ))}
      </select>
    </label>
  )
}

/* ---------------------------------------------------------------- Slider --- */
export function Slider({
  rotulo,
  valor,
  onChange,
  min,
  max,
  passo = 1,
  formatar,
  className,
}: {
  rotulo: string
  valor: number
  onChange: (v: number) => void
  min: number
  max: number
  passo?: number
  formatar?: (v: number) => string
  className?: string
}) {
  return (
    <label className={cn('block', className)}>
      <span className="mb-1.5 flex items-baseline justify-between">
        <span className="kicker text-muted">{rotulo}</span>
        <span className="tabular text-xs font-semibold text-primary">
          {formatar ? formatar(valor) : valor}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={passo}
        value={valor}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-zinc-500/25 accent-accent-500"
      />
    </label>
  )
}

/* ------------------------------------------------------------- SegmentBar -- */
export function Segmentos<T extends string>({
  valor,
  onChange,
  opcoes,
  className,
}: {
  valor: T
  onChange: (v: T) => void
  opcoes: { valor: T; rotulo: string; icone?: ReactNode }[]
  className?: string
}) {
  return (
    <div
      role="tablist"
      className={cn(
        'inline-flex gap-1 rounded-lg border border-subtle bg-sunken p-1',
        className,
      )}
    >
      {opcoes.map((o) => {
        const ativo = o.valor === valor
        return (
          <button
            key={o.valor}
            role="tab"
            aria-selected={ativo}
            onClick={() => onChange(o.valor)}
            className={cn(
              'relative rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
              'inline-flex items-center gap-1.5 whitespace-nowrap',
              ativo ? 'text-primary' : 'text-muted hover:text-secondary',
            )}
          >
            {ativo && (
              <motion.span
                layoutId={`seg-${opcoes.map((x) => x.valor).join('-')}`}
                className="absolute inset-0 rounded-md bg-surface shadow-sm ring-1 ring-black/5 dark:ring-white/10"
                transition={{ type: 'spring', duration: 0.35, bounce: 0.15 }}
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-1.5">
              {o.icone}
              {o.rotulo}
            </span>
          </button>
        )
      })}
    </div>
  )
}

/* --------------------------------------------------------------- Estados --- */
export function Carregando({ texto = 'Calculando…' }: { texto?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-muted">
      <div className="h-7 w-7 animate-spin rounded-full border-2 border-zinc-500/25 border-t-accent-500" />
      <p className="text-sm">{texto}</p>
    </div>
  )
}

export function ErroBox({ erro }: { erro: unknown }) {
  const msg = erro instanceof Error ? erro.message : String(erro)
  return (
    <div className="rounded-[--radius-card] border border-rose-500/30 bg-rose-500/5 p-5">
      <p className="mb-1 text-sm font-semibold text-rose-600 dark:text-rose-400">
        Não foi possível carregar
      </p>
      <p className="text-sm text-secondary">{msg}</p>
      <p className="mt-3 text-xs text-muted">
        Verifique se o backend está no ar:{' '}
        <code className="rounded bg-zinc-500/10 px-1.5 py-0.5 font-mono">
          uvicorn web_app.backend.main:app --reload
        </code>
      </p>
    </div>
  )
}

export function Vazio({ texto }: { texto: string }) {
  return (
    <div className="flex items-center justify-center rounded-[--radius-card] border border-dashed border-strong py-14 text-sm text-muted">
      {texto}
    </div>
  )
}

/* ---------------------------------------------------------------- Legenda -- */
export function Legenda({
  itens,
  titulo,
  className,
}: {
  itens: {
    cor?: string
    forma?: 'quadrado' | 'circulo' | 'triangulo' | 'linha' | 'linha-tracejada'
    rotulo: string
    descricao?: string
  }[]
  titulo?: string
  className?: string
}) {
  return (
    <div className={cn('flex flex-wrap items-start gap-x-5 gap-y-2', className)}>
      {titulo && <span className="kicker shrink-0 pt-0.5">{titulo}</span>}
      {itens.map((item, i) => (
        <span key={i} className="inline-flex items-center gap-1.5 text-xs">
          <MarcadorLegenda cor={item.cor} forma={item.forma} />
          <span className="text-secondary">{item.rotulo}</span>
          {item.descricao && (
            <span className="text-muted">— {item.descricao}</span>
          )}
        </span>
      ))}
    </div>
  )
}

function MarcadorLegenda({
  cor = '#94a3b8',
  forma = 'quadrado',
}: {
  cor?: string
  forma?: 'quadrado' | 'circulo' | 'triangulo' | 'linha' | 'linha-tracejada'
}) {
  if (forma === 'linha' || forma === 'linha-tracejada') {
    return (
      <svg width="18" height="10" aria-hidden className="shrink-0">
        <line
          x1="0"
          y1="5"
          x2="18"
          y2="5"
          stroke={cor}
          strokeWidth="2"
          strokeDasharray={forma === 'linha-tracejada' ? '4 3' : undefined}
        />
      </svg>
    )
  }
  if (forma === 'triangulo') {
    return (
      <svg width="12" height="12" aria-hidden className="shrink-0">
        <polygon points="6,1 11,11 1,11" fill={cor} />
      </svg>
    )
  }
  return (
    <span
      className={cn(
        'inline-block h-3 w-3 shrink-0 border',
        forma === 'circulo' ? 'rounded-full' : 'rounded-[3px]',
      )}
      style={{ backgroundColor: cor, borderColor: 'rgba(0,0,0,.15)' }}
    />
  )
}

/* ------------------------------------------------------------------ Nota --- */
export function Nota({
  children,
  tom = 'info',
  titulo,
  className,
}: {
  children: ReactNode
  tom?: 'info' | 'atencao' | 'ok'
  titulo?: string
  className?: string
}) {
  const tons = {
    info: 'border-sky-500/30 bg-sky-500/5',
    atencao: 'border-accent-500/30 bg-accent-500/5',
    ok: 'border-emerald-500/30 bg-emerald-500/5',
  }
  const cores = {
    info: 'text-sky-600 dark:text-sky-400',
    atencao: 'text-accent-700 dark:text-accent-400',
    ok: 'text-emerald-600 dark:text-emerald-400',
  }
  return (
    <div
      className={cn('rounded-lg border px-4 py-3 text-sm', tons[tom], className)}
    >
      {titulo && (
        <p className={cn('mb-1 font-semibold', cores[tom])}>{titulo}</p>
      )}
      <div className="text-secondary leading-relaxed">{children}</div>
    </div>
  )
}
