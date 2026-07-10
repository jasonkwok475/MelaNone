import { cn } from '@/lib/utils'

export function Table({ className, ...props }) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn('w-full caption-bottom text-sm', className)} {...props} />
    </div>
  )
}

export function TableHeader({ className, ...props }) {
  return <thead className={cn('[&_tr]:border-b [&_tr]:border-[var(--surface-border)]', className)} {...props} />
}

export function TableBody({ className, ...props }) {
  return <tbody className={cn('[&_tr:last-child]:border-0', className)} {...props} />
}

export function TableRow({ className, ...props }) {
  return (
    <tr
      className={cn(
        'border-b border-[var(--surface-border)] transition-colors hover:bg-black/[0.02] dark:hover:bg-white/[0.03]',
        className,
      )}
      {...props}
    />
  )
}

export function TableHead({ className, ...props }) {
  return (
    <th
      className={cn(
        'h-10 px-3 text-left align-middle text-xs font-medium text-[var(--text-muted)]',
        className,
      )}
      {...props}
    />
  )
}

export function TableCell({ className, ...props }) {
  return <td className={cn('px-3 py-2.5 align-middle', className)} {...props} />
}
