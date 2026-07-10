import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
  {
    variants: {
      variant: {
        neutral:
          'bg-black/5 text-[var(--text-muted)] dark:bg-white/10',
        benign: 'bg-green-100 text-green-800 dark:bg-green-500/15 dark:text-green-300',
        watch: 'bg-amber-100 text-amber-800 dark:bg-amber-500/15 dark:text-amber-300',
        concern: 'bg-red-100 text-red-800 dark:bg-red-500/15 dark:text-red-300',
        brand: 'bg-brand-100 text-brand-700 dark:bg-brand-500/15 dark:text-brand-300',
      },
    },
    defaultVariants: { variant: 'neutral' },
  },
)

export function Badge({ className, variant, ...props }) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
