import { ShieldAlert } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * Persistent, unmissable "not a medical diagnosis" badge.
 * Guardrail: this must stay visible on the shell and on every results/report view.
 */
export default function DisclaimerBadge({ className }) {
  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50',
        'px-3 py-1 text-xs font-medium text-amber-800',
        'dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-300',
        className,
      )}
      role="note"
    >
      <ShieldAlert className="h-3.5 w-3.5 shrink-0" aria-hidden />
      Research / educational use only — not a medical diagnosis
    </div>
  )
}
