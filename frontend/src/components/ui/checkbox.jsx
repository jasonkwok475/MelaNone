import * as CheckboxPrimitive from '@radix-ui/react-checkbox'
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

export function Checkbox({ className, ...props }) {
  return (
    <CheckboxPrimitive.Root
      className={cn(
        'peer h-4 w-4 shrink-0 rounded border border-[var(--surface-border)]',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
        'data-[state=checked]:border-brand-600 data-[state=checked]:bg-brand-600 data-[state=checked]:text-white',
        className,
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator className="flex items-center justify-center">
        <Check className="h-3.5 w-3.5" />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}
