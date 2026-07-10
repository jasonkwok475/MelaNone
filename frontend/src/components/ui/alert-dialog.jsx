import * as AlertDialogPrimitive from '@radix-ui/react-alert-dialog'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'

export const AlertDialog = AlertDialogPrimitive.Root
export const AlertDialogTrigger = AlertDialogPrimitive.Trigger

export function AlertDialogContent({ className, children, ...props }) {
  return (
    <AlertDialogPrimitive.Portal>
      <AlertDialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/50" />
      <AlertDialogPrimitive.Content
        className={cn(
          'fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2',
          'rounded-[var(--radius-card)] border border-[var(--surface-border)] bg-[var(--surface-card)] p-6 shadow-lg',
          className,
        )}
        {...props}
      >
        {children}
      </AlertDialogPrimitive.Content>
    </AlertDialogPrimitive.Portal>
  )
}

export function AlertDialogHeader({ className, ...props }) {
  return <div className={cn('mb-2 flex flex-col gap-1', className)} {...props} />
}

export function AlertDialogTitle({ className, ...props }) {
  return (
    <AlertDialogPrimitive.Title className={cn('text-lg font-semibold', className)} {...props} />
  )
}

export function AlertDialogDescription({ className, ...props }) {
  return (
    <AlertDialogPrimitive.Description
      className={cn('text-sm text-[var(--text-muted)]', className)}
      {...props}
    />
  )
}

export function AlertDialogFooter({ className, ...props }) {
  return <div className={cn('mt-6 flex justify-end gap-2', className)} {...props} />
}

export function AlertDialogCancel({ className, ...props }) {
  return (
    <AlertDialogPrimitive.Cancel
      className={cn(buttonVariants({ variant: 'outline' }), className)}
      {...props}
    />
  )
}

export function AlertDialogAction({ className, ...props }) {
  return (
    <AlertDialogPrimitive.Action
      className={cn(buttonVariants({ variant: 'destructive' }), className)}
      {...props}
    />
  )
}
