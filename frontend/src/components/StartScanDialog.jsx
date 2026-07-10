import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { BODY_SITES, bodySiteLabel } from '@/lib/lesions'

export default function StartScanDialog({ open, onOpenChange, patient, onSubmit, isPending, error }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {open && (
          <ScanForm
            patient={patient}
            onSubmit={onSubmit}
            onCancel={() => onOpenChange(false)}
            isPending={isPending}
            error={error}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}

function ScanForm({ patient, onSubmit, onCancel, isPending, error }) {
  const [bodySite, setBodySite] = useState(BODY_SITES[0])
  const [notes, setNotes] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit({ patient_id: patient.id, body_site: bodySite, notes })
  }

  return (
    <>
      <DialogHeader>
        <DialogTitle>Start scan</DialogTitle>
        <DialogDescription>
          New scan for <span className="font-medium">{patient?.display_id}</span>. Use the same
          body site across scans to enable comparison over time.
        </DialogDescription>
      </DialogHeader>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="body_site">Body site</Label>
          <select
            id="body_site"
            value={bodySite}
            onChange={(e) => setBodySite(e.target.value)}
            className="h-9 rounded-lg border border-[var(--surface-border)] bg-[var(--surface-app)] px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          >
            {BODY_SITES.map((site) => (
              <option key={site} value={site}>
                {bodySiteLabel(site)}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="scan_notes">Notes (optional)</Label>
          <textarea
            id="scan_notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={2}
            className="flex w-full rounded-lg border border-[var(--surface-border)] bg-[var(--surface-app)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          />
          <p className="text-xs text-[var(--text-muted)]">
            Demo tip: include <code>[demo:fail]</code> in notes to exercise the failure path.
          </p>
        </div>

        {error && (
          <p className="text-sm text-concern" role="alert">
            {error.message}
          </p>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit" disabled={isPending}>
            {isPending ? 'Starting…' : 'Start scan'}
          </Button>
        </DialogFooter>
      </form>
    </>
  )
}
