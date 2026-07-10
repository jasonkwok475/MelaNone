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
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'

// Consent wording (short version, chosen by the operator). This is acknowledgement copy,
// not a diagnosis claim.
const CONSENT_LABEL =
  "I consent to local storage of this patient's scans for research/educational use."

const EMPTY = {
  display_id: '',
  name: '',
  sex: '',
  date_of_birth: '',
  notes: '',
  consent_ack: false,
}

function fromPatient(p) {
  if (!p) return { ...EMPTY }
  return {
    display_id: p.display_id ?? '',
    name: p.name ?? '',
    sex: p.sex ?? '',
    date_of_birth: p.date_of_birth ?? '',
    notes: p.notes ?? '',
    consent_ack: !!p.consent_ack,
  }
}

/** Serialize the form to the API payload, coercing empty optional strings to null. */
function toPayload(form) {
  return {
    display_id: form.display_id.trim(),
    name: form.name.trim() || null,
    sex: form.sex.trim() || null,
    date_of_birth: form.date_of_birth || null,
    notes: form.notes,
    consent_ack: form.consent_ack,
  }
}

export default function PatientFormDialog({ open, onOpenChange, patient, onSubmit, isPending, error }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {/*
          The inner form is keyed so it remounts each time the dialog opens for a
          (possibly different) patient — its useState initializer re-seeds from `patient`
          without a state-syncing effect.
        */}
        {open && (
          <PatientForm
            key={patient?.id ?? 'new'}
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

function PatientForm({ patient, onSubmit, onCancel, isPending, error }) {
  const [form, setForm] = useState(() => fromPatient(patient))
  const isEdit = !!patient
  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(toPayload(form))
  }

  return (
    <>
      <DialogHeader>
        <DialogTitle>{isEdit ? 'Edit patient' : 'New patient'}</DialogTitle>
        <DialogDescription>
          Use a de-identified ID where possible. Name and other details are optional.
        </DialogDescription>
      </DialogHeader>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="display_id">Display ID *</Label>
            <Input
              id="display_id"
              value={form.display_id}
              onChange={set('display_id')}
              placeholder="e.g. P-001"
              required
              autoFocus
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="name">Name (optional)</Label>
              <Input id="name" value={form.name} onChange={set('name')} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="sex">Sex (optional)</Label>
              <Input id="sex" value={form.sex} onChange={set('sex')} />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="dob">Date of birth (optional)</Label>
            <Input id="dob" type="date" value={form.date_of_birth} onChange={set('date_of_birth')} />
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="notes">Notes</Label>
            <textarea
              id="notes"
              value={form.notes}
              onChange={set('notes')}
              rows={2}
              className="flex w-full rounded-lg border border-[var(--surface-border)] bg-[var(--surface-app)] px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
            />
          </div>

          <label className="flex items-start gap-2.5 rounded-lg border border-[var(--surface-border)] p-3 text-sm">
            <Checkbox
              checked={form.consent_ack}
              onCheckedChange={(v) => setForm((f) => ({ ...f, consent_ack: !!v }))}
              className="mt-0.5"
            />
            <span className="text-[var(--text-muted)]">{CONSENT_LABEL}</span>
          </label>

          {error && (
            <p className="text-sm text-concern" role="alert">
              {error.message}
            </p>
          )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit" disabled={isPending || !form.display_id.trim()}>
            {isPending ? 'Saving…' : isEdit ? 'Save changes' : 'Create patient'}
          </Button>
        </DialogFooter>
      </form>
    </>
  )
}
