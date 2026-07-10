import { useState } from 'react'
import { Pencil, Plus, Trash2, UserPlus, Users } from 'lucide-react'
import {
  useCreatePatient,
  useDeletePatient,
  usePatients,
  useUpdatePatient,
} from '@/lib/patients'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import PatientFormDialog from '@/components/PatientFormDialog'

function formatDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString()
}

export default function PatientsPage() {
  const patients = usePatients()
  const createPatient = useCreatePatient()
  const updatePatient = useUpdatePatient()
  const deletePatient = useDeletePatient()

  const [formOpen, setFormOpen] = useState(false)
  const [editing, setEditing] = useState(null) // patient being edited, or null for create
  const [pendingDelete, setPendingDelete] = useState(null)

  const openCreate = () => {
    setEditing(null)
    createPatient.reset()
    updatePatient.reset()
    setFormOpen(true)
  }
  const openEdit = (p) => {
    setEditing(p)
    createPatient.reset()
    updatePatient.reset()
    setFormOpen(true)
  }

  const activeMutation = editing ? updatePatient : createPatient

  const handleSubmit = (payload) => {
    const mutation = editing
      ? updatePatient.mutateAsync({ id: editing.id, ...payload })
      : createPatient.mutateAsync(payload)
    mutation.then(() => setFormOpen(false)).catch(() => {})
  }

  const confirmDelete = () => {
    if (!pendingDelete) return
    deletePatient.mutateAsync(pendingDelete.id).finally(() => setPendingDelete(null))
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Patients</h1>
          <p className="mt-1 text-sm text-[var(--text-muted)]">
            Manage patient records. Prefer de-identified IDs; data is stored locally.
          </p>
        </div>
        <Button onClick={openCreate}>
          <Plus /> New patient
        </Button>
      </div>

      <div className="mt-6 rounded-[var(--radius-card)] border border-[var(--surface-border)] bg-[var(--surface-card)]">
        {patients.isLoading && (
          <p className="p-6 text-sm text-[var(--text-muted)]">Loading patients…</p>
        )}

        {patients.isError && (
          <div className="p-6">
            <p className="text-sm text-concern">
              Couldn’t load patients: {patients.error.message}
            </p>
            <Button variant="outline" size="sm" className="mt-3" onClick={() => patients.refetch()}>
              Retry
            </Button>
          </div>
        )}

        {patients.isSuccess && patients.data.length === 0 && (
          <div className="flex flex-col items-center gap-3 p-12 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-600 dark:bg-brand-500/10">
              <Users className="h-6 w-6" />
            </div>
            <div>
              <p className="font-medium">No patients yet</p>
              <p className="text-sm text-[var(--text-muted)]">
                Create your first patient record to start scanning.
              </p>
            </div>
            <Button onClick={openCreate}>
              <UserPlus /> Add patient
            </Button>
          </div>
        )}

        {patients.isSuccess && patients.data.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Display ID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Consent</TableHead>
                <TableHead>Created</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {patients.data.map((p) => (
                <TableRow key={p.id}>
                  <TableCell className="font-medium">{p.display_id}</TableCell>
                  <TableCell className="text-[var(--text-muted)]">{p.name || '—'}</TableCell>
                  <TableCell>
                    {p.consent_ack ? (
                      <Badge variant="benign">Acknowledged</Badge>
                    ) : (
                      <Badge variant="watch">Not acknowledged</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-[var(--text-muted)]">{formatDate(p.created_at)}</TableCell>
                  <TableCell>
                    <div className="flex justify-end gap-1">
                      <Button variant="ghost" size="icon" onClick={() => openEdit(p)} aria-label="Edit">
                        <Pencil />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setPendingDelete(p)}
                        aria-label="Delete"
                      >
                        <Trash2 className="text-concern" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      <PatientFormDialog
        open={formOpen}
        onOpenChange={setFormOpen}
        patient={editing}
        onSubmit={handleSubmit}
        isPending={activeMutation.isPending}
        error={activeMutation.error}
      />

      <AlertDialog open={!!pendingDelete} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete patient?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes {pendingDelete?.display_id} and all associated scans and
              artifacts. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete} disabled={deletePatient.isPending}>
              {deletePatient.isPending ? 'Deleting…' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
