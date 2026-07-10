/**
 * TanStack Query hooks for the Patients API (mirrors backend/app/schemas/patient.py).
 * Mutations invalidate the list so the UI stays in sync with the server — no local
 * optimistic guesses that could diverge from the backend.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiGet, apiPatch, apiPost } from '@/lib/api'

const patientsKey = ['patients']

export function usePatients() {
  return useQuery({ queryKey: patientsKey, queryFn: () => apiGet('/patients') })
}

export function useCreatePatient() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload) => apiPost('/patients', payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: patientsKey }),
  })
}

export function useUpdatePatient() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...payload }) => apiPatch(`/patients/${id}`, payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: patientsKey }),
  })
}

export function useDeletePatient() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id) => apiDelete(`/patients/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: patientsKey }),
  })
}
