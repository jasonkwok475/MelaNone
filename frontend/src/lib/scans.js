/**
 * Scans API hooks: list/get/create/delete + a live SSE progress hook.
 */
import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiGet, apiPost } from '@/lib/api'

export function useScans(patientId) {
  return useQuery({
    queryKey: ['scans', { patientId: patientId ?? null }],
    queryFn: () => apiGet(patientId ? `/scans?patient_id=${patientId}` : '/scans'),
  })
}

export function useScan(scanId, options = {}) {
  return useQuery({
    queryKey: ['scan', scanId],
    queryFn: () => apiGet(`/scans/${scanId}`),
    enabled: !!scanId,
    ...options,
  })
}

export function useCreateScan() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload) => apiPost('/scans', payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scans'] }),
  })
}

export function useDeleteScan() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id) => apiDelete(`/scans/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scans'] }),
  })
}

/**
 * Subscribe to a scan's SSE progress stream. Returns live { status, progress, step, message }.
 * Closes on the terminal event and invokes onTerminal('complete'|'failed').
 */
export function useScanEvents(scanId, { enabled = true, onTerminal } = {}) {
  const [state, setState] = useState({
    status: 'connecting',
    progress: 0,
    step: null,
    message: null,
  })
  const onTerminalRef = useRef(onTerminal)
  useEffect(() => {
    onTerminalRef.current = onTerminal
  }, [onTerminal])

  useEffect(() => {
    if (!scanId || !enabled) return undefined

    const es = new EventSource(`/api/scans/${scanId}/events`)

    const handleProgress = (e) => {
      const d = JSON.parse(e.data)
      setState({ status: 'running', progress: d.progress, step: d.stage, message: d.message })
    }
    const handleComplete = (e) => {
      const d = JSON.parse(e.data)
      setState({ status: 'complete', progress: 100, step: d.stage, message: d.message })
      es.close()
      onTerminalRef.current?.('complete')
    }
    const handleFailed = (e) => {
      const d = JSON.parse(e.data)
      setState({ status: 'failed', progress: 0, step: d.stage, message: d.reason })
      es.close()
      onTerminalRef.current?.('failed')
    }

    es.addEventListener('progress', handleProgress)
    es.addEventListener('complete', handleComplete)
    es.addEventListener('failed', handleFailed)

    return () => es.close()
  }, [scanId, enabled])

  return state
}
