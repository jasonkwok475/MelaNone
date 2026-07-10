import { Link } from 'react-router-dom'
import { Box } from 'lucide-react'
import { useScans } from '@/lib/scans'
import { bodySiteLabel } from '@/lib/lesions'
import { Badge } from '@/components/ui/badge'

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

export default function ViewerPage() {
  const scans = useScans()
  const complete = (scans.data ?? []).filter((s) => s.status === 'complete')

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-2xl font-semibold">3D Viewer</h1>
      <p className="mt-1 text-sm text-[var(--text-muted)]">
        Open a completed scan to explore its interactive 3D reconstruction with lesion markers.
      </p>

      <div className="mt-6 rounded-[var(--radius-card)] border border-[var(--surface-border)] bg-[var(--surface-card)]">
        {scans.isLoading && (
          <p className="p-6 text-sm text-[var(--text-muted)]">Loading scans…</p>
        )}
        {scans.isError && (
          <p className="p-6 text-sm text-concern">Couldn’t load scans: {scans.error.message}</p>
        )}
        {scans.isSuccess && complete.length === 0 && (
          <div className="flex flex-col items-center gap-2 p-12 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-600 dark:bg-brand-500/10">
              <Box className="h-6 w-6" />
            </div>
            <p className="font-medium">No completed scans yet</p>
            <p className="text-sm text-[var(--text-muted)]">
              Run a scan from{' '}
              <Link to="/patients" className="text-brand-600 hover:underline">
                Patients &amp; Scan
              </Link>{' '}
              to view a 3D model here.
            </p>
          </div>
        )}
        {scans.isSuccess &&
          complete.map((s) => (
            <Link
              key={s.id}
              to={`/scans/${s.id}`}
              className="flex items-center gap-4 border-b border-[var(--surface-border)] p-4 last:border-b-0 hover:bg-black/5 dark:hover:bg-white/5"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-500/10">
                <Box className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium">{bodySiteLabel(s.body_site)}</p>
                <p className="text-xs text-[var(--text-muted)]">
                  {formatDate(s.completed_at ?? s.created_at)} · {s.total_lesions} spots
                </p>
              </div>
              {s.concerning_count > 0 ? (
                <Badge variant="concern">{s.concerning_count} concerning</Badge>
              ) : (
                <Badge variant="benign">None flagged</Badge>
              )}
            </Link>
          ))}
      </div>

      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Results are algorithmic estimates for research/education — not a medical diagnosis.
      </p>
    </div>
  )
}
