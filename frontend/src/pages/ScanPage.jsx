import { useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { AlertTriangle, ArrowLeft, Box, Loader2 } from 'lucide-react'
import { useScan, useScanEvents } from '@/lib/scans'
import { classificationMeta, confidencePct, bodySiteLabel } from '@/lib/lesions'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

const ACTIVE = new Set(['queued', 'running'])

export default function ScanPage() {
  const { scanId } = useParams()
  const qc = useQueryClient()
  const scan = useScan(scanId)

  const isActive = scan.isSuccess && ACTIVE.has(scan.data.status)
  const live = useScanEvents(scanId, {
    enabled: isActive,
    onTerminal: () => qc.invalidateQueries({ queryKey: ['scan', scanId] }),
  })

  if (scan.isLoading) {
    return <p className="text-sm text-[var(--text-muted)]">Loading scan…</p>
  }
  if (scan.isError) {
    return <p className="text-sm text-concern">Couldn’t load scan: {scan.error.message}</p>
  }

  const s = scan.data
  const status = isActive ? live.status : s.status

  return (
    <div className="mx-auto max-w-4xl">
      <Link to="/patients" className="mb-4 inline-flex items-center gap-1.5 text-sm text-[var(--text-muted)] hover:text-brand-600">
        <ArrowLeft className="h-4 w-4" /> Back to patients
      </Link>

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{bodySiteLabel(s.body_site)} scan</h1>
          <p className="mt-1 text-sm text-[var(--text-muted)]">Scan {s.id.slice(0, 8)}</p>
        </div>
        <StatusBadge status={status} />
      </div>

      {(status === 'queued' || status === 'running') && (
        <ProgressView live={live} fallbackProgress={s.progress} />
      )}
      {status === 'failed' && <FailureCard scan={s} liveMessage={live.message} />}
      {status === 'complete' && <ResultsView scan={s} />}
    </div>
  )
}

function StatusBadge({ status }) {
  const map = {
    queued: { variant: 'neutral', label: 'Queued' },
    running: { variant: 'brand', label: 'Running' },
    connecting: { variant: 'neutral', label: 'Connecting' },
    complete: { variant: 'benign', label: 'Complete' },
    failed: { variant: 'concern', label: 'Failed' },
  }
  const m = map[status] ?? { variant: 'neutral', label: status }
  return <Badge variant={m.variant}>{m.label}</Badge>
}

function ProgressView({ live, fallbackProgress }) {
  const pct = Math.max(live.progress ?? 0, fallbackProgress ?? 0)
  return (
    <Card className="mt-6">
      <CardContent className="p-6">
        <div className="mb-2 flex items-center gap-2 text-sm">
          <Loader2 className="h-4 w-4 animate-spin text-brand-600" />
          <span className="font-medium">{live.message ?? 'Starting scan…'}</span>
          <span className="ml-auto text-[var(--text-muted)]">{pct}%</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-black/5 dark:bg-white/10">
          <div
            className="h-full rounded-full bg-brand-600 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
        {live.step && (
          <p className="mt-2 text-xs uppercase tracking-wide text-[var(--text-muted)]">
            Stage: {live.step}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function FailureCard({ scan, liveMessage }) {
  return (
    <Card className="mt-6 border-red-300 dark:border-red-500/40">
      <CardContent className="flex items-start gap-3 p-6">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-concern" />
        <div>
          <p className="font-medium text-concern">Scan failed</p>
          <p className="mt-1 text-sm text-[var(--text-muted)]">
            Stage <span className="font-medium">{scan.failure_stage ?? '—'}</span>:{' '}
            {scan.failure_reason ?? liveMessage ?? 'Unknown error'}
          </p>
          <p className="mt-3 text-xs text-[var(--text-muted)]">
            No results were produced — the scan is not usable. Start a new scan to try again.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

function ResultsView({ scan }) {
  const concerning = scan.concerning_count > 0
  return (
    <div className="mt-6 flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Spots analyzed" value={scan.total_lesions} />
        <Stat
          label="Flagged concerning"
          value={scan.concerning_count}
          tone={concerning ? 'concern' : 'benign'}
        />
        <Stat label="Model" value={scan.model_version ?? '—'} small />
        <Stat label="Vertices" value={scan.vertex_count ?? '—'} />
      </div>

      <Card>
        <CardContent className="flex flex-col gap-4 p-6 sm:flex-row">
          {scan.thumbnail_url && (
            <img
              src={scan.thumbnail_url}
              alt="Scan surface preview"
              className="h-40 w-40 shrink-0 rounded-lg border border-[var(--surface-border)] object-cover"
            />
          )}
          <div className="flex flex-col justify-center gap-2">
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <Box className="h-4 w-4" />
              Interactive 3D model with lesion markers arrives in the next milestone.
            </div>
            <p className="text-xs text-[var(--text-muted)]">
              Results are algorithmic estimates for research/education — not a medical diagnosis.
            </p>
          </div>
        </CardContent>
      </Card>

      <div>
        <h2 className="mb-2 text-sm font-semibold text-[var(--text-muted)]">
          Detected spots ({scan.lesions.length})
        </h2>
        <div className="rounded-[var(--radius-card)] border border-[var(--surface-border)] bg-[var(--surface-card)]">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Classification</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>UV position</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {scan.lesions.map((le) => {
                const meta = classificationMeta(le.classification)
                return (
                  <TableRow key={le.id}>
                    <TableCell>
                      <Badge variant={meta.variant}>{meta.label}</Badge>
                    </TableCell>
                    <TableCell>{confidencePct(le.confidence)}</TableCell>
                    <TableCell className="text-[var(--text-muted)]">
                      {le.uv_x.toFixed(2)}, {le.uv_y.toFixed(2)}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, tone, small }) {
  const toneClass =
    tone === 'concern' ? ' text-concern' : tone === 'benign' ? ' text-benign' : ''
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-xs text-[var(--text-muted)]">{label}</p>
        <p className={'mt-1 font-semibold ' + (small ? 'text-sm' : 'text-2xl') + toneClass}>
          {value}
        </p>
      </CardContent>
    </Card>
  )
}
