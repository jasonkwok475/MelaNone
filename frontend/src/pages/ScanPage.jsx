import { Suspense, lazy, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import { AlertTriangle, ArrowLeft, FileDown, Loader2 } from 'lucide-react'
import { useScan, useScanEvents } from '@/lib/scans'
import { classificationMeta, confidencePct, bodySiteLabel } from '@/lib/lesions'
import { openScanReport } from '@/lib/report'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

// three.js is heavy; only pull it in when a completed scan actually renders the viewer.
const LimbViewer = lazy(() => import('@/components/LimbViewer'))
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'

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
  const [selectedId, setSelectedId] = useState(null)
  const canRender3d = !!(scan.mesh_url && scan.texture_url)
  const selected = scan.lesions.find((le) => le.id === selectedId) ?? null

  const toggleSelect = (id) => setSelectedId((cur) => (cur === id ? null : id))

  return (
    <div className="mt-6 flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4">
        <div className="grid flex-1 grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Spots analyzed" value={scan.total_lesions} />
          <Stat
            label="Flagged concerning"
            value={scan.concerning_count}
            tone={concerning ? 'concern' : 'benign'}
          />
          <Stat label="Model" value={scan.model_version ?? '—'} small />
          <Stat label="Vertices" value={scan.vertex_count ?? '—'} />
        </div>
        <Button variant="outline" size="sm" onClick={() => openScanReport(scan)}>
          <FileDown /> Export report
        </Button>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div>
          {canRender3d ? (
            <Suspense
              fallback={
                <div className="flex h-[440px] items-center justify-center rounded-[var(--radius-card)] border border-[var(--surface-border)] text-sm text-[var(--text-muted)]">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading 3D viewer…
                </div>
              }
            >
              <LimbViewer
                meshUrl={scan.mesh_url}
                textureUrl={scan.texture_url}
                lesions={scan.lesions}
                selectedId={selectedId}
                onSelectLesion={toggleSelect}
              />
            </Suspense>
          ) : (
            <Card>
              <CardContent className="flex items-center gap-2 p-6 text-sm text-[var(--text-muted)]">
                <AlertTriangle className="h-4 w-4 text-watch" />
                No 3D mesh artifact is available for this scan.
              </CardContent>
            </Card>
          )}
          <p className="mt-2 text-xs text-[var(--text-muted)]">
            Interactive 3D reconstruction with lesion markers — rotate, pan, and zoom; click a
            marker for details. Results are algorithmic estimates for research/education,{' '}
            <span className="font-medium">not a medical diagnosis</span>.
          </p>
        </div>

        <LesionDetailPanel lesion={selected} onClear={() => setSelectedId(null)} />
      </div>

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
                  <TableRow
                    key={le.id}
                    onClick={() => toggleSelect(le.id)}
                    className={cn(
                      'cursor-pointer',
                      le.id === selectedId && 'bg-brand-50 dark:bg-brand-500/10',
                    )}
                  >
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

function LesionDetailPanel({ lesion, onClear }) {
  if (!lesion) {
    return (
      <Card className="h-fit">
        <CardContent className="p-5 text-sm text-[var(--text-muted)]">
          Select a marker or a table row to inspect a spot.
        </CardContent>
      </Card>
    )
  }
  const meta = classificationMeta(lesion.classification)
  return (
    <Card className="h-fit">
      <CardContent className="flex flex-col gap-3 p-5">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Spot detail</h3>
          <button
            type="button"
            onClick={onClear}
            className="text-xs text-[var(--text-muted)] hover:text-brand-600"
          >
            Clear
          </button>
        </div>
        <div>
          <Badge variant={meta.variant}>{meta.label}</Badge>
        </div>
        <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1.5 text-sm">
          <dt className="text-[var(--text-muted)]">Confidence</dt>
          <dd>{confidencePct(lesion.confidence)}</dd>
          <dt className="text-[var(--text-muted)]">UV</dt>
          <dd>
            {lesion.uv_x.toFixed(3)}, {lesion.uv_y.toFixed(3)}
          </dd>
          <dt className="text-[var(--text-muted)]">Position</dt>
          <dd>
            {lesion.x.toFixed(2)}, {lesion.y.toFixed(2)}, {lesion.z.toFixed(2)}
          </dd>
          {lesion.area != null && (
            <>
              <dt className="text-[var(--text-muted)]">Area</dt>
              <dd>{lesion.area.toFixed(3)}</dd>
            </>
          )}
        </dl>
        <p className="text-xs text-[var(--text-muted)]">
          Estimated classification and confidence — not a diagnosis.
        </p>
      </CardContent>
    </Card>
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
