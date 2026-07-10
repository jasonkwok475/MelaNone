/**
 * Minimal print-to-PDF report for a completed scan. Opens a self-contained window with a clean
 * summary and triggers the browser print dialog. A fuller report (branding, images) lands in M9.
 *
 * Always carries the "not a diagnosis" disclaimer — required on every result surface.
 */
import { classificationMeta, confidencePct, bodySiteLabel } from '@/lib/lesions'

function esc(value) {
  return String(value ?? '').replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ))
}

function fmtDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

export function openScanReport(scan) {
  const rows = scan.lesions
    .map((le) => {
      const meta = classificationMeta(le.classification)
      return `<tr>
        <td>${esc(meta.label)}</td>
        <td>${esc(confidencePct(le.confidence))}</td>
        <td>${le.uv_x.toFixed(2)}, ${le.uv_y.toFixed(2)}</td>
      </tr>`
    })
    .join('')

  const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>MelaNone scan report — ${esc(scan.id.slice(0, 8))}</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #1e293b; margin: 40px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .muted { color: #64748b; font-size: 13px; }
  .disclaimer { border: 1px solid #f59e0b; background: #fffbeb; color: #92400e;
    padding: 10px 12px; border-radius: 8px; font-size: 13px; margin: 16px 0; }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 14px; margin: 16px 0; }
  dt { color: #64748b; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e2e8f0; }
  th { color: #64748b; font-weight: 600; }
  @media print { body { margin: 16px; } button { display: none; } }
</style>
</head>
<body>
  <h1>MelaNone scan report</h1>
  <p class="muted">Research / educational tool — generated ${esc(fmtDate(new Date().toISOString()))}</p>

  <div class="disclaimer">
    <strong>Not a medical diagnosis.</strong> These results are algorithmic estimates produced by a
    research/educational tool and must not be used for clinical decisions. Consult a qualified
    clinician about any skin concern.
  </div>

  <dl>
    <dt>Scan ID</dt><dd>${esc(scan.id)}</dd>
    <dt>Body site</dt><dd>${esc(bodySiteLabel(scan.body_site))}</dd>
    <dt>Created</dt><dd>${esc(fmtDate(scan.created_at))}</dd>
    <dt>Completed</dt><dd>${esc(fmtDate(scan.completed_at))}</dd>
    <dt>Model version</dt><dd>${esc(scan.model_version ?? '—')}</dd>
    <dt>Spots analyzed</dt><dd>${esc(scan.total_lesions)}</dd>
    <dt>Flagged concerning</dt><dd>${esc(scan.concerning_count)}</dd>
  </dl>

  <h2 style="font-size:15px;">Detected spots</h2>
  <table>
    <thead><tr><th>Classification</th><th>Confidence</th><th>UV position</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="3" class="muted">No spots detected.</td></tr>'}</tbody>
  </table>

  <script>window.addEventListener('load', () => window.print())</script>
</body>
</html>`

  const win = window.open('', '_blank')
  if (!win) return // popup blocked; user can retry
  win.document.open()
  win.document.write(html)
  win.document.close()
}
