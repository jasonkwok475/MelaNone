import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  Box,
  GitCompare,
  History,
  LayoutDashboard,
  Settings,
  SlidersHorizontal,
  Users,
} from 'lucide-react'
import { apiGet } from '@/lib/api'
import { cn } from '@/lib/utils'
import DisclaimerBadge from '@/components/DisclaimerBadge'
import PatientsPage from '@/pages/PatientsPage'
import ScanPage from '@/pages/ScanPage'
import ViewerPage from '@/pages/ViewerPage'

// Sections from docs/04. Each routes to a placeholder until its milestone lands.
const NAV = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, milestone: null },
  { to: '/controls', label: 'Controls', icon: SlidersHorizontal, milestone: 6 },
  { to: '/patients', label: 'Patients & Scan', icon: Users, milestone: 1 },
  { to: '/viewer', label: '3D Viewer', icon: Box, milestone: 4 },
  { to: '/history', label: 'History', icon: History, milestone: 8 },
  { to: '/compare', label: 'Compare', icon: GitCompare, milestone: 8 },
  { to: '/settings', label: 'Settings', icon: Settings, milestone: 9 },
]

function Sidebar() {
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-[var(--surface-border)] bg-[var(--surface-card)]">
      <div className="flex items-center gap-2 px-5 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white">
          <Activity className="h-5 w-5" aria-hidden />
        </div>
        <span className="text-lg font-semibold tracking-tight">MelaNone</span>
      </div>
      <nav className="flex flex-col gap-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300'
                  : 'text-[var(--text-muted)] hover:bg-black/5 dark:hover:bg-white/5',
              )
            }
          >
            <Icon className="h-4 w-4" aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}

function Topbar() {
  const health = useQuery({ queryKey: ['health'], queryFn: () => apiGet('/health') })
  const backendOk = health.isSuccess

  return (
    <header className="flex items-center justify-between gap-4 border-b border-[var(--surface-border)] bg-[var(--surface-card)] px-6 py-3">
      <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
        <span
          className={cn(
            'inline-block h-2 w-2 rounded-full',
            health.isLoading && 'bg-slate-400',
            backendOk && 'bg-benign',
            health.isError && 'bg-concern',
          )}
        />
        {health.isLoading && 'Connecting to API…'}
        {backendOk && `API online · v${health.data.version}`}
        {health.isError && 'API offline'}
        {backendOk && health.data.demo_mode && (
          <span className="rounded bg-brand-100 px-1.5 py-0.5 text-xs font-medium text-brand-700 dark:bg-brand-500/15 dark:text-brand-300">
            DEMO MODE
          </span>
        )}
      </div>
      <DisclaimerBadge />
    </header>
  )
}

function Placeholder({ title, milestone }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      <h1 className="text-2xl font-semibold">{title}</h1>
      <p className="max-w-md text-sm text-[var(--text-muted)]">
        This section is part of the planned build and arrives in milestone {milestone}.
      </p>
    </div>
  )
}

function Dashboard() {
  const config = useQuery({ queryKey: ['config'], queryFn: () => apiGet('/config') })

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      <p className="mt-1 text-sm text-[var(--text-muted)]">
        Scaffold is up. Sections populate over the coming milestones.
      </p>

      <div className="mt-6 rounded-[var(--radius-card)] border border-[var(--surface-border)] bg-[var(--surface-card)] p-5">
        <h2 className="text-sm font-semibold text-[var(--text-muted)]">Backend connectivity</h2>
        {config.isLoading && <p className="mt-2 text-sm">Loading configuration…</p>}
        {config.isError && (
          <p className="mt-2 text-sm text-concern">
            Could not reach the API. Is the backend running on port 8000?
          </p>
        )}
        {config.isSuccess && (
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <div>
              <dt className="text-[var(--text-muted)]">App</dt>
              <dd className="font-medium">{config.data.app_name}</dd>
            </div>
            <div>
              <dt className="text-[var(--text-muted)]">Demo mode</dt>
              <dd className="font-medium">{config.data.demo_mode ? 'on' : 'off'}</dd>
            </div>
            <div>
              <dt className="text-[var(--text-muted)]">Rotation steps</dt>
              <dd className="font-medium">{config.data.rotation_steps}</dd>
            </div>
            <div>
              <dt className="text-[var(--text-muted)]">Cameras</dt>
              <dd className="font-medium">{config.data.camera_indices.join(', ')}</dd>
            </div>
          </dl>
        )}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/patients" element={<PatientsPage />} />
            <Route path="/scans/:scanId" element={<ScanPage />} />
            <Route path="/viewer" element={<ViewerPage />} />
            {NAV.filter((n) => n.milestone && !['/patients', '/viewer'].includes(n.to)).map(
              ({ to, label, milestone }) => (
                <Route
                  key={to}
                  path={to}
                  element={<Placeholder title={label} milestone={milestone} />}
                />
              ),
            )}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
