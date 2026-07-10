/**
 * Thin fetch wrapper for the MelaNone API.
 * All server data goes through TanStack Query; this is the fetch layer beneath it.
 * Never returns invented fallback data — a failed request throws so the UI can show
 * an explicit error state.
 */
const BASE = '/api'

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      // non-JSON error body — keep the status line
    }
    throw new Error(detail)
  }
  if (res.status === 204) return null
  return res.json()
}

export function apiGet(path) {
  return request(path)
}

export function apiPost(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function apiPatch(path, body) {
  return request(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function apiDelete(path) {
  return request(path, { method: 'DELETE' })
}
