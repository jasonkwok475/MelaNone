/**
 * Thin typed-ish fetch wrapper for the MelaNone API.
 * All server data goes through TanStack Query; this is the fetch layer beneath it.
 * Never returns invented fallback data — a failed request throws so the UI can show
 * an explicit error state.
 */
const BASE = '/api'

export async function apiGet(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`)
  }
  return res.json()
}
