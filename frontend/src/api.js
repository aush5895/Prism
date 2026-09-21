// The only place that knows the API exists. Nothing here reimplements pipeline logic:
// every number and every verdict the UI shows is computed by the backend and read off
// the response verbatim.

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function json(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`${options?.method || 'GET'} ${path} -> ${response.status}: ${body.slice(0, 300)}`)
  }
  return response.json()
}

export function fetchSamples() {
  return json('/v1/samples')
}

export function troubleshoot(query, siisResponse) {
  return json('/v1/troubleshoot/debug', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, siis_response: siisResponse }),
  })
}

export function health() {
  return json('/health')
}
