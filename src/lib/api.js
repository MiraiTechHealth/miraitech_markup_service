// Where the backend and the calculator companion API live.

// ── Constants ──────────────────────────────────────────────────────────────
export const API_BASE = import.meta.env.VITE_API_BASE ?? (
  import.meta.env.DEV ? 'http://localhost:8000' : 'https://dev-api.miraitech.health'
)

export const CALCULATOR_API = import.meta.env.VITE_CALCULATOR_API ?? '/calculator-api'

export const MARKUP_API = `${CALCULATOR_API}/markup`

export function parseApiError(errData, status) {
  const detail = errData?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(d => d.msg || String(d)).join('; ')
  if (detail && typeof detail === 'object') return detail.message || JSON.stringify(detail)
  return `Ошибка ${status}`
}
