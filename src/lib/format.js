// Formatting of times, metrics and session metadata for the UI.

export function formatTime(s) {
  if (!isFinite(s) || s < 0) return '0:00.0'
  const m = Math.floor(s / 60)
  const sec = (s % 60).toFixed(1).padStart(4, '0')
  return `${m}:${sec}`
}

export function formatDuration(d, unit) {
  if (!isFinite(d) || d < 0) return '—'
  if (unit === 'ms') {
    if (d >= 1000) return (d / 1000).toFixed(2) + 'с'
    return d.toFixed(0) + 'мс'
  }
  return d.toFixed(3) + 'с'
}

export function formatMetric(value, digits = 2, suffix = '') {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  return `${Number(value).toFixed(digits)}${suffix}`
}

export function formatInterval(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  if (Number.isInteger(number)) return String(number)
  return number.toFixed(number < 10 ? 2 : 1).replace(/0+$/, '').replace(/\.$/, '')
}

// The chart reads its Time axis in milliseconds or seconds. Converting between
// the two rounds, so a round trip gives the original value back and the seconds
// carry no float dust (0.596, not 0.5960000000000001).
export function convertTime(value, from, to) {
  if (from === to || value == null) return value
  const n = Number(value)
  if (!Number.isFinite(n)) return value
  return to === 'ms'
    ? Math.round(n * 1000 * 1e6) / 1e6
    : Math.round((n / 1000) * 1e9) / 1e9
}

export function convertTimes(values, from, to) {
  if (from === to || !Array.isArray(values)) return values
  return values.map(v => convertTime(v, from, to))
}

// Unit of a session's Time column, from its largest stamp: no recording in
// seconds runs past an hour, and none in milliseconds is under 3.6 s.
export function inferTimeUnit(tMax) {
  return Number(tMax) > 3600 ? 'ms' : 's'
}

// Unit a saved markup's times are in. Files written before the unit toggle
// converted anything carry the label the chart showed, not the unit the values
// were placed in; a value past an hour cannot be seconds, so it is read as ms.
export function storedTimeUnit(meta, values, fallback) {
  let unit = meta?.timeUnit ?? fallback
  if (unit === 's' && values.some(v => Math.abs(Number(v)) > 3600)) unit = 'ms'
  return unit
}

export function hasSessionMetaValue(value) {
  return value != null && value !== ''
}

// sessions.time_offset is an integer number of milliseconds (the column spans
// -3469..11221 across the DB). The sign is always explicit so a measured zero
// reads as a real value; a NULL is filtered out by the callers instead.
export function formatTimeOffset(value) {
  if (!hasSessionMetaValue(value)) return ''
  const ms = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(ms)) return String(value).trim()
  return `${ms >= 0 ? '+' : ''}${ms} ms`
}

// The S1/S2 shift that sessions.time_offset implies, in the unit the shift boxes
// are currently reading. The sign is flipped: the column records how far the
// device clock ran ahead, so the traces move the other way to meet the video.
// Confirmed by hand on session 7797 (-1753 in the DB, +1753 ms on screen lines
// the video up); the column is otherwise undocumented — the backend schema calls
// it seconds, which the values contradict.
export function timeOffsetAsShift(value, unit) {
  if (!hasSessionMetaValue(value)) return null
  const ms = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(ms)) return null
  const shift = -ms
  return unit === 'ms' ? shift : Math.round(shift) / 1000
}

// The markup API fills an absent athlete with an em dash; the header badge is
// hidden instead of showing a placeholder.
export function normalizeMemberName(value) {
  const name = typeof value === 'string' ? value.trim() : ''
  return name && name !== '—' ? name : ''
}

export function normalizeSessionTitle(value) {
  return typeof value === 'string' ? value.trim() : ''
}
