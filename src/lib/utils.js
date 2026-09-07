// Small numeric helpers shared by the other modules.

// Math.max/min(...array) blows the call stack on long sessions (V8 caps spread
// argument count well below typical sample counts) — reduce instead.
export function arrayMax(arr) {
  let m = -Infinity
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]
  return m
}

export function arrayMin(arr) {
  let m = Infinity
  for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i]
  return m
}

export function medianOf(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

export function safeNum(v) {
  if (v === null || v === undefined) return null
  const n = typeof v === 'bigint' ? Number(v) : Number(v)
  return isFinite(n) ? n : null
}

export function unwrapAngleDegrees(arr, threshold = 180.0) {
  if (!arr || arr.length === 0) return arr
  const result = new Array(arr.length)
  result[0] = arr[0]
  let offset = 0
  for (let i = 1; i < arr.length; i++) {
    const curr = safeNum(arr[i])
    const prev = safeNum(arr[i - 1])
    if (curr === null || prev === null) { result[i] = arr[i]; continue }
    const diff = curr - prev
    if (diff > threshold) offset -= 360
    else if (diff < -threshold) offset += 360
    result[i] = arr[i] + offset
  }
  return result
}

// np.median — the mean of the two middle values on even lengths.
export function median(values) {
  const n = values.length
  if (!n) return NaN
  const sorted = [...values].sort((a, b) => a - b)
  const mid = n >> 1
  return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// Population standard deviation — divides by N, matching np.std.
export function populationStd(values) {
  const n = values.length
  if (!n) return NaN
  let sum = 0
  for (let i = 0; i < n; i++) sum += values[i]
  const mean = sum / n
  let acc = 0
  for (let i = 0; i < n; i++) { const d = values[i] - mean; acc += d * d }
  return Math.sqrt(acc / n)
}

// Linear interpolation of (xp, fp) at each x, CLAMPED at both ends. Never
// extrapolates: outside the curve's range there is no evidence, and both feet
// holding the same constant is what makes the correction cancel out of the body
// average. `x` must be ascending — a cursor walks xp alongside it.
export function interpClamped(x, xp, fp) {
  const n = xp.length
  const out = new Float64Array(x.length)
  if (!n) { out.fill(NaN); return out }
  let cursor = 0
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    if (v <= xp[0]) { out[i] = fp[0]; continue }
    if (v >= xp[n - 1]) { out[i] = fp[n - 1]; continue }
    while (cursor < n - 2 && xp[cursor + 1] <= v) cursor++
    const span = xp[cursor + 1] - xp[cursor]
    out[i] = span === 0
      ? fp[cursor]
      : fp[cursor] + (fp[cursor + 1] - fp[cursor]) * ((v - xp[cursor]) / span)
  }
  return out
}
