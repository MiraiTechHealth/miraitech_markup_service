// Derived signal channels: TKEO, insole sums, calibration, weighted total, gaps.
import { SENSOR_COLS, inferSensorFoot } from './sensors.js'
import { arrayMax, medianOf, safeNum } from './utils.js'

export const TKEO_WIN = 15

export const TKEO_PLOT_COLS = ['TKEO_AcX', 'TKEO_AcY', 'TKEO_AcZ', 'TKEO_AccMag']

export const SENSOR_SUM_RAW_COL = 'Sensor_Sum_Raw'

export const SENSOR_SUM_NORM_COL = 'Sensor_Sum_Normalized'

// pandas Series.rolling(win, center=True, min_periods=1).mean() over TKEO psi.
export function tkeoSeries(x, win = TKEO_WIN) {
  const m = x.length
  const psi = new Array(m).fill(0)
  if (m >= 3) {
    for (let k = 1; k < m - 1; k++) psi[k] = x[k] * x[k] - x[k - 1] * x[k + 1]
  }
  const back = Math.floor(win / 2)
  const fwd = Math.floor((win - 1) / 2)
  const cum = new Array(m + 1)
  cum[0] = 0
  for (let k = 0; k < m; k++) cum[k + 1] = cum[k] + psi[k]
  const out = new Array(m)
  for (let k = 0; k < m; k++) {
    const lo = Math.max(0, k - back)
    const hi = Math.min(m - 1, k + fwd)
    out[k] = Math.max((cum[hi + 1] - cum[lo]) / (hi - lo + 1), 0)
  }
  return out
}

// Derived channel: TKEO of the accel magnitude, mirroring the backend
// (ml_speed_calculator._foot_features / build_session_parquets._tkeo):
// psi[i] = x[i]² − x[i−1]·x[i+1], centered rolling mean over
// max(3, round(0.03 s · fs)) samples, clamped to ≥ 0 after smoothing.
// Computed per sensor (rows are interleaved across sensors) in time order.
export function addAccTkeoColumn(colMap, tCol) {
  if (colMap['acc_tkeo']) return // parquet already carries it
  const { AcX, AcY, AcZ } = colMap
  const times = colMap[tCol]
  const names = colMap['Name']
  if (!AcX || !AcY || !AcZ || !times) return

  const n = times.length
  const out = new Array(n).fill(null)
  const sensors = names
    ? [...new Set(names.filter(v => v != null && v !== ''))]
    : [null]

  sensors.forEach(sensor => {
    const idx = []
    for (let i = 0; i < n; i++) {
      if (sensor !== null && names[i] !== sensor) continue
      if (safeNum(times[i]) === null) continue
      if (safeNum(AcX[i]) === null || safeNum(AcY[i]) === null || safeNum(AcZ[i]) === null) continue
      idx.push(i)
    }
    if (idx.length < 3) return
    idx.sort((a, b) => safeNum(times[a]) - safeNum(times[b]))

    const t   = idx.map(i => safeNum(times[i]))
    const mag = idx.map(i => Math.hypot(safeNum(AcX[i]), safeNum(AcY[i]), safeNum(AcZ[i])))

    // Sample rate from the median dt; Time can be ms or s (~500 Hz either way).
    const dts = []
    for (let k = 1; k < t.length; k++) { const d = t[k] - t[k - 1]; if (d > 0) dts.push(d) }
    if (!dts.length) return
    dts.sort((a, b) => a - b)
    let dt = dts[Math.floor(dts.length / 2)]
    if (dt > 0.5) dt /= 1000 // ms → s
    const fs = 1 / dt

    const m = mag.length
    const psi = new Array(m).fill(0)
    for (let k = 1; k < m - 1; k++) psi[k] = mag[k] * mag[k] - mag[k - 1] * mag[k + 1]

    // pandas rolling(win, center=True, min_periods=1): [k−⌊win/2⌋, k+⌊(win−1)/2⌋]
    const win  = Math.max(3, Math.round(0.03 * fs))
    const back = Math.floor(win / 2)
    const fwd  = Math.floor((win - 1) / 2)
    const cum = new Array(m + 1)
    cum[0] = 0
    for (let k = 0; k < m; k++) cum[k + 1] = cum[k] + psi[k]
    for (let k = 0; k < m; k++) {
      const lo = Math.max(0, k - back)
      const hi = Math.min(m - 1, k + fwd)
      out[idx[k]] = Math.max((cum[hi + 1] - cum[lo]) / (hi - lo + 1), 0)
    }
  })

  colMap['acc_tkeo'] = out
}

// Extra TKEO traces for the column picker: AcX / AcY / AcZ and
// magg(Acc) = sqrt(AcX² + AcY² + AcZ²). Same operator as the backend
// `_tkeo` (psi = x² − x[i−1]·x[i+1], centered rolling mean, clamp ≥ 0),
// computed per sensor in time order.
export function addTkeoColumns(colMap, tCol) {
  const times = colMap[tCol]
  const names = colMap['Name']
  if (!times) return []
  const n = times.length
  const sources = [
    { col: 'TKEO_AcX', from: 'AcX' },
    { col: 'TKEO_AcY', from: 'AcY' },
    { col: 'TKEO_AcZ', from: 'AcZ' },
    { col: 'TKEO_AccMag', from: 'mag' },
  ]
  const needed = sources.filter(s => !colMap[s.col])
  if (!needed.length) return sources.map(s => s.col).filter(c => colMap[c])

  const buffers = {}
  needed.forEach(s => { buffers[s.col] = new Array(n).fill(null) })

  const sensors = names
    ? [...new Set(names.filter(v => v != null && v !== ''))]
    : [null]

  sensors.forEach(sensor => {
    const idx = []
    for (let i = 0; i < n; i++) {
      if (sensor !== null && names[i] !== sensor) continue
      if (safeNum(times[i]) === null) continue
      idx.push(i)
    }
    if (idx.length < 3) return
    idx.sort((a, b) => safeNum(times[a]) - safeNum(times[b]))

    needed.forEach(({ col, from }) => {
      const series = []
      const seriesIdx = []
      for (let k = 0; k < idx.length; k++) {
        const i = idx[k]
        let v = null
        if (from === 'mag') {
          const ax = safeNum(colMap.AcX?.[i])
          const ay = safeNum(colMap.AcY?.[i])
          const az = safeNum(colMap.AcZ?.[i])
          if (ax !== null && ay !== null && az !== null) v = Math.hypot(ax, ay, az)
        } else {
          v = safeNum(colMap[from]?.[i])
        }
        if (v === null) continue
        series.push(v)
        seriesIdx.push(i)
      }
      if (series.length < 3) return
      const tkeo = tkeoSeries(series, TKEO_WIN)
      for (let k = 0; k < seriesIdx.length; k++) buffers[col][seriesIdx[k]] = tkeo[k]
    })
  })

  const added = []
  needed.forEach(({ col }) => {
    if (buffers[col].some(v => v !== null)) {
      colMap[col] = buffers[col]
      added.push(col)
    }
  })
  return added
}

export function sumSensorColumns(colMap, sourceCols, outCol) {
  if (colMap[outCol]) return outCol
  if (sourceCols.some(col => !colMap[col])) return null
  const n = colMap[sourceCols[0]].length
  const out = new Array(n).fill(null)
  let wrote = false
  for (let i = 0; i < n; i++) {
    let total = 0
    let ok = true
    for (let s = 0; s < sourceCols.length; s++) {
      const v = safeNum(colMap[sourceCols[s]][i])
      if (v === null) { ok = false; break }
      total += v
    }
    if (!ok) continue
    out[i] = total
    wrote = true
  }
  if (!wrote) return null
  colMap[outCol] = out
  return outCol
}

export function addSensorSumColumns(colMap) {
  const added = []
  const raw = sumSensorColumns(colMap, SENSOR_COLS, SENSOR_SUM_RAW_COL)
  if (raw) added.push(raw)
  const norm = sumSensorColumns(
    colMap,
    SENSOR_COLS.map(col => `${col}_Normalized`),
    SENSOR_SUM_NORM_COL,
  )
  if (norm) added.push(norm)
  return added
}

export function addDerivedSessionColumns(colMap, tCol, additionalInfo) {
  addAccTkeoColumn(colMap, tCol)
  addTkeoColumns(colMap, tCol)
  addNormalizedSensorColumns(colMap, additionalInfo)
  addWeightedInsoleTotalColumn(colMap, tCol, additionalInfo)
  addSensorSumColumns(colMap)
}

// A stored additional_info blob can arrive JSON-encoded one or more levels deep
// (the backend double/triple-encodes it). Peel string layers until we reach a
// real value or give up.
export function deepUnwrapJson(value, maxDepth = 4) {
  let v = value
  for (let i = 0; i < maxDepth && typeof v === 'string'; i++) {
    try { v = JSON.parse(v) } catch { break }
  }
  return v
}

// A calibration bound must be exactly four finite numbers (booleans excluded —
// typeof true !== 'number').
export function isCalibQuad(a) {
  return Array.isArray(a) && a.length === 4
    && a.every(x => typeof x === 'number' && isFinite(x))
}

// Pull { left, right } insole calibration out of a session's additional_info,
// mirroring the backend shape
// additional_info.intake_data.insole_calibration.{left,right}.{min,max}.
// Returns null when it's absent or invalid, so callers fall back to the raw
// Sensor_* values untouched.
export function extractInsoleCalibration(additionalInfo) {
  const info = deepUnwrapJson(additionalInfo)
  if (!info || typeof info !== 'object') return null
  const intake = deepUnwrapJson(info.intake_data)
  if (!intake || typeof intake !== 'object') return null
  const calib = deepUnwrapJson(intake.insole_calibration)
  if (!calib || typeof calib !== 'object') return null

  const parseFoot = (footRaw) => {
    const foot = deepUnwrapJson(footRaw)
    if (!foot || typeof foot !== 'object') return null
    const min = deepUnwrapJson(foot.min)
    const max = deepUnwrapJson(foot.max)
    return isCalibQuad(min) && isCalibQuad(max) ? { min, max } : null
  }

  const left = parseFoot(calib.left)
  const right = parseFoot(calib.right)
  if (!left && !right) return null
  return { left, right }
}

// Per-timestep min-max normalization of one sensor reading. Values outside the
// calibration's [min, max] are left unclamped (can go <0 or >1).
// Degenerate calibration (max <= min) contributes nothing (0).
export function normalizeSensorValue(value, mn, mx) {
  const range = mx - mn
  if (range <= 0) return 0.0
  return (value - mn) / range
}

// Derived channels: Sensor_1..4_Normalized in [0, 1]. Each row is normalized
// with its own foot's calibration (ESP32_Sensor_1 → left, ESP32_Sensor_2 →
// right), mirroring the backend insole_normalization but WITHOUT the
// session-level aggregation to percentages — these stay raw per-timestep
// normalized values so they can be plotted over time. Returns the list of
// columns added (empty when the session carries no valid calibration, in which
// case consumers keep using the raw Sensor_* columns).
export function addNormalizedSensorColumns(colMap, additionalInfo) {
  const calib = extractInsoleCalibration(additionalInfo)
  if (!calib) return []
  const names = colMap['Name']
  const added = []

  SENSOR_COLS.forEach((col, si) => {
    const raw = colMap[col]
    if (!raw) return
    const normCol = `${col}_Normalized`
    if (colMap[normCol]) { added.push(normCol); return }

    const out = new Array(raw.length).fill(null)
    for (let i = 0; i < raw.length; i++) {
      const v = safeNum(raw[i])
      if (v === null) continue
      const foot = names ? inferSensorFoot(names[i]) : null
      const footCalib = foot ? calib[foot] : null
      if (!footCalib) continue // this foot lacks calibration → leave as no-data
      out[i] = normalizeSensorValue(v, footCalib.min[si], footCalib.max[si])
    }
    colMap[normCol] = out
    added.push(normCol)
  })

  return added
}

// ── Weighted insole total ──────────────────────────────────────────────────
// Derived channel `Sensor_Total_Weighted`: one summary trace per foot, because
// reading four pressure pads at once is not how you check whether an insole saw
// a step. Ported from the force-plate analysis notebook so a contact marked
// against this curve here is the same curve the model is trained on.
//
// Per foot, over that foot's rows in time order:
//   raw ADC → 25 Hz zero-phase Butterworth low-pass → per-pad normalization
//   → weighted sum.
//
// The low-pass runs on the raw ADC *before* normalizing. Normalization is affine
// per pad so the two commute, except for the clamp at 0 — and clamping an
// already-smooth signal is cleaner than smoothing a truncated one.
//
// The sum is WEIGHTED, not the flat S1+S2+S3+S4 that `Sensor_Total` means
// elsewhere, and the weights come from what each pad is worth against a force
// plate. Over 58 clean single-foot stances the mean within-stance correlation of
// one normalized pad against that plate's Fz is S4 heel +0.756, S3 arch +0.735,
// S2 forefoot +0.541, S1 big toe +0.115 — S3 and S4 carry load across the whole
// stance, S2 only sees push-off and S1 sees essentially nothing, which is why
// the flat sum (+0.681) scores worse than S3 alone. Against the flat sum this
// weighting lifts mean within-stance r to +0.763, loading-onset error 32 → 24 ms
// median and peak-time error 72 → 59 ms. The weights are FIXED, not fitted:
// fitted per session they disagree wildly and generalize worse held out. They
// sum to 1, so the curve sits in the same 0..1 band as the normalized pads.
//
// What it does not do is reproduce Fz's double hump — it tracks the loading
// envelope, when the foot took weight and let it go, not the M-shape inside it.
export const INSOLE_TOTAL_COL = 'Sensor_Total_Weighted'

export const INSOLE_TOTAL_WEIGHTS = { Sensor_1: 0.05, Sensor_2: 0.15, Sensor_3: 0.50, Sensor_4: 0.30 }

// The pads are read by a 10-bit ADC at 500 Hz and what looks like noise on the
// raw trace is mostly the quantization staircase, so they go through a 4th-order
// Butterworth low-pass applied forwards and backwards — zero-phase, no group
// delay at any frequency, which is what keeps the curve on the same clock as the
// video and the markup. 25 Hz because noise removal saturates well below it
// (15/25/30 Hz all land at ~0.46% jitter) while a higher cutoff keeps the peaks
// as sharp as the data supports; measured across walking and running sessions a
// 25 Hz cut moves gait-peak FWHM by −0.8% and apex height by +0.05%.
export const INSOLE_LP_HZ = 25.0

export const INSOLE_LP_ORDER = 4

export const INSOLE_CLOCK_MS = 2.0   // the 500 Hz insole clock these cutoffs were chosen for

// scipy.signal.butter(order, cutoffHz, 'lowpass', fs=fs, output='sos') for an
// even order: analog Butterworth prototype poles, scaled to the pre-warped
// cutoff, bilinear-transformed, then paired into biquads ordered by increasing
// pole radius with the whole gain in the first section — the layout scipy emits.
export function butterLowpassSos(order, cutoffHz, fs) {
  const wn = (2 * cutoffHz) / fs          // cutoff normalized so Nyquist = 1
  const warped = 4 * Math.tan((Math.PI * wn) / 2)
  const angleOf = k => (Math.PI * (-order + 1 + 2 * k)) / (2 * order)

  // Prototype pole -exp(i·θ) scaled by `warped`, mapped through (4 + s)/(4 − s).
  const poles = []
  for (let k = 0; k < order; k++) {
    const theta = angleOf(k)
    const re = -Math.cos(theta) * warped
    const im = -Math.sin(theta) * warped
    const dRe = 4 - re, dIm = -im
    const den = dRe * dRe + dIm * dIm
    poles.push([((4 + re) * dRe + im * dIm) / den, (im * dRe - (4 + re) * dIm) / den])
  }

  // Gain warped**order · Re(1 / Π(4 − s_k)); the transform leaves no analog zeros.
  let pRe = 1, pIm = 0
  for (let k = 0; k < order; k++) {
    const theta = angleOf(k)
    const dRe = 4 + Math.cos(theta) * warped
    const dIm = Math.sin(theta) * warped
    const nRe = pRe * dRe - pIm * dIm
    pIm = pRe * dIm + pIm * dRe
    pRe = nRe
  }
  const gain = Math.pow(warped, order) * (pRe / (pRe * pRe + pIm * pIm))

  const sos = poles
    .filter(([, im]) => im > 0)
    .sort((a, b) => (a[0] * a[0] + a[1] * a[1]) - (b[0] * b[0] + b[1] * b[1]))
    .map(([re, im]) => [1, 2, 1, 1, -2 * re, re * re + im * im])
  sos[0][0] *= gain
  sos[0][1] *= gain
  sos[0][2] *= gain
  return sos
}

// scipy.signal.sosfilt_zi: the steady-state delays each section holds for a unit
// step, scaled by the DC gain of the sections ahead of it.
export function sosfiltZi(sos) {
  const zi = []
  let scale = 1
  sos.forEach(([b0, b1, b2, , a1, a2]) => {
    const c0 = b1 - a1 * b0
    const c1 = b2 - a2 * b0
    const det = 1 + a1 + a2
    zi.push([scale * ((c0 + c1) / det), scale * (((1 + a1) * c1 - a2 * c0) / det)])
    scale *= (b0 + b1 + b2) / det
  })
  return zi
}

// scipy.signal.sosfilt with initial conditions: transposed direct form II, each
// sample carried through every section before the next sample is read.
export function sosfilt(sos, x, zi) {
  const out = new Float64Array(x.length)
  const z = zi.map(([z0, z1]) => [z0, z1])
  for (let i = 0; i < x.length; i++) {
    let v = x[i]
    for (let s = 0; s < sos.length; s++) {
      const [b0, b1, b2, , a1, a2] = sos[s]
      const y = b0 * v + z[s][0]
      z[s][0] = b1 * v - a1 * y + z[s][1]
      z[s][1] = b2 * v - a2 * y
      v = y
    }
    out[i] = v
  }
  return out
}

// scipy.signal.sosfiltfilt(sos, x) at its defaults (padtype='odd', padlen=None):
// odd-extend both ends by 3·(2·sections+1), filter forwards then backwards from
// step-matched initial conditions, then trim the extension back off.
export function sosfiltfilt(sos, x) {
  const n = x.length
  const edge = 3 * (2 * sos.length + 1)
  if (n <= edge) return Float64Array.from(x)

  const ext = new Float64Array(n + 2 * edge)
  for (let i = 0; i < edge; i++) ext[i] = 2 * x[0] - x[edge - i]
  ext.set(x, edge)
  for (let i = 0; i < edge; i++) ext[edge + n + i] = 2 * x[n - 1] - x[n - 2 - i]

  const zi = sosfiltZi(sos)
  const scaled = k => zi.map(([z0, z1]) => [z0 * k, z1 * k])

  let y = sosfilt(sos, ext, scaled(ext[0]))
  y.reverse()
  y = sosfilt(sos, y, scaled(y[0]))
  y.reverse()
  return y.slice(edge, y.length - edge)
}

// Zero-phase low-pass one foot's four pressure channels, already in time order.
// `fs` is read off the median sample interval rather than assumed, so a foot that
// did not record at 500 Hz is still filtered against its own clock. A block too
// short for the filter to settle, or one carrying a non-finite reading, comes back
// untouched rather than mangled — filtfilt would otherwise raise, or smear a
// single missing sample across the whole trace.
export function smoothInsole(channels, times, label) {
  const n = times.length
  const bail = (reason) => {
    console.warn(`${INSOLE_TOTAL_COL} (${label}): ${reason} — leaving ${n} samples unfiltered`)
    return channels
  }
  if (n < 2) return channels
  if (times.some(v => !isFinite(v))) return bail('the time column has non-finite values')
  if (channels.some(ch => ch.some(v => !isFinite(v)))) return bail('a pad reading is missing')

  const steps = []
  for (let i = 1; i < n; i++) steps.push(times[i] - times[i - 1])
  const dt = medianOf(steps)
  if (!(dt > 0)) return bail(`the median sample interval is ${dt}, not a usable clock`)
  // Session parquet carries Time in ms; some imported files use seconds. The same
  // test addAccTkeoColumn uses tells them apart at any plausible insole rate.
  const dtMs = dt > 0.5 ? dt : dt * 1000
  const fs = 1000 / dtMs

  // An unexpected clock is reported rather than quietly filtered against. The
  // tolerance is relative, unlike the notebook's exact comparison: seconds-based
  // timestamps carry float dust that an exact test reports as half the samples
  // being off clock, which buries the real dropped-sample case.
  const offClock = steps.filter(s => Math.abs(s - dt) > 1e-6 * dt).length / steps.length
  if (Math.abs(dtMs - INSOLE_CLOCK_MS) > 0.1 || offClock > 0.01) {
    console.warn(`${INSOLE_TOTAL_COL} (${label}): clock is ${dtMs.toFixed(3)} ms `
      + `(${fs.toFixed(0)} Hz) with ${Math.round(offClock * 100)}% of samples off it, `
      + `not the expected ${INSOLE_CLOCK_MS} ms — filtering against ${fs.toFixed(0)} Hz`)
  }
  // Too low an fs is the one thing here that really would distort a trace, because
  // the cutoff then bites far harder than the 25 Hz it claims to.
  if (!(INSOLE_LP_HZ > 0 && INSOLE_LP_HZ < 0.5 * fs)) {
    return bail(`${INSOLE_LP_HZ} Hz is not below this block's ${(0.5 * fs).toFixed(0)} Hz Nyquist`)
  }

  const sos = butterLowpassSos(INSOLE_LP_ORDER, INSOLE_LP_HZ, fs)
  if (n <= 3 * (2 * sos.length + 1)) return channels   // too short for filtfilt to settle
  return channels.map(ch => sosfiltfilt(sos, ch))
}

// The production normalizer (insole_normalization.normalize_matrix):
// (value − min) / (max − min) per pad against the session's intake calibration,
// clamped at 0 with no upper bound, so a reading above the calibrated max
// legitimately exceeds 1. Unlike the Sensor_*_Normalized display channels above,
// which deliberately leave negatives in so a drifting pad stays visible, this one
// clamps — the weighted total has to match what the calculators see.
export function normalizeInsoleValue(value, mn, mx) {
  const range = mx - mn
  if (range <= 0) return 0.0
  return Math.max((value - mn) / range, 0)
}

// Build INSOLE_TOTAL_COL into `colMap`. Returns the columns added, empty when the
// session carries no pressure pads at all. Without calibration the curve falls
// back to weighted raw ADC counts exactly as production does — it still shows
// where the foot loaded, just not on a 0..1 scale.
export function addWeightedInsoleTotalColumn(colMap, tCol, additionalInfo) {
  if (colMap[INSOLE_TOTAL_COL]) return [INSOLE_TOTAL_COL]   // parquet already carries it
  const times = colMap[tCol]
  if (!times || SENSOR_COLS.some(col => !colMap[col])) return []

  const calib = extractInsoleCalibration(additionalInfo)
  const names = colMap['Name']
  const weights = SENSOR_COLS.map(col => INSOLE_TOTAL_WEIGHTS[col])
  const n = times.length
  const out = new Array(n).fill(null)
  let wrote = false

  // Rows of the two insoles are interleaved in a session frame, so each foot is
  // filtered over its own rows in its own time order, as the notebook does.
  const sensors = names
    ? [...new Set(names.filter(v => v != null && v !== ''))]
    : [null]

  sensors.forEach(sensor => {
    const foot = inferSensorFoot(sensor)
    // The SpeedTracker and any unrecognised device carry no pads; they stay empty.
    if (names && !foot) return
    // A calibrated session missing this one foot leaves it as no-data rather than
    // mixing a normalized foot and a raw one into the same column.
    const footCalib = foot && calib ? calib[foot] : null
    if (calib && !footCalib) return

    const at = i => { const v = safeNum(times[i]); return v === null ? NaN : v }
    const idx = []
    for (let i = 0; i < n; i++) {
      if (sensor !== null && names[i] !== sensor) continue
      idx.push(i)
    }
    if (!idx.length) return
    idx.sort((a, b) => at(a) - at(b))

    const smoothed = smoothInsole(
      SENSOR_COLS.map(col => Float64Array.from(idx, i => {
        const v = safeNum(colMap[col][i])
        return v === null ? NaN : v
      })),
      idx.map(at),
      sensor || 'insole',
    )

    for (let k = 0; k < idx.length; k++) {
      let total = 0
      let ok = true
      for (let s = 0; s < SENSOR_COLS.length; s++) {
        const v = smoothed[s][k]
        if (!isFinite(v)) { ok = false; break }
        total += weights[s] * (footCalib
          ? normalizeInsoleValue(v, footCalib.min[s], footCalib.max[s])
          : v)
      }
      if (!ok) continue
      out[idx[k]] = total
      wrote = true
    }
  })

  if (!wrote) return []
  colMap[INSOLE_TOTAL_COL] = out
  return [INSOLE_TOTAL_COL]
}

export function computeGapStats(colMap, timeColumn) {
  const names = colMap?.Name || []
  const times = colMap?.[timeColumn] || []
  if (!names.length || !times.length) return {}

  const bySensor = new Map()
  for (let index = 0; index < Math.min(names.length, times.length); index++) {
    const name = names[index]
    const time = safeNum(times[index])
    if (!name || time === null) continue
    if (!bySensor.has(name)) bySensor.set(name, [])
    bySensor.get(name).push(time)
  }

  const result = {}
  bySensor.forEach((sensorTimes, name) => {
    sensorTimes.sort((a, b) => a - b)
    const diffs = []
    for (let index = 1; index < sensorTimes.length; index++) {
      const diff = sensorTimes[index] - sensorTimes[index - 1]
      if (diff > 0) diffs.push(diff)
    }
    const mean = diffs.length
      ? diffs.reduce((sum, value) => sum + value, 0) / diffs.length
      : null
    const threshold = mean === null ? null : mean * 2
    const gaps = []
    if (threshold !== null) {
      for (let index = 1; index < sensorTimes.length; index++) {
        if (sensorTimes[index] - sensorTimes[index - 1] > threshold) {
          gaps.push([sensorTimes[index - 1], sensorTimes[index]])
        }
      }
    }
    result[name] = {
      count: sensorTimes.length,
      time_diff_mean: mean,
      time_diff_max: diffs.length ? arrayMax(diffs) : null,
      gaps,
    }
  })
  return result
}

export const UNWRAPPABLE_ANGLE_COLUMNS = new Set(['XData', 'YData', 'ZData'])

// The left insole is mounted mirrored relative to the right one, so its X and Y
// accelerations point the opposite way and the two feet plot as reflections of
// each other. Negating these channels on the left puts both feet in one frame.
export const MIRRORED_LEFT_COLUMNS = new Set(['AcX', 'AcY'])

// Channels the IMU postprocessing rewrites — snapshotted so it can be undone.
export const IMU_SNAPSHOT_COLUMNS = ['AcX', 'AcY', 'AcZ', 'XData', 'YData', 'ZData', 'acc_tkeo', ...TKEO_PLOT_COLS]
