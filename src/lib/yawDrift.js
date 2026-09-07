// Gyro yaw drift estimate and correction (port of yaw_drift_calculator.py).
import { arrayMax, arrayMin, interpClamped, median, populationStd, unwrapAngleDegrees } from './utils.js'

// ── Gyro yaw drift ───────────────────────────────────────────────────────────
// A port of the backend's app/services/calculators/yaw_drift_calculator.py,
// kept numerically identical so the markup tool shows the same corrected
// heading the analysis pipeline computes. Runs on the Parquet already in memory.
//
// XData is a bounded 0..359° heading at ~500 Hz. The gyro behind it integrates a
// constant bias, so one foot's unwrapped heading drifts away from the other's —
// thousands of degrees over a four-minute recording. Drift and real turning are
// encoded identically in ONE foot's signal, so a single foot cannot be
// detrended: a per-foot least-squares fit on a real session read +8.18 °/s of
// "drift" on the left and 0 on the right, when the right foot's -5 °/s bias was
// merely cancelling 2380° of real turning. Both boards are strapped to one body
// and must accumulate the same net yaw, so real rotation cancels in
// (yawLeft - yawRight) and any slow trend left in that difference is drift by
// construction. Only that difference is identifiable; a bias both feet SHARE
// cannot be told apart from the athlete genuinely turning, and is left alone.
//
// The correction is split symmetrically — half to each foot, opposite signs — so
// the body yaw (left + right) / 2 is arithmetically unchanged whatever shape the
// curve takes. Nothing here can flatten a real rotation or invent a turn.
//
// Output contract: the corrected yaw is CONTINUOUS. It is no longer bounded to
// 0..359 and may exceed 360 or go negative.
export const YAW_LEFT_DEVICE = 'ESP32_Sensor_1'

export const YAW_RIGHT_DEVICE = 'ESP32_Sensor_2'

// Angle into fixed time blocks before the baseline fit: denoises the per-step
// wobble and makes the rolling statistics cheap (5 minutes becomes ~300 points).
export const BLOCK_WIN_S = 1.0

// Window of the rolling median+mean that separates drift from gait. Long enough
// that the feet's turn-by-turn differences average out instead of being mistaken
// for drift, short enough to follow a drift rate that CHANGES mid-session.
export const DRIFT_WINDOW_S = 31.0

// A recording shorter than one window cannot be detrended, only offset.
export const MIN_SPAN_S = DRIFT_WINDOW_S

// Absolute sanity ceiling on one foot's gyro bias, °/s.
export const MAX_DRIFT_DEG_S = 15.0

// A differential this small over a whole recording is noise, not drift.
export const MIN_DRIFT_DEG = 5.0

// Number(value) or NaN — the counterpart of pandas' to_numeric(errors='coerce').
// Blank strings are NaN, not 0: Number('') is 0 in JavaScript but float('')
// raises in Python, and a blank reading is missing data, not a heading of zero.
export function yawNumber(value) {
  if (value === null || value === undefined) return NaN
  if (typeof value === 'bigint') return Number(value)
  if (typeof value === 'string' && value.trim() === '') return NaN
  const n = Number(value)
  return isFinite(n) ? n : NaN
}

// (row positions, t_ms, unwrapped yaw) for one device, time-sorted and finite.
// The positions are offsets into the input, so the apply step can write values
// back to the exact rows they came from. Null under two usable samples.
export function footYaw(names, times, yaw, device) {
  const positions = []
  for (let i = 0; i < times.length; i++) {
    if (names[i] !== device) continue
    if (!isFinite(times[i]) || !isFinite(yaw[i])) continue
    positions.push(i)
  }
  if (positions.length < 2) return null

  // Array.prototype.sort is stable, which keeps samples sharing a timestamp in
  // recording order — the same tie-break as np.argsort(kind='stable').
  positions.sort((a, b) => times[a] - times[b])

  const count = positions.length
  const tMs = new Float64Array(count)
  const raw = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    tMs[i] = times[positions[i]]
    raw[i] = yaw[positions[i]]
  }
  return { positions, tMs, yaw: unwrapAngleDegrees(raw) }
}

// Average `values` into fixed BLOCK_WIN_S time blocks, dropping empty ones.
export function blockMeans(timeS, values, winS = BLOCK_WIN_S) {
  const n = timeS.length
  const t0 = timeS[0]
  const size = Math.floor((timeS[n - 1] - t0) / winS) + 1
  const count = new Float64Array(size)
  const sumT = new Float64Array(size)
  const sumV = new Float64Array(size)
  for (let i = 0; i < n; i++) {
    const idx = Math.floor((timeS[i] - t0) / winS)
    count[idx] += 1
    sumT[idx] += timeS[i]
    sumV[idx] += values[i]
  }

  let kept = 0
  for (let i = 0; i < size; i++) if (count[i] > 0) kept++
  const blockT = new Float64Array(kept)
  const blockValue = new Float64Array(kept)
  let w = 0
  for (let i = 0; i < size; i++) {
    if (count[i] <= 0) continue
    blockT[w] = sumT[i] / count[i]
    blockValue[w] = sumV[i] / count[i]
    w++
  }
  return { blockT, blockValue }
}

// Centred rolling statistic requiring a FULL window — no shrinking at the edges,
// matching pandas rolling(win, center=True, min_periods=win). NaN therefore
// propagates outward by half a window on each pass.
export function rollingCentered(values, win, reduce) {
  const n = values.length
  const out = new Float64Array(n).fill(NaN)
  const half = (win - 1) >> 1
  for (let i = half; i < n - half; i++) {
    const window = []
    let ok = true
    for (let k = i - half; k <= i + half; k++) {
      if (!isFinite(values[k])) { ok = false; break }
      window.push(values[k])
    }
    if (ok) out[i] = reduce(window)
  }
  return out
}

// Fill the half-window of NaN at each end by continuing the nearest drift rate.
// Letting the window shrink there biases it toward the middle of the recording:
// the curve goes flat exactly where the drift is still climbing, leaving the
// first and last seconds under-corrected.
//
// Null when there is less than a window of interior to shape a curve from — a
// recording barely longer than the window is nearly all extrapolation, and a
// curve fitted there tracks its own edge noise. The caller falls back to a rate.
export function extendEdges(t, smooth, win) {
  const valid = []
  for (let i = 0; i < smooth.length; i++) if (isFinite(smooth[i])) valid.push(i)
  if (valid.length < win) return null

  const out = Float64Array.from(smooth)
  const lo = valid[0]
  const hi = valid[valid.length - 1]
  const reach = Math.min(win, hi - lo)
  const slopeLo = (out[lo + reach] - out[lo]) / (t[lo + reach] - t[lo])
  const slopeHi = (out[hi] - out[hi - reach]) / (t[hi] - t[hi - reach])
  for (let i = 0; i < lo; i++) out[i] = out[lo] + slopeLo * (t[i] - t[lo])
  for (let i = hi + 1; i < out.length; i++) out[i] = out[hi] + slopeHi * (t[i] - t[hi])
  return out
}

// One drift rate for a recording too short to shape a curve, °/s. Theil-Sen —
// the median of the slopes between pairs at least a third of the recording apart
// — rather than least squares, which cannot tell a trend from a wiggle over a
// handful of turns: on a real 34 s session least squares read the gait as
// +0.83 °/s and over-corrected it, where the paired median gave +0.18 °/s.
export function singleRate(t, y) {
  const n = t.length
  if (n < 2) return 0
  const minDt = (t[n - 1] - t[0]) / 3
  const slopes = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dt = t[j] - t[i]
      if (dt >= minDt) slopes.push((y[j] - y[i]) / dt)
    }
  }
  return slopes.length === 0 ? 0 : median(slopes)
}

// (block times, drift curve) — the slow, drift-only part of the divergence. A
// rolling median over DRIFT_WINDOW_S keeps the gyros' slow disagreement and
// rejects the feet's turn-by-turn differences — median rather than mean because
// a turn is a one-sided excursion, not symmetric noise; the second pass takes
// the staircase off the median's output. The curve is zero-referenced (a
// constant offset between the feet is not drift) and its rate is clipped.
export function differentialBaseline(timeS, divergence) {
  const { blockT, blockValue: blockDiv } = blockMeans(timeS, divergence, BLOCK_WIN_S)
  const win = Math.max(3, Math.round(DRIFT_WINDOW_S / BLOCK_WIN_S)) | 1

  const meanOf = (w) => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i]; return s / w.length }
  const smoothed = rollingCentered(rollingCentered(blockDiv, win, median), win, meanOf)
  let smooth = extendEdges(blockT, smoothed, win)
  if (smooth === null) {
    const rate = singleRate(blockT, blockDiv)
    smooth = new Float64Array(blockT.length)
    for (let i = 0; i < blockT.length; i++) smooth[i] = rate * (blockT[i] - blockT[0])
  }

  const curve = new Float64Array(blockT.length)
  let acc = 0
  for (let i = 1; i < blockT.length; i++) {
    const ceiling = 2 * MAX_DRIFT_DEG_S * (blockT[i] - blockT[i - 1])
    let step = smooth[i] - smooth[i - 1]
    if (step > ceiling) step = ceiling
    else if (step < -ceiling) step = -ceiling
    acc += step
    curve[i] = acc
  }
  return { blockT, curve }
}

export function readYawColumns(colMap) {
  const { Name, Time, XData } = colMap || {}
  if (!Name || !Time || !XData) return null
  const n = Math.min(Name.length, Time.length, XData.length)
  const times = new Float64Array(n)
  const yaw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    times[i] = yawNumber(Time[i])
    yaw[i] = yawNumber(XData[i])
  }
  return { names: Name, times, yaw }
}

export function noYawDrift(reason, overrides = {}) {
  return {
    applied: false,
    reason,
    differentialDegS: 0,
    leftDegS: 0,
    rightDegS: 0,
    nonlinearityDeg: 0,
    divergenceStdBefore: 0,
    divergenceStdAfter: 0,
    spanS: 0,
    curveTMs: null,
    curveDeg: null,
    ...overrides,
  }
}

// Columns with Time in milliseconds, which is the clock the estimate assumes.
// The backend never needs this — its analysis loader is fed ms. The markup tool
// reads Parquet directly and some sessions carry Time in SECONDS; the companion
// API already rescales those (_rows_with_time_in_ms) using this same rule: a
// median positive interval under 0.5 means seconds. Without it a whole class of
// real sessions fails the overlap gate with a nonsense "overlap for 0.2 s".
// Scaling the clock moves no heading, so the corrected XData is identical either
// way — only spanS and curveTMs are expressed in the clock passed in.
export function withTimeInMs(colMap) {
  const time = colMap?.Time
  if (!time) return colMap

  const values = []
  for (let i = 0; i < time.length; i++) {
    const v = yawNumber(time[i])
    if (isFinite(v)) values.push(v)
  }
  if (values.length < 2) return colMap

  const ordered = [...new Set(values)].sort((a, b) => a - b)
  const deltas = []
  for (let i = 1; i < ordered.length; i++) {
    const d = ordered[i] - ordered[i - 1]
    if (d > 0) deltas.push(d)
  }
  if (!deltas.length || median(deltas) >= 0.5) return colMap

  const scaled = new Array(time.length)
  for (let i = 0; i < time.length; i++) scaled[i] = yawNumber(time[i]) * 1000
  return { ...colMap, Time: scaled }
}

// Per-foot gyro drift from the trend of yawLeft - yawRight. Both feet are
// required — the whole method is one foot checking the other. Time is read as
// milliseconds; run withTimeInMs first if the session might carry seconds.
export function estimateYawDrift(colMap) {
  const read = readYawColumns(colMap)
  if (read === null) return noYawDrift('no Name/Time/XData columns')

  const left = footYaw(read.names, read.times, read.yaw, YAW_LEFT_DEVICE)
  const right = footYaw(read.names, read.times, read.yaw, YAW_RIGHT_DEVICE)
  if (left === null || right === null) {
    const missing = [
      left === null ? YAW_LEFT_DEVICE : null,
      right === null ? YAW_RIGHT_DEVICE : null,
    ].filter(Boolean).join(' and ')
    return noYawDrift(`no usable yaw for ${missing}`)
  }

  // Only the window both feet actually recorded is evidence. Outside it there is
  // nothing to compare against: interpolation would hold the shorter foot's last
  // heading flat while the other keeps moving, and that manufactured divergence
  // is indistinguishable from drift. Real recordings hit this — a foot whose
  // Time rolls over at 2^32 µs claims a 4295 s span against a partner covering
  // 12 s, and fitting the overhang removed thousands of degrees never recorded.
  const first = Math.max(left.tMs[0], right.tMs[0])
  const last = Math.min(left.tMs[left.tMs.length - 1], right.tMs[right.tMs.length - 1])
  const spanS = (last - first) / 1000

  const sharedIdx = []
  for (let i = 0; i < left.tMs.length; i++) {
    if (left.tMs[i] >= first && left.tMs[i] <= last) sharedIdx.push(i)
  }
  if (spanS < MIN_SPAN_S || sharedIdx.length < 2) {
    return noYawDrift(
      `the feet overlap for ${Math.max(spanS, 0).toFixed(1)} s, under the ${MIN_SPAN_S} s drift window`,
      { spanS: Math.max(spanS, 0) },
    )
  }

  const tShared = new Float64Array(sharedIdx.length)
  const yawShared = new Float64Array(sharedIdx.length)
  const tSharedS = new Float64Array(sharedIdx.length)
  for (let i = 0; i < sharedIdx.length; i++) {
    tShared[i] = left.tMs[sharedIdx[i]]
    yawShared[i] = left.yaw[sharedIdx[i]]
    tSharedS[i] = tShared[i] / 1000
  }

  // Body rotation cancels in the difference, so what is left is drift alone.
  const rightAtLeft = interpClamped(tShared, right.tMs, right.yaw)
  const divergence = new Float64Array(tShared.length)
  for (let i = 0; i < tShared.length; i++) divergence[i] = yawShared[i] - rightAtLeft[i]

  const { blockT, curve } = differentialBaseline(tSharedS, divergence)

  const excursion = arrayMax(curve) - arrayMin(curve)
  if (excursion < MIN_DRIFT_DEG) {
    const std = populationStd(divergence)
    return noYawDrift(
      `the feet drift apart by ${excursion.toFixed(1)}°, under ${MIN_DRIFT_DEG}°`,
      { divergenceStdBefore: std, divergenceStdAfter: std, spanS },
    )
  }

  // Removing the curve from the divergence is the same as splitting it
  // symmetrically between the feet, which is what yawCorrection does.
  const curveAtLeft = interpClamped(tSharedS, blockT, curve)
  const corrected = new Float64Array(divergence.length)
  for (let i = 0; i < divergence.length; i++) corrected[i] = divergence[i] - curveAtLeft[i]

  const lastBlock = curve.length - 1
  const differential = (curve[lastBlock] - curve[0]) / spanS
  const straight = interpClamped(
    blockT,
    Float64Array.of(blockT[0], blockT[lastBlock]),
    Float64Array.of(curve[0], curve[lastBlock]),
  )
  let nonlinearity = 0
  for (let i = 0; i < curve.length; i++) {
    const d = Math.abs(curve[i] - straight[i])
    if (d > nonlinearity) nonlinearity = d
  }

  const curveTMs = new Float64Array(blockT.length)
  for (let i = 0; i < blockT.length; i++) curveTMs[i] = blockT[i] * 1000

  return {
    applied: true,
    reason: 'differential drift split between the feet',
    differentialDegS: differential,
    leftDegS: differential / 2,
    rightDegS: -differential / 2,
    nonlinearityDeg: nonlinearity,
    divergenceStdBefore: populationStd(divergence),
    divergenceStdAfter: populationStd(corrected),
    spanS,
    curveTMs,
    curveDeg: curve,
  }
}

// Degrees to SUBTRACT from one foot's yaw at each of tMs. Equal and opposite
// between the feet, so the body yaw (left + right) / 2 comes out of this
// untouched, whatever shape the curve has.
export function yawCorrection(drift, foot, tMs) {
  const out = new Float64Array(tMs.length)
  if (!drift.curveTMs || !drift.curveDeg) return out
  const half = foot === 'left' ? 0.5 : -0.5
  const curve = interpClamped(tMs, drift.curveTMs, drift.curveDeg)
  for (let i = 0; i < out.length; i++) out[i] = half * curve[i]
  return out
}

// A copy of the XData column with both feet replaced by drift-corrected yaw. The
// written values are UNWRAPPED (continuous, not 0..359). Other devices and a
// foot's own unusable rows are passed through exactly as they came in. Null when
// the correction does not apply, so callers keep the raw column.
export function correctedXData(colMap, drift) {
  if (!drift.applied) return null
  const read = readYawColumns(colMap)
  if (read === null) return null

  const out = [...colMap.XData]
  for (const [foot, device] of [['left', YAW_LEFT_DEVICE], ['right', YAW_RIGHT_DEVICE]]) {
    const got = footYaw(read.names, read.times, read.yaw, device)
    if (got === null) return null
    const correction = yawCorrection(drift, foot, got.tMs)
    for (let i = 0; i < got.positions.length; i++) {
      out[got.positions[i]] = got.yaw[i] - correction[i]
    }
  }
  return out
}

// How far the two feet may disagree before an angle is refused. Kept very wide
// on purpose: measured over 158 turns in five sessions, the turns a tighter
// limit would have dropped had a median error of 2.3 degrees against the body's
// own rotation — BETTER than the ones it kept (3.8). Feet disagreeing is normal
// asynchrony, not a fault, and averaging already handles it, so a strict limit
// only discards good measurements. What remains here is a guard against a board
// that has genuinely failed, where one foot reports rotation the other never saw.
export const TURN_DISAGREEMENT_BASE_DEG = 200.0

export const TURN_DISAGREEMENT_PER_S_DEG = 40.0

export const TURN_DISAGREEMENT_MAX_DEG = 720.0

/** Tolerance for the feet disagreeing over a span of the given length. */
export function turnDisagreementLimitDeg(spanMs) {
  const seconds = Math.max(0, spanMs) / 1000
  return Math.min(
    TURN_DISAGREEMENT_BASE_DEG + seconds * TURN_DISAGREEMENT_PER_S_DEG,
    TURN_DISAGREEMENT_MAX_DEG,
  )
}

/**
 * Net rotation over [fromMs, toMs], in degrees. Signed: positive and negative
 * are turns in opposite directions, which a classifier wants kept apart rather
 * than collapsed to a magnitude.
 *
 * Averages the two feet. They swing out of phase, so either one alone overshoots
 * or undershoots the body by tens of degrees; the average cancels most of that,
 * and it cancels a per-foot gyro bias too, since the drift correction splits
 * equally with opposite signs.
 *
 * Measured against the body's own rotation over 139 turns in three sessions,
 * grouped by how large the turn actually was — free-mode turns are any angle,
 * not just 180. Median absolute error:
 *
 *   true angle    average   furthest foot
 *     40-90         1-3         16-47
 *     90-150        2-6          9-12
 *     150+          3-5          5-10
 *
 * Taking the foot that rotated furthest was tried and is clearly worse: on small
 * turns it reads one foot's overswing as the turn itself, biasing 40-90 degree
 * turns upward by a median of +47 degrees.
 *
 * Null when the heading is missing, a foot has no samples inside the span, or
 * the feet disagree so badly that the number would be meaningless.
 */
export function turnAngleDeg(colMap, fromMs, toMs, correctedCol = null) {
  // The corrected heading is passed in when the session has one: reading the raw
  // XData instead would fold a few deg/s of gyro bias into every long span.
  const source = correctedCol ? { ...colMap, XData: correctedCol } : colMap
  const read = readYawColumns(source)
  if (read === null) return null

  const perFoot = []
  for (const device of [YAW_LEFT_DEVICE, YAW_RIGHT_DEVICE]) {
    const got = footYaw(read.names, read.times, read.yaw, device)
    if (got === null) return null
    // First and last sample inside the span; the heading is already unwrapped,
    // so the net rotation is simply the difference between the two ends.
    let first = -1
    let last = -1
    for (let i = 0; i < got.tMs.length; i++) {
      if (got.tMs[i] < fromMs) continue
      if (got.tMs[i] > toMs) break
      if (first < 0) first = i
      last = i
    }
    if (first < 0 || last <= first) return null
    perFoot.push(got.yaw[last] - got.yaw[first])
  }

  // Some disagreement is normal — the feet swing out of phase — but past the
  // span's limit one board is broken and any number would be invented.
  if (Math.abs(perFoot[0] - perFoot[1]) > turnDisagreementLimitDeg(toMs - fromMs)) return null
  return (perFoot[0] + perFoot[1]) / 2
}
