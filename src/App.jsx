import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import Plotly from 'plotly.js-dist-min'
import { parquetReadObjects } from 'hyparquet'
import './App.css'

// ── Constants ──────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_BASE ?? (
  import.meta.env.DEV ? 'http://localhost:8000' : 'https://dev-api.miraitech.health'
)
const CALCULATOR_API = import.meta.env.VITE_CALCULATOR_API ?? '/calculator-api'
const MARKUP_API = `${CALCULATOR_API}/markup`
const UI_FONT_FAMILY = 'Inter, "Segoe UI Variable", "Segoe UI", Arial, sans-serif'

// Math.max/min(...array) blows the call stack on long sessions (V8 caps spread
// argument count well below typical sample counts) — reduce instead.
function arrayMax(arr) {
  let m = -Infinity
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]
  return m
}
function arrayMin(arr) {
  let m = Infinity
  for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i]
  return m
}

function parseApiError(errData, status) {
  const detail = errData?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map(d => d.msg || String(d)).join('; ')
  if (detail && typeof detail === 'object') return detail.message || JSON.stringify(detail)
  return `Ошибка ${status}`
}

const PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#17becf',
]
const NON_DATA_COLS = new Set([
  'Name', 'Time', 'time', 'timestamp', 'Timestamp', 't',
  'target', 'Target', 'label', 'Label',
])
const PREFERRED_COLS = ['AcX', 'AcY', 'AcZ', 'XData', 'YData', 'ZData', 'GravityZ']
const SPEED_TRACKER = 'ESP32_SpeedTracker'
const ST_COLOR = '#2ca02c'
const ST_COL_NAMES = ['Distance', 'Speed', 'DistanceM', 'VelocityMs']
const ST_ONLY_COLS = new Set(ST_COL_NAMES)
// Distinct colours per SpeedTracker column so Distance and Speed don't look alike.
const ST_COL_COLORS = {
  Speed: '#2ca02c',       // green — keeps the SpeedTracker brand colour
  VelocityMs: '#2ca02c',
  Distance: '#9467bd',    // purple
  DistanceM: '#9467bd',
}
// Speed/Distance-predict overlays (charts/sprint): drawn on top of their
// respective subplots. Both read from the same fetched series.
const SPEED_PRED_COLS = new Set(['Speed', 'VelocityMs'])
const DISTANCE_PRED_COLS = new Set(['Distance', 'DistanceM'])
const PRED_COLOR = '#d62728'
const TRACE_HOVER_TEMPLATE = '<b>%{fullData.name}</b><br>Время: %{x}<br>Значение: %{y:.4g}<extra></extra>'
const EXTRA_CALCULATORS = [
  {
    id: 'step-detector-ttest',
    label: 'T-тест · Step detector',
    description: 'Детектирует шаги левой и правой ноги по давлению Sensor 1 + Sensor 2',
    color: '#7c3aed',
    fill: 'rgba(124,58,237,0.10)',
  },
  {
    id: 'tkeo-cadence',
    label: 'TKEO Cadence · без ML',
    description: 'Каденс и контакты по TKEO акселерометра',
    color: '#0891b2',
    fill: 'rgba(8,145,178,0.10)',
  },
  {
    id: 'step-cadence',
    label: 'Step Cadence · StepResUNet',
    description: 'ML-контакты, GCT и метрики по клику на колонку',
    color: '#d97706',
    fill: 'rgba(217,119,6,0.10)',
  },
  {
    id: 'jump-metrics',
    label: 'Jump BiLSTM',
    description: '24 признака IMU + давление · flight time, высота, contact time и RSI',
    color: '#db2777',
    fill: 'rgba(219,39,119,0.10)',
  },
  {
    id: 'force-jump',
    label: 'Bilateral GRF · BiLSTM+CNN',
    description: 'Пиковая вертикальная сила по двум стопам',
    color: '#dc2626',
    fill: 'rgba(220,38,38,0.10)',
  },
]
const FEATURED_EXTRA_CALCULATOR_IDS = new Set(['step-detector-ttest'])
const FEATURED_EXTRA_CALCULATORS = EXTRA_CALCULATORS.filter(
  calculator => FEATURED_EXTRA_CALCULATOR_IDS.has(calculator.id),
)
const COLLAPSIBLE_EXTRA_CALCULATORS = EXTRA_CALCULATORS.filter(
  calculator => !FEATURED_EXTRA_CALCULATOR_IDS.has(calculator.id),
)
const PROTOCOL_DETECTORS = [
  {
    id: 'protocol-walking-detector',
    label: 'Тест ходьбы · Step detector',
    description: 'Определяет интервалы контакта стоп при ходьбе',
    color: '#2563eb',
    fill: 'rgba(37,99,235,0.10)',
  },
  {
    id: 'protocol-running-detector',
    label: 'Анализ бега · Run detector',
    description: 'Определяет контакты левой и правой стопы в беге',
    color: '#16a34a',
    fill: 'rgba(22,163,74,0.10)',
  },
  {
    id: 'protocol-jumping-detector',
    label: 'Анализ прыжков · Jump detector',
    description: 'Определяет интервалы отрыва и приземления',
    color: '#db2777',
    fill: 'rgba(219,39,119,0.10)',
  },
  {
    id: 'protocol-shuttle-detector',
    label: 'Челночный бег · Turn detector',
    description: 'Определяет повороты и беговые отрезки',
    color: '#d97706',
    fill: 'rgba(217,119,6,0.10)',
  },
  {
    id: 'protocol-sprint-detector',
    label: 'Спринт 30 м · Sprint detector',
    description: 'Старт/финиш 30 м, шаги, step length и stride length',
    color: '#dc2626',
    fill: 'rgba(220,38,38,0.10)',
  },
  {
    id: 'protocol-beep-detector',
    label: 'Тест Beep · Yo-Yo detector',
    description: 'Определяет развороты на 180° и беговые фазы',
    color: '#0891b2',
    fill: 'rgba(8,145,178,0.10)',
  },
  {
    id: 'protocol-ttest-detector',
    label: 'T-тест · Phase detector',
    description: 'Определяет четыре поворота и беговые фазы T-теста',
    color: '#7c3aed',
    fill: 'rgba(124,58,237,0.10)',
  },
]
const EXTRA_CALCULATOR_BY_ID = Object.fromEntries(EXTRA_CALCULATORS.map(calc => [calc.id, calc]))
const PROTOCOL_DETECTOR_BY_ID = Object.fromEntries(PROTOCOL_DETECTORS.map(detector => [detector.id, detector]))
const PROTOCOL_SECTION_CALCULATOR_IDS = new Set([
  ...PROTOCOL_DETECTORS.map(detector => detector.id),
  ...FEATURED_EXTRA_CALCULATORS.map(calculator => calculator.id),
])
const CALCULATOR_BY_ID = { ...EXTRA_CALCULATOR_BY_ID, ...PROTOCOL_DETECTOR_BY_ID }
const PER_FOOT_TURN_DETECTOR_IDS = new Set([
  'protocol-shuttle-detector',
  'protocol-beep-detector',
  'protocol-ttest-detector',
])

const EVENT_STYLE_BY_KIND = {
  run: {
    label: 'Беговая фаза', color: '#16a34a', fill: 'rgba(22,163,74,0.10)', dash: 'dash', width: 1.5,
  },
  turn: {
    label: 'Поворот', color: '#f97316', fill: 'rgba(249,115,22,0.18)', dash: 'solid', width: 2.25,
  },
  sprint: {
    label: 'Отрезок 30 м', color: '#dc2626', fill: 'rgba(220,38,38,0.07)', dash: 'solid', width: 2.5,
  },
}
const TURN_EVENT_STYLE_BY_FOOT = {
  left: {
    run: { color: '#2563eb', fill: 'rgba(37,99,235,0.11)', dash: 'dash', width: 1.75 },
    turn: { color: '#db2777', fill: 'rgba(219,39,119,0.20)', dash: 'solid', width: 2.5 },
  },
  right: {
    run: { color: '#0f766e', fill: 'rgba(15,118,110,0.11)', dash: 'dash', width: 1.75 },
    turn: { color: '#f97316', fill: 'rgba(249,115,22,0.20)', dash: 'solid', width: 2.5 },
  },
}
const FOOT_EVENT_STYLE = {
  left: { color: '#2563eb', fill: 'rgba(37,99,235,0.13)', dash: 'dash', width: 1.5 },
  right: { color: '#f97316', fill: 'rgba(249,115,22,0.13)', dash: 'dot', width: 1.5 },
}
const FLIGHT_EVENT_STYLE = {
  left: { color: '#db2777', fill: 'rgba(219,39,119,0.14)', dash: 'dash', width: 1.75 },
  right: { color: '#7c3aed', fill: 'rgba(124,58,237,0.14)', dash: 'dot', width: 1.75 },
}

function calculatorEventStyle(calculator, contact) {
  const turnFootStyle = TURN_EVENT_STYLE_BY_FOOT[contact?.foot]?.[contact?.kind]
  if (turnFootStyle) return turnFootStyle
  const semanticStyle = EVENT_STYLE_BY_KIND[contact?.kind]
  if (semanticStyle) return semanticStyle
  if (contact?.kind === 'flight' && FLIGHT_EVENT_STYLE[contact?.foot]) {
    return FLIGHT_EVENT_STYLE[contact.foot]
  }
  if (FOOT_EVENT_STYLE[contact?.foot]) return FOOT_EVENT_STYLE[contact.foot]
  return {
    color: calculator?.color || '#64748b',
    fill: calculator?.fill || 'rgba(100,116,139,0.10)',
    dash: 'dash',
    width: 1.25,
  }
}

function calculatorEventLegend(calculator, result) {
  const items = new Map()
  ;(result?.contacts || []).forEach(contact => {
    const selectedTurnFoot = PER_FOOT_TURN_DETECTOR_IDS.has(calculator?.id)
      && ['left', 'right'].includes(result?.summary?.detection_foot)
      ? result.summary.detection_foot
      : ''
    const eventFoot = ['left', 'right'].includes(contact.foot) ? contact.foot : selectedTurnFoot
    const foot = eventFoot === 'left' ? 'L' : eventFoot === 'right' ? 'R' : ''
    const kind = contact.kind || 'event'
    const footSpecificKind = ['step', 'contact', 'plateau', 'flight', 'run', 'turn'].includes(kind)
    const key = footSpecificKind && foot ? `${kind}-${foot}` : kind
    if (items.has(key)) return
    const style = calculatorEventStyle(
      calculator,
      eventFoot && contact.foot !== eventFoot ? { ...contact, foot: eventFoot } : contact,
    )
    const footSuffix = foot ? ` ${foot}` : ''
    const label = kind === 'run'
      ? `Беговая фаза${footSuffix}`
      : kind === 'turn'
        ? `Поворот${footSuffix}`
        : kind === 'sprint'
          ? contact?.is_complete === false ? 'Неполный спринт' : 'Отрезок 30 м'
          : kind === 'flight'
            ? `Прыжок ${foot}`
            : kind === 'step'
              ? `Шаг ${foot}`
              : ['contact', 'plateau'].includes(kind)
                ? `Контакт ${foot}`
                : `Событие ${foot}`.trim()
    items.set(key, { key, label, ...style })
  })
  return [...items.values()]
}

function getCalculatorColumns(calculatorId, data) {
  if (!data) return {}
  const ALWAYS = ['Time', 'time', 'timestamp', 'Timestamp', 'Name']
  let needed = []
  if (PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)) {
    needed = [...ALWAYS, 'XData', 'Yaw', 'yaw', 'Heading', 'heading']
  } else if (['force-jump'].includes(calculatorId)) {
    needed = [
      ...ALWAYS,
      'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4',
      'Sensor_5', 'Sensor_6', 'Sensor_7', 'Sensor_8',
    ]
  } else if (calculatorId === 'protocol-sprint-detector') {
    needed = [
      ...ALWAYS,
      'AcX', 'AcY', 'AcZ', 'GravityZ',
      'XData', 'YData', 'ZData',
      'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4',
      'Distance', 'Speed', 'DistanceM', 'VelocityMs',
    ]
  } else {
    // Walking, Running, Jump, Step Cadence, TKEO Cadence need full IMU (accel + gyro) and pressure pads
    needed = [
      ...ALWAYS,
      'AcX', 'AcY', 'AcZ', 'GravityZ',
      'XData', 'YData', 'ZData',
      'Roll', 'Pitch', 'Heading',
      'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4',
      'Sensor_5', 'Sensor_6', 'Sensor_7', 'Sensor_8',
    ]
  }
  const filtered = {}
  for (const col of needed) {
    if (data[col] != null) {
      filtered[col] = data[col]
    }
  }
  return filtered
}

// Insole pressure channels and the device-name → foot mapping used for
// per-foot calibration/normalization (mirrors the backend: ESP32_Sensor_1 is
// the left insole, ESP32_Sensor_2 the right).
const SENSOR_COLS = ['Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']
const SENSOR_NAME_TO_FOOT = { ESP32_Sensor_1: 'left', ESP32_Sensor_2: 'right' }
const TURN_DETECTION_FOOT_OPTIONS = [
  { value: 'both', label: 'L+R', title: 'Наложить независимые детекции левой и правой ног' },
  { value: 'left', label: 'L', title: 'Детектировать только по левой ноге' },
  { value: 'right', label: 'R', title: 'Детектировать только по правой ноге' },
]

function inferSensorFoot(name) {
  if (!name) return null
  if (SENSOR_NAME_TO_FOOT[name]) return SENSOR_NAME_TO_FOOT[name]

  const normalized = String(name).trim().toLowerCase()
  if (/(^|[_\s-])left($|[_\s-])/.test(normalized)) return 'left'
  if (/(^|[_\s-])right($|[_\s-])/.test(normalized)) return 'right'
  if (/^(?:esp32_)?sensor[_\s-]*1(?:\D|$)/.test(normalized)) return 'left'
  if (/^(?:esp32_)?sensor[_\s-]*2(?:\D|$)/.test(normalized)) return 'right'
  return null
}

function groupSensorNamesByFoot(names) {
  const groups = { left: [], right: [] }
  const unknown = []

  names.forEach(name => {
    const foot = inferSensorFoot(name)
    if (foot) groups[foot].push(name)
    else unknown.push(name)
  })

  // Legacy parquet files sometimes carry anonymous device names. Keep their
  // original ordering as a deterministic left/right fallback.
  unknown.forEach((name, index) => {
    const foot = groups.left.length === 0
      ? 'left'
      : groups.right.length === 0
        ? 'right'
        : index % 2 === 0 ? 'left' : 'right'
    groups[foot].push(name)
  })

  return groups
}

function sensorFootForName(name, names) {
  const inferred = inferSensorFoot(name)
  if (inferred) return inferred
  const groups = groupSensorNamesByFoot(names)
  if (groups.left.includes(name)) return 'left'
  if (groups.right.includes(name)) return 'right'
  return null
}

function sensorNameForFoot(names, foot) {
  return groupSensorNamesByFoot(names)[foot]?.[0] || ''
}

function rowsToColMap(rows) {
  const colMap = {}
  rows.forEach((row, index) => {
    Object.keys(row).forEach(k => {
      if (!colMap[k]) colMap[k] = Array(index).fill(null)
    })
    Object.keys(colMap).forEach(k => {
      const value = row[k]
      colMap[k].push(typeof value === 'bigint' ? Number(value) : (value ?? null))
    })
  })
  return colMap
}

function colMapToRows(colMap) {
  const columns = Object.keys(colMap || {})
  if (!columns.length) return []
  const length = Math.max(...columns.map(column => colMap[column]?.length || 0))
  return Array.from({ length }, (_, index) => {
    const row = {}
    columns.forEach(column => { row[column] = colMap[column]?.[index] ?? null })
    return row
  })
}

function detectTimeCol(allCols) {
  return allCols.find(c => c === 'Time')
    || allCols.find(c => ['time', 'timestamp', 'Timestamp', 't'].includes(c))
    || allCols[0]
}

function computeNumericColumns(colMap, tCol) {
  return Object.keys(colMap).filter(c => {
    if (NON_DATA_COLS.has(c) || c === tCol) return false
    return (colMap[c] || []).some(v => safeNum(v) !== null)
  })
}

function sortSensorNames(colMap) {
  if (!colMap['Name']) return []
  return [...new Set(colMap['Name'].filter(v => v != null && v !== ''))]
    .sort((a, b) => a.localeCompare(b))
}

function computeAutoOffsetST(colMap, timeCol, insoleNames) {
  if (!insoleNames.length || !colMap[timeCol] || !colMap['Name']) return 0
  const times = colMap[timeCol]
  const names = colMap['Name']
  let insoleMin = Infinity
  let stMin = Infinity
  for (let i = 0; i < times.length; i++) {
    const t = safeNum(times[i])
    if (t === null) continue
    const n = names[i]
    if (n === SPEED_TRACKER) { if (t < stMin) stMin = t }
    else if (insoleNames.includes(n)) { if (t < insoleMin) insoleMin = t }
  }
  if (!isFinite(insoleMin) || !isFinite(stMin)) return 0
  return insoleMin - stMin
}

function resolveStDataCol(data, col) {
  if ((data[col] || []).some(v => safeNum(v) !== null)) return col
  const alt = { Distance: 'DistanceM', Speed: 'VelocityMs', DistanceM: 'Distance', VelocityMs: 'Speed' }[col]
  if (alt && (data[alt] || []).some(v => safeNum(v) !== null)) return alt
  return col
}

function buildDefaultCols(numCols, hasSpeedTracker, colMap, sensorNames) {
  const sensorSet = new Set(sensorNames)
  const nameArr = colMap?.Name
  const hasVisibleData = (col) => {
    const values = colMap?.[col] || []
    return values.some((value, index) => (
      safeNum(value) !== null
      && (!nameArr || sensorSet.size === 0 || sensorSet.has(nameArr[index]))
    ))
  }
  const plottable = numCols.filter(hasVisibleData)
  const imu = PREFERRED_COLS.filter(c => plottable.includes(c)).slice(0, 3)
  const st  = hasSpeedTracker ? ST_COL_NAMES.filter(c => numCols.includes(c)) : []
  const merged = [...imu]
  st.forEach(c => { if (!merged.includes(c)) merged.push(c) })
  return merged.length ? merged : plottable.slice(0, 3)
}

const TKEO_WIN = 15
const TKEO_PLOT_COLS = ['TKEO_AcX', 'TKEO_AcY', 'TKEO_AcZ', 'TKEO_AccMag']
const SENSOR_SUM_RAW_COL = 'Sensor_Sum_Raw'
const SENSOR_SUM_NORM_COL = 'Sensor_Sum_Normalized'

// pandas Series.rolling(win, center=True, min_periods=1).mean() over TKEO psi.
function tkeoSeries(x, win = TKEO_WIN) {
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
function addAccTkeoColumn(colMap, tCol) {
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
function addTkeoColumns(colMap, tCol) {
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

function sumSensorColumns(colMap, sourceCols, outCol) {
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

function addSensorSumColumns(colMap) {
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

function addDerivedSessionColumns(colMap, tCol, additionalInfo) {
  addAccTkeoColumn(colMap, tCol)
  addTkeoColumns(colMap, tCol)
  addNormalizedSensorColumns(colMap, additionalInfo)
  addWeightedInsoleTotalColumn(colMap, tCol, additionalInfo)
  addSensorSumColumns(colMap)
}

// A stored additional_info blob can arrive JSON-encoded one or more levels deep
// (the backend double/triple-encodes it). Peel string layers until we reach a
// real value or give up.
function deepUnwrapJson(value, maxDepth = 4) {
  let v = value
  for (let i = 0; i < maxDepth && typeof v === 'string'; i++) {
    try { v = JSON.parse(v) } catch { break }
  }
  return v
}

// A calibration bound must be exactly four finite numbers (booleans excluded —
// typeof true !== 'number').
function isCalibQuad(a) {
  return Array.isArray(a) && a.length === 4
    && a.every(x => typeof x === 'number' && isFinite(x))
}

// Pull { left, right } insole calibration out of a session's additional_info,
// mirroring the backend shape
// additional_info.intake_data.insole_calibration.{left,right}.{min,max}.
// Returns null when it's absent or invalid, so callers fall back to the raw
// Sensor_* values untouched.
function extractInsoleCalibration(additionalInfo) {
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
function normalizeSensorValue(value, mn, mx) {
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
function addNormalizedSensorColumns(colMap, additionalInfo) {
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
const INSOLE_TOTAL_COL = 'Sensor_Total_Weighted'
const INSOLE_TOTAL_WEIGHTS = { Sensor_1: 0.05, Sensor_2: 0.15, Sensor_3: 0.50, Sensor_4: 0.30 }

// The pads are read by a 10-bit ADC at 500 Hz and what looks like noise on the
// raw trace is mostly the quantization staircase, so they go through a 4th-order
// Butterworth low-pass applied forwards and backwards — zero-phase, no group
// delay at any frequency, which is what keeps the curve on the same clock as the
// video and the markup. 25 Hz because noise removal saturates well below it
// (15/25/30 Hz all land at ~0.46% jitter) while a higher cutoff keeps the peaks
// as sharp as the data supports; measured across walking and running sessions a
// 25 Hz cut moves gait-peak FWHM by −0.8% and apex height by +0.05%.
const INSOLE_LP_HZ = 25.0
const INSOLE_LP_ORDER = 4
const INSOLE_CLOCK_MS = 2.0   // the 500 Hz insole clock these cutoffs were chosen for

function medianOf(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// scipy.signal.butter(order, cutoffHz, 'lowpass', fs=fs, output='sos') for an
// even order: analog Butterworth prototype poles, scaled to the pre-warped
// cutoff, bilinear-transformed, then paired into biquads ordered by increasing
// pole radius with the whole gain in the first section — the layout scipy emits.
function butterLowpassSos(order, cutoffHz, fs) {
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
function sosfiltZi(sos) {
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
function sosfilt(sos, x, zi) {
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
function sosfiltfilt(sos, x) {
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
function smoothInsole(channels, times, label) {
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
function normalizeInsoleValue(value, mn, mx) {
  const range = mx - mn
  if (range <= 0) return 0.0
  return Math.max((value - mn) / range, 0)
}

// Build INSOLE_TOTAL_COL into `colMap`. Returns the columns added, empty when the
// session carries no pressure pads at all. Without calibration the curve falls
// back to weighted raw ADC counts exactly as production does — it still shows
// where the foot loaded, just not on a 0..1 scale.
function addWeightedInsoleTotalColumn(colMap, tCol, additionalInfo) {
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

const L_FILL = 'rgba(31,119,180,0.35)'
const R_FILL = 'rgba(255,127,14,0.35)'
const L_LINE = 'rgba(31,119,180,0.9)'
const R_LINE = 'rgba(255,127,14,0.9)'
const GAP_FILL = 'rgba(220,53,69,0.35)'
const GAP_LINE = 'rgba(220,53,69,0.92)'
const SEL_FILL = 'rgba(234,179,8,0.5)'
const SEL_LINE = '#ca8a04'

function buildGapBandShapes(intervals, nSubplots) {
  if (!intervals.length || nSubplots < 1) return []
  const shapes = []
  for (const [x0, x1] of intervals) {
    for (let i = 0; i < nSubplots; i++) {
      shapes.push({
        type: 'rect',
        x0, x1,
        xref: i === 0 ? 'x' : `x${i + 1}`,
        y0: 0, y1: 1,
        yref: i === 0 ? 'y domain' : `y${i + 1} domain`,
        fillcolor: GAP_FILL,
        line: { color: GAP_LINE, width: 1.5 },
        layer: 'below',
      })
    }
  }
  return shapes
}

function safeNum(v) {
  if (v === null || v === undefined) return null
  const n = typeof v === 'bigint' ? Number(v) : Number(v)
  return isFinite(n) ? n : null
}

function computeGapStats(colMap, timeColumn) {
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

function unwrapAngleDegrees(arr, threshold = 180.0) {
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

const UNWRAPPABLE_ANGLE_COLUMNS = new Set(['XData', 'YData', 'ZData'])
// The left insole is mounted mirrored relative to the right one, so its X and Y
// accelerations point the opposite way and the two feet plot as reflections of
// each other. Negating these channels on the left puts both feet in one frame.
const MIRRORED_LEFT_COLUMNS = new Set(['AcX', 'AcY'])
// Channels the IMU postprocessing rewrites — snapshotted so it can be undone.
const IMU_SNAPSHOT_COLUMNS = ['AcX', 'AcY', 'AcZ', 'XData', 'YData', 'ZData', 'acc_tkeo', ...TKEO_PLOT_COLS]

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
const YAW_LEFT_DEVICE = 'ESP32_Sensor_1'
const YAW_RIGHT_DEVICE = 'ESP32_Sensor_2'
// Angle into fixed time blocks before the baseline fit: denoises the per-step
// wobble and makes the rolling statistics cheap (5 minutes becomes ~300 points).
const BLOCK_WIN_S = 1.0
// Window of the rolling median+mean that separates drift from gait. Long enough
// that the feet's turn-by-turn differences average out instead of being mistaken
// for drift, short enough to follow a drift rate that CHANGES mid-session.
const DRIFT_WINDOW_S = 31.0
// A recording shorter than one window cannot be detrended, only offset.
const MIN_SPAN_S = DRIFT_WINDOW_S
// Absolute sanity ceiling on one foot's gyro bias, °/s.
const MAX_DRIFT_DEG_S = 15.0
// A differential this small over a whole recording is noise, not drift.
const MIN_DRIFT_DEG = 5.0

// Number(value) or NaN — the counterpart of pandas' to_numeric(errors='coerce').
// Blank strings are NaN, not 0: Number('') is 0 in JavaScript but float('')
// raises in Python, and a blank reading is missing data, not a heading of zero.
function yawNumber(value) {
  if (value === null || value === undefined) return NaN
  if (typeof value === 'bigint') return Number(value)
  if (typeof value === 'string' && value.trim() === '') return NaN
  const n = Number(value)
  return isFinite(n) ? n : NaN
}

// np.median — the mean of the two middle values on even lengths.
function median(values) {
  const n = values.length
  if (!n) return NaN
  const sorted = [...values].sort((a, b) => a - b)
  const mid = n >> 1
  return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// Population standard deviation — divides by N, matching np.std.
function populationStd(values) {
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
function interpClamped(x, xp, fp) {
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

// (row positions, t_ms, unwrapped yaw) for one device, time-sorted and finite.
// The positions are offsets into the input, so the apply step can write values
// back to the exact rows they came from. Null under two usable samples.
function footYaw(names, times, yaw, device) {
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
function blockMeans(timeS, values, winS = BLOCK_WIN_S) {
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
function rollingCentered(values, win, reduce) {
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
function extendEdges(t, smooth, win) {
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
function singleRate(t, y) {
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
function differentialBaseline(timeS, divergence) {
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

function readYawColumns(colMap) {
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

function noYawDrift(reason, overrides = {}) {
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
function withTimeInMs(colMap) {
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
function estimateYawDrift(colMap) {
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
function yawCorrection(drift, foot, tMs) {
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
function correctedXData(colMap, drift) {
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

function formatTime(s) {
  if (!isFinite(s) || s < 0) return '0:00.0'
  const m = Math.floor(s / 60)
  const sec = (s % 60).toFixed(1).padStart(4, '0')
  return `${m}:${sec}`
}
function formatDuration(d, unit) {
  if (!isFinite(d) || d < 0) return '—'
  if (unit === 'ms') {
    if (d >= 1000) return (d / 1000).toFixed(2) + 'с'
    return d.toFixed(0) + 'мс'
  }
  return d.toFixed(3) + 'с'
}

function formatMetric(value, digits = 2, suffix = '') {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  return `${Number(value).toFixed(digits)}${suffix}`
}

function formatInterval(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  if (Number.isInteger(number)) return String(number)
  return number.toFixed(number < 10 ? 2 : 1).replace(/0+$/, '').replace(/\.$/, '')
}

function protocolDetectorSummary(result) {
  const summary = result?.summary
  if (!summary) return 'события ещё не рассчитаны'
  if (summary.turn_count != null) {
    const detectionFoot = PER_FOOT_TURN_DETECTOR_IDS.has(result?.calculator)
      ? { both: 'L+R', left: 'L', right: 'R' }[summary.detection_foot || 'both']
      : ''
    if (detectionFoot === 'L+R' && summary.left_turn_count != null && summary.right_turn_count != null) {
      return `L+R · повороты L ${summary.left_turn_count} / R ${summary.right_turn_count} · фазы L ${summary.left_run_count || 0} / R ${summary.right_run_count || 0}`
    }
    return `${detectionFoot ? `${detectionFoot} · ` : ''}повороты ${summary.turn_count} · беговые фазы ${summary.run_count || 0}`
  }
  if (summary.sprint_count != null) {
    if (summary.sprint_count === 0 && summary.segment_found) {
      return `неполный спринт ${formatMetric(summary.distance_m, 1, ' м')} · шаги ${summary.step_count || 0} (L ${summary.left_count || 0} / R ${summary.right_count || 0}) · step ${formatMetric(summary.step_length_m, 2, ' м')} · stride ${formatMetric(summary.stride_length_m, 2, ' м')}`
    }
    if (summary.sprint_count === 0) return 'старт спринта не найден'
    return `30 м найдено · шаги ${summary.step_count || 0} (L ${summary.left_count || 0} / R ${summary.right_count || 0}) · step ${formatMetric(summary.step_length_m, 2, ' м')} · stride ${formatMetric(summary.stride_length_m, 2, ' м')}`
  }
  if (summary.flight_count != null) {
    return `прыжки ${summary.flight_count} · L ${summary.left_count || 0} · R ${summary.right_count || 0}`
  }
  return `контакты ${summary.contact_count || 0} · L ${summary.left_count || 0} · R ${summary.right_count || 0}`
}

function normaliseSpeedPrediction(data) {
  const modelPoints = Array.isArray(data?.speed_series)
    ? data.speed_series
      .map(point => ({
        // Charts API returns CausalSpeedTCN timestamps in milliseconds.
        time: Number(point.time) / 1000,
        speed: Number(point.speed),
        distance: Number(point.distance),
      }))
      .filter(point => Number.isFinite(point.time) && Number.isFinite(point.speed) && Number.isFinite(point.distance))
    : []

  const trackerPoints = Array.isArray(data?.speed?.data_points)
    ? data.speed.data_points
      .map(point => ({
        time: Number(point.time),
        speed: Number(point.speed),
        distance: Number(point.distance),
      }))
      .filter(point => Number.isFinite(point.time) && Number.isFinite(point.speed) && Number.isFinite(point.distance))
    : []

  const dataPoints = modelPoints.length > 0 ? modelPoints : trackerPoints
  if (!modelPoints.length) {
    return {
      ...data,
      data_points: dataPoints,
      stat: data?.speed?.stat || null,
      model: 'SpeedTracker',
    }
  }

  const peak = dataPoints.reduce((best, point) => point.speed > best.speed ? point : best, dataPoints[0])
  const start = dataPoints.find(point => point.speed > 0 && point.distance > 0) || dataPoints[0]
  const finish = dataPoints.find(point => point.distance >= 30)
  const duration = finish && finish.time > start.time ? finish.time - start.time : null

  return {
    ...data,
    data_points: dataPoints,
    stat: {
      timestep_at_peak_speed: peak.time,
      distance_at_peak_speed: peak.distance,
      peak_speed: peak.speed,
      start_time: start.time,
      end_time: finish?.time ?? null,
      average_speed: duration ? 30 / duration : null,
      duration,
    },
    model: 'CausalSpeedTCN ensemble',
  }
}

function buildCursorShapes(x, n) {
  return Array.from({ length: n }, (_, i) => ({
    type: 'line',
    x0: x, x1: x,
    y0: 0, y1: 1,
    xref: i === 0 ? 'x' : `x${i + 1}`,
    yref: i === 0 ? 'y domain' : `y${i + 1} domain`,
    line: { color: 'rgba(220,40,40,0.85)', width: 2, dash: 'dot' },
  }))
}

function buildSelectedPointShapes(x, n) {
  return Array.from({ length: n }, (_, i) => ({
    type: 'line',
    x0: x, x1: x,
    y0: 0, y1: 1,
    xref: i === 0 ? 'x' : `x${i + 1}`,
    yref: i === 0 ? 'y domain' : `y${i + 1} domain`,
    line: { color: SEL_LINE, width: 3.5 },
    layer: 'above',
  }))
}

function chartSubplotCenterTop(index, total) {
  if (total <= 0) return '50%'
  const gap = 0.03
  const subplotHeight = (1 - gap * (total - 1)) / total
  const centerDomain = 1 - index * (subplotHeight + gap) - subplotHeight / 2
  return `calc(12px + (100% - 54px) * ${1 - centerDomain})`
}

function chartSubplotMetrics(index, total, chartHeight) {
  const gap = 0.03
  const plotTop = 12
  const plotHeight = Math.max(1, chartHeight - 54)
  const subplotHeight = total > 0 ? (1 - gap * (total - 1)) / total : 1
  const topDomain = 1 - index * (subplotHeight + gap)
  return {
    top: plotTop + (1 - topDomain) * plotHeight,
    height: subplotHeight * plotHeight,
  }
}

function plotAxisKey(index, axis) {
  return index === 0 ? axis : `${axis}${index + 1}`
}

function readPlotRange(eventData, axisKey) {
  if (!eventData) return null
  const start = eventData[`${axisKey}.range[0]`]
  const end = eventData[`${axisKey}.range[1]`]
  if (start !== undefined && end !== undefined) return [Number(start), Number(end)]
  const nested = eventData[`${axisKey}.range`] ?? eventData[axisKey]?.range
  if (Array.isArray(nested) && nested.length >= 2) return [Number(nested[0]), Number(nested[1])]
  return null
}

function currentAxisRange(gd, axisKey) {
  const layoutRange = gd?.layout?.[axisKey]?.range
  if (Array.isArray(layoutRange) && layoutRange.length >= 2) {
    return [Number(layoutRange[0]), Number(layoutRange[1])]
  }
  const fullRange = gd?._fullLayout?.[axisKey]?.range
  if (Array.isArray(fullRange) && fullRange.length >= 2) {
    return [Number(fullRange[0]), Number(fullRange[1])]
  }
  return null
}

function hasSessionMetaValue(value) {
  return value != null && value !== ''
}

// The markup API fills an absent athlete with an em dash; the header badge is
// hidden instead of showing a placeholder.
function normalizeMemberName(value) {
  const name = typeof value === 'string' ? value.trim() : ''
  return name && name !== '—' ? name : ''
}

function normalizeSessionTitle(value) {
  return typeof value === 'string' ? value.trim() : ''
}

function getPairStartIndex(index) {
  return index - (index % 2)
}

function parseCsvRow(line) {
  const out = []
  let cur = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++ }
        else inQuotes = false
      } else cur += ch
    } else if (ch === '"') {
      inQuotes = true
    } else if (ch === ',') {
      out.push(cur)
      cur = ''
    } else {
      cur += ch
    }
  }
  out.push(cur)
  return out
}

function parseCsvText(text) {
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/)
  if (!lines.length) throw new Error('CSV пустой')
  const headers = parseCsvRow(lines[0]).map(h => h.trim())
  const rows = []
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue
    const vals = parseCsvRow(lines[i])
    const row = {}
    headers.forEach((h, j) => { row[h] = vals[j] ?? '' })
    rows.push(row)
  }
  return { headers, rows }
}

// CSV cells arrive as strings; parquet columns are already typed. Convert the
// columns that hold only numbers so the chart, calculators and gap stats see
// the same shape from both sources ('' → null, text columns kept as is).
function coerceCsvColumnsToNumbers(colMap) {
  Object.keys(colMap).forEach(col => {
    const arr = colMap[col] || []
    let hasNumber = false
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i]
      if (v === null || v === undefined || v === '') continue
      if (safeNum(v) === null) return // non-numeric column (Name, …)
      hasNumber = true
    }
    if (!hasNumber) return
    colMap[col] = arr.map(v => (v === null || v === undefined || v === '' ? null : safeNum(v)))
  })
}

function isTargetOne(v) {
  if (v === 1 || v === '1' || v === 1.0) return true
  const s = String(v ?? '').trim()
  return s === '1' || s === '1.0'
}

function extractContactPairsFromTargetRuns(times, targets, offset = 0) {
  const contacts = []
  let i = 0
  while (i < targets.length) {
    while (i < targets.length && !isTargetOne(targets[i])) i++
    if (i >= targets.length) break
    const startT = safeNum(times[i])
    if (startT === null) { i++; continue }
    let endT = startT
    while (i < targets.length && isTargetOne(targets[i])) {
      const t = safeNum(times[i])
      if (t !== null) endT = t
      i++
    }
    contacts.push(startT + offset, endT + offset)
  }
  return contacts
}

function extractContactsFromLabeledCsv(rows, timeCol, leftSensors, rightSensors, offsetS1, offsetS2) {
  const bySensors = (sensorNames) => {
    const candidates = sensorNames.map(sensorName => rows
      .filter(r => (r.Name || r.name) === sensorName)
      .map(r => ({
        t: safeNum(r[timeCol]),
        target: r.Target ?? r.target ?? r.Label ?? r.label ?? '',
      }))
      .filter(r => r.t !== null)
      .sort((a, b) => a.t - b.t))
      .filter(candidate => candidate.length > 0)

    // A foot can have separate pressure and IMU devices. Prefer the densest
    // labeled timeline instead of interleaving timestamps from both devices.
    return candidates.sort((a, b) => {
      const aTargets = a.reduce((n, row) => n + Number(isTargetOne(row.target)), 0)
      const bTargets = b.reduce((n, row) => n + Number(isTargetOne(row.target)), 0)
      return bTargets - aTargets || b.length - a.length
    })[0] || []
  }

  const leftRows = bySensors(leftSensors)
  const rightRows = bySensors(rightSensors)
  if (!leftRows.length && !rightRows.length) {
    throw new Error('Строки для сенсоров левой и правой ноги не найдены')
  }

  const leftContacts = extractContactPairsFromTargetRuns(
    leftRows.map(r => r.t),
    leftRows.map(r => r.target),
    offsetS1,
  )
  const rightContacts = extractContactPairsFromTargetRuns(
    rightRows.map(r => r.t),
    rightRows.map(r => r.target),
    offsetS2,
  )

  return { leftContacts, rightContacts, leftCount: leftContacts.length / 2, rightCount: rightContacts.length / 2 }
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  // Auth
  const [token, setToken]               = useState(() => sessionStorage.getItem('auth_token') || '')
  const [loginEmail, setLoginEmail]     = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [loginError, setLoginError]     = useState('')
  const [authLoading, setAuthLoading]   = useState(false)

  // Session
  const [sessionId, setSessionId]       = useState('')
  const [loadedSessionId, setLoadedSessionId] = useState(null)
  const [sessionLabel, setSessionLabel] = useState('')
  const [sessionProtocolName, setSessionProtocolName] = useState('')
  const [sessionDeviceId, setSessionDeviceId] = useState(null)
  const [sessionMemberName, setSessionMemberName] = useState('')
  const [sessionTitle, setSessionTitle] = useState('')
  // The session id the header meta was loaded for. The input box is a draft the
  // user can retype, so edits must never target whatever it happens to hold.
  const [loadedSessionId, setLoadedSessionId] = useState('')
  const [sessionTitleExpanded, setSessionTitleExpanded] = useState(false)
  const [markupFiles, setMarkupFiles]   = useState([])
  const [activeMarkupFileId, setActiveMarkupFileId] = useState('')
  const [sessionRecordAvailable, setSessionRecordAvailable] = useState(false)
  const [isSaving, setIsSaveLoading]    = useState(false)
  const [pendingImportFilename, setPendingImportFilename] = useState('')

  // Sessions list (for autocomplete)
  const [sessionsList, setSessionsList]               = useState([])
  const [sessionsListLoading, setSessionsListLoading] = useState(false)
  const [showSessionDropdown, setShowSessionDropdown] = useState(false)
  const sessionInputRef = useRef(null)
  const dropdownRef     = useRef(null)

  // Files
  const [videoUrl, setVideoUrl]         = useState(null)
  const [videoName, setVideoName]       = useState('')

  // Data
  const [parquetData, setParquetData]   = useState(null)
  const [columns, setColumns]           = useState([])
  const [sensorNames, setSensorNames]   = useState([])
  const [showSensor1, setShowSensor1]   = useState(true)
  const [showSensor2, setShowSensor2]   = useState(true)
  const [showSpeedTracker, setShowSpeedTracker] = useState(false)
  const [speedPredict, setSpeedPredict] = useState(null)
  const [predictLoading, setPredictLoading] = useState(false)
  const [showSpeedPredict, setShowSpeedPredict] = useState(false)
  const [showDistancePredict, setShowDistancePredict] = useState(false)
  const [extraCalculatorsOpen, setExtraCalculatorsOpen] = useState(false)
  const [protocolDetectorsOpen, setProtocolDetectorsOpen] = useState(false)
  const [calculatorResults, setCalculatorResults] = useState({})
  const [activeCalculators, setActiveCalculators] = useState([])
  const [calculatorLoading, setCalculatorLoading] = useState('')
  const [selectedCalculatorContact, setSelectedCalculatorContact] = useState(null)
  const [turnDetectionFeet, setTurnDetectionFeet] = useState({
    'protocol-shuttle-detector': 'both',
    'protocol-beep-detector': 'both',
    'protocol-ttest-detector': 'both',
  })
  const [weightKg, setWeightKg] = useState('70')
  const [imuTargetSensor, setImuTargetSensor] = useState('auto')
  const [imuProcessing, setImuProcessing] = useState(false)
  const [imuApplied, setImuApplied] = useState(false)
  const [checkHzData, setCheckHzData]   = useState(null)
  const [selectedCols, setSelectedCols] = useState([])
  const [timeCol, setTimeCol]           = useState('Time')
  const [offsetS1, setOffsetS1]         = useState(0)
  const [offsetS2, setOffsetS2]         = useState(0)
  const [offsetST, setOffsetST]         = useState(0)
  const [timeUnit, setTimeUnit]         = useState('ms')

  // Video state
  const [videoDuration, setVideoDuration] = useState(0)
  const [currentTime, setCurrentTime]     = useState(0)

  // Video zoom/pan
  const [zoom, setZoom]   = useState(1)
  const [panX, setPanX]   = useState(0)
  const [panY, setPanY]   = useState(0)

  // UI
  const [status, setStatus]         = useState({ text: '', type: 'idle' })
  const [chartReady, setChartReady] = useState(false)
  const [dragOver, setDragOver]     = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [dataPanelOpen, setDataPanelOpen]       = useState(true)
  const [chartPanelOpen, setChartPanelOpen]     = useState(true)
  const [columnsPanelOpen, setColumnsPanelOpen] = useState(false)
  const [modelsPanelOpen, setModelsPanelOpen]   = useState(false)
  const [videoPanelOpen, setVideoPanelOpen]     = useState(false)
  const [chartReorder, setChartReorder]         = useState(null)
  const [sidebarWidth, setSidebarWidth]         = useState(292)
  const [videoPanelWidth, setVideoPanelWidth]   = useState(null)
  const [labMenuOpen, setLabMenuOpen]           = useState(false)
  const [chartsLocked, setChartsLocked]         = useState(false)
  const [mobileTab, setMobileTab]               = useState('data')
  const [isMobile, setIsMobile]                 = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches,
  )
  const labMenuRef = useRef(null)

  // Labeling
  const [labelingMode, setLabelingMode]           = useState(false)
  const [currentFoot, setCurrentFoot]             = useState('left')
  const [leftContacts, setLeftContacts]           = useState([])
  const [rightContacts, setRightContacts]         = useState([])
  const [showLeftPatterns, setShowLeftPatterns]   = useState(true)
  const [showRightPatterns, setShowRightPatterns] = useState(true)
  const [showGaps, setShowGaps]                   = useState(false)
  const [selectedMarkup, setSelectedMarkup]       = useState(null)
  const [anglesUnwrapped, setAnglesUnwrapped]     = useState(false)
  const [mirrorLeft, setMirrorLeft]               = useState(false)
  const [relabelStep, setRelabelStep]             = useState(null)
  // Gyro yaw drift: the estimate runs once per loaded session and both series
  // stay in memory, so the toggle swaps between them without recomputing.
  const [yawDrift, setYawDrift]                   = useState(null)
  const [yawFixed, setYawFixed]                   = useState(false)
  const [correctedXDataCol, setCorrectedXDataCol] = useState(null)

  // Refs
  const videoRef        = useRef(null)
  const videoWrapRef    = useRef(null)
  const videoSideRef    = useRef(null)
  const chartAreaRef    = useRef(null)
  const chartDivRef     = useRef(null)
  const chartNativeClickRef = useRef(null)
  const timelineRef     = useRef(null)
  const videoUrlRef     = useRef(null)
  const offsetS1Ref     = useRef(0)
  const offsetS2Ref     = useRef(0)
  const offsetSTRef     = useRef(0)
  const showSpeedTrackerRef = useRef(false)
  const timeUnitRef     = useRef('ms')
  const lastTRef        = useRef(null)
  const plotInitRef      = useRef(false)
  const contactShapesRef = useRef([])
  const gapShapesRef     = useRef([])
  const calculatorShapesRef = useRef([])
  const calculatorResultsRef = useRef({})
  const activeCalculatorsRef = useRef([])
  const calculatorDataVersionRef = useRef(0)
  const cursorShapesRef  = useRef([])
  const selectedColsRef  = useRef([])
  const columnSelectionInitializedRef = useRef(false)
  const anglesUnwrappedRef = useRef(false)
  const mirrorLeftRef      = useRef(false)
  const isDragging       = useRef(false)
  const isVideoPan      = useRef(false)
  const vidLblRef       = useRef(null)
  const imuLblRef       = useRef(null)
  const labelingRef      = useRef(false)
  const currentFootRef   = useRef('left')
  const leftContactsRef  = useRef([])
  const rightContactsRef = useRef([])
  const showLeftRef      = useRef(true)
  const showRightRef     = useRef(true)
  const showGapsRef      = useRef(false)
  const s1TraceIdxRef    = useRef([])
  const s2TraceIdxRef    = useRef([])
  const stTraceIdxRef    = useRef([])
  const selectedMarkupRef = useRef(null)
  const relabelStepRef   = useRef(null)
  const subplotRangesRef = useRef({})
  const chartsLockedRef  = useRef(false)
  const chartReorderRef  = useRef(null)
  const importedCsvTextRef = useRef('')
  const skipClearImportCsvRef = useRef(false)
  // Raw IMU channels as loaded, kept until the postprocessing is undone.
  const imuOriginalRef = useRef(null)

  const insoleSensorNames = useMemo(
    () => sensorNames.filter(n => n !== SPEED_TRACKER),
    [sensorNames],
  )
  const sensorGroups = useMemo(
    () => groupSensorNamesByFoot(insoleSensorNames),
    [insoleSensorNames],
  )
  const hasSpeedTracker = useMemo(
    () => sensorNames.includes(SPEED_TRACKER),
    [sensorNames],
  )

  useEffect(() => { offsetS1Ref.current    = offsetS1     }, [offsetS1])
  useEffect(() => { offsetS2Ref.current    = offsetS2     }, [offsetS2])
  useEffect(() => { offsetSTRef.current    = offsetST     }, [offsetST])
  useEffect(() => { showSpeedTrackerRef.current = showSpeedTracker }, [showSpeedTracker])
  useEffect(() => { timeUnitRef.current    = timeUnit     }, [timeUnit])
  useEffect(() => { labelingRef.current    = labelingMode }, [labelingMode])
  useEffect(() => { currentFootRef.current = currentFoot  }, [currentFoot])
  useEffect(() => { selectedColsRef.current = selectedCols }, [selectedCols])
  useEffect(() => { showGapsRef.current = showGaps }, [showGaps])
  useEffect(() => { anglesUnwrappedRef.current = anglesUnwrapped }, [anglesUnwrapped])
  useEffect(() => { mirrorLeftRef.current = mirrorLeft }, [mirrorLeft])
  useEffect(() => { selectedMarkupRef.current = selectedMarkup }, [selectedMarkup])
  useEffect(() => { relabelStepRef.current = relabelStep }, [relabelStep])
  useEffect(() => { calculatorResultsRef.current = calculatorResults }, [calculatorResults])
  useEffect(() => { activeCalculatorsRef.current = activeCalculators }, [activeCalculators])
  useEffect(() => { chartsLockedRef.current = chartsLocked }, [chartsLocked])

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 768px)')
    const onChange = () => setIsMobile(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    if (!selectedMarkup) return
    const contacts = selectedMarkup.foot === 'left' ? leftContacts : rightContacts
    if (selectedMarkup.index < contacts.length) return
    const timeout = window.setTimeout(() => setSelectedMarkup(null), 0)
    return () => window.clearTimeout(timeout)
  }, [leftContacts, rightContacts, selectedMarkup])

  useEffect(() => {
    if (skipClearImportCsvRef.current) {
      skipClearImportCsvRef.current = false
      return
    }
    if (importedCsvTextRef.current) importedCsvTextRef.current = ''
  }, [leftContacts, rightContacts])

  // ── Auth ──────────────────────────────────────────────────────────────────
  const handleLogin = useCallback(async (e) => {
    e.preventDefault()
    setAuthLoading(true)
    setLoginError('')
    try {
      const resp = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
        body: JSON.stringify({ email: loginEmail, password: loginPassword }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }
      const data = await resp.json()
      const tok = data.access_token || data.token
      if (!tok) throw new Error('Токен не получен от сервера')
      setToken(tok)
      sessionStorage.setItem('auth_token', tok)
    } catch (err) {
      if (err instanceof TypeError) {
        setLoginError('API-сервер недоступен. Запустите backend на порту 8000.')
      } else {
        setLoginError(err.message)
      }
    } finally {
      setAuthLoading(false)
    }
  }, [loginEmail, loginPassword])

  const handleLogout = useCallback(() => {
    setToken('')
    setSessionsList([])
    setSessionsListLoading(false)
    setSessionProtocolName('')
    setSessionDeviceId(null)
    setSessionMemberName('')
    setSessionTitle('')
    setSessionTitleExpanded(false)
    setLoadedSessionId('')
    setSessionRecordAvailable(false)
    sessionStorage.removeItem('auth_token')
  }, [])

  // ── Sessions list fetch ───────────────────────────────────────────────────
  useEffect(() => {
    if (!token) return
    let cancelled = false
    queueMicrotask(() => {
      if (!cancelled) setSessionsListLoading(true)
    })
    fetch(`${MARKUP_API}/sessions?page_size=100`, {
      headers: { 'accept': 'application/json', 'Authorization': `Bearer ${token}` },
    })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => { if (!cancelled) setSessionsList(data.items || []) })
      .catch(() => { if (!cancelled) setSessionsList([]) })
      .finally(() => { if (!cancelled) setSessionsListLoading(false) })
    return () => { cancelled = true }
  }, [token])

  const filteredSessions = useMemo(() => {
    const q = sessionId.trim().toLowerCase()
    if (!q) return sessionsList.slice(0, 25)
    return sessionsList.filter(s =>
      String(s.id).includes(q) ||
      (s.member_name && s.member_name.toLowerCase().includes(q)) ||
      (s.session_title && s.session_title.toLowerCase().includes(q)) ||
      (s.protocol_name && s.protocol_name.toLowerCase().includes(q)) ||
      (s.device_id != null && String(s.device_id).includes(q))
    ).slice(0, 25)
  }, [sessionsList, sessionId])

  // close dropdown on outside click
  useEffect(() => {
    const onDown = (e) => {
      if (
        sessionInputRef.current && !sessionInputRef.current.contains(e.target) &&
        dropdownRef.current && !dropdownRef.current.contains(e.target)
      ) setShowSessionDropdown(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  useEffect(() => {
    if (!labMenuOpen) return
    const onDown = (e) => {
      if (labMenuRef.current && !labMenuRef.current.contains(e.target)) {
        setLabMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [labMenuOpen])

  const fetchSessionMarkupsFromDb = useCallback(async (sid, { restoreLatest = false } = {}) => {
    const resp = await fetch(`${MARKUP_API}/sessions/${sid}`, {
      headers: { 'accept': 'application/json', 'Authorization': `Bearer ${token}` },
    })
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}))
      throw new Error(parseApiError(errData, resp.status))
    }

    const result = await resp.json()
    const files = result.additional_info?.markup_files || []
    setMarkupFiles(files)

    if (restoreLatest && files.length > 0) {
      const lastFile = files[files.length - 1]
      setActiveMarkupFileId(lastFile.id)
      setLeftContacts(lastFile.leftContacts || [])
      setRightContacts(lastFile.rightContacts || [])
      importedCsvTextRef.current = lastFile.csv || ''
      if (lastFile.meta) {
        if (lastFile.meta.offsetS1 !== undefined) setOffsetS1(lastFile.meta.offsetS1)
        if (lastFile.meta.offsetS2 !== undefined) setOffsetS2(lastFile.meta.offsetS2)
        if (lastFile.meta.offsetST !== undefined) setOffsetST(lastFile.meta.offsetST)
        if (lastFile.meta.timeUnit !== undefined) setTimeUnit(lastFile.meta.timeUnit)
      }
    }

    return result
  }, [token])

  // ── Gyro yaw drift ────────────────────────────────────────────────────────
  /**
   * Estimate the feet's differential yaw drift once, and cache the corrected
   * XData beside the raw column. Both live in memory for the rest of the
   * session so the toggle is instant — no re-estimate, no re-download.
   */
  const prepareYawDrift = useCallback((colMap) => {
    // Some sessions carry Time in seconds; the estimate reads milliseconds.
    const columns = withTimeInMs(colMap)
    const drift = estimateYawDrift(columns)
    setYawDrift(drift)
    setCorrectedXDataCol(correctedXData(columns, drift))
    setYawFixed(false)
    return drift
  }, [])

  const resetYawDrift = useCallback(() => {
    setYawDrift(null)
    setCorrectedXDataCol(null)
    setYawFixed(false)
  }, [])

  const saveSessionTitle = useCallback(async (nextTitle) => {
    const sid = loadedSessionId
    if (!sid) throw new Error('Сессия не загружена')
    const resp = await fetch(`${MARKUP_API}/sessions/${sid}/title`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ session_title: nextTitle }),
    })
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}))
      throw new Error(parseApiError(errData, resp.status))
    }
    const result = await resp.json().catch(() => ({}))
    const saved = normalizeSessionTitle(result.session_title ?? nextTitle)
    setSessionTitle(saved)
    // The dropdown list was fetched once at login; keep its row in step.
    setSessionsList(list => list.map(item => (
      String(item.id) === sid ? { ...item, session_title: saved || null } : item
    )))
    setStatus({ text: '✓ Название сессии сохранено', type: 'ok' })
  }, [loadedSessionId, token])

  // ── Session loader ────────────────────────────────────────────────────────
  const loadSession = useCallback(async () => {
    const sid = sessionId.trim()
    if (!sid) { setStatus({ text: 'Введите номер сессии', type: 'error' }); return }
    const previousSelectedCols = columnSelectionInitializedRef.current
      ? [...selectedColsRef.current]
      : null

    setStatus({ text: `Загружаю сессию ${sid}…`, type: 'loading' })
    setSessionLabel(`Сессия #${sid}`)
    setSessionProtocolName('')
    setSessionDeviceId(null)
    setSessionMemberName('')
    setSessionTitle('')
    setSessionTitleExpanded(false)
    setLoadedSessionId('')
    setChartReady(false)
    plotInitRef.current = false
    if (chartDivRef.current) {
      if (chartNativeClickRef.current) {
        chartDivRef.current.removeEventListener('click', chartNativeClickRef.current, true)
        chartNativeClickRef.current = null
      }
      Plotly.purge(chartDivRef.current)
    }
    setParquetData(null)
    setColumns([])
    setColumnsPanelOpen(false)
    setSensorNames([])
    setLeftContacts([])
    setRightContacts([])
    setMarkupFiles([])
    setActiveMarkupFileId('')
    setSessionRecordAvailable(false)
    setPendingImportFilename('')
    importedCsvTextRef.current = ''
    imuOriginalRef.current = null
    setImuApplied(false)
    subplotRangesRef.current = {}
    setRelabelStep(null)
    setShowLeftPatterns(true)
    setShowRightPatterns(true)
    setShowSensor1(true)
    setShowSensor2(true)
    setShowSpeedTracker(false)
    setSpeedPredict(null)
    setShowSpeedPredict(false)
    setShowDistancePredict(false)
    calculatorDataVersionRef.current += 1
    setCalculatorResults({})
    setActiveCalculators([])
    setCalculatorLoading('')
    setSelectedCalculatorContact(null)
    setExtraCalculatorsOpen(false)
    setProtocolDetectorsOpen(false)
    setOffsetST(0)
    setShowGaps(false)
    setCheckHzData(null)
    setSelectedMarkup(null)
    anglesUnwrappedRef.current = false
    setAnglesUnwrapped(false)
    mirrorLeftRef.current = false
    setMirrorLeft(false)
    resetYawDrift()

    try {
      const [metadataResp, parquetResp] = await Promise.all([
        fetch(`${MARKUP_API}/sessions/${sid}`, {
          headers: { 'accept': 'application/json', 'Authorization': `Bearer ${token}` },
        }),
        fetch(`${MARKUP_API}/sessions/${sid}/parquet`, {
          headers: {
            'accept': 'application/vnd.apache.parquet',
            'Authorization': `Bearer ${token}`,
          },
        }),
      ])
      if (!parquetResp.ok) {
        const errData = await parquetResp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, parquetResp.status))
      }

      const metadataAvailable = metadataResp.ok
      const result = metadataAvailable
        ? await metadataResp.json()
        : { additional_info: null }
      const protocolName = result.protocol_name || result.protocolName || ''
      const deviceId = result.device_id ?? result.deviceId ?? null
      setSessionRecordAvailable(metadataAvailable)
      setSessionProtocolName(protocolName)
      setSessionDeviceId(hasSessionMetaValue(deviceId) ? deviceId : null)
      setSessionMemberName(normalizeMemberName(result.member_name ?? result.memberName))
      setSessionTitle(normalizeSessionTitle(result.session_title ?? result.sessionTitle))
      setLoadedSessionId(metadataAvailable ? String(sid) : '')
      if (typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches) {
        setMobileTab('chart')
      }

      const initialMarkupFiles = result.additional_info?.markup_files || []
      setMarkupFiles(initialMarkupFiles)
      if (initialMarkupFiles.length > 0) {
        const lastFile = initialMarkupFiles[initialMarkupFiles.length - 1]
        setActiveMarkupFileId(lastFile.id)
        setLeftContacts(lastFile.leftContacts || [])
        setRightContacts(lastFile.rightContacts || [])
        importedCsvTextRef.current = lastFile.csv || ''
        if (lastFile.meta) {
          if (lastFile.meta.offsetS1 !== undefined) setOffsetS1(lastFile.meta.offsetS1)
          if (lastFile.meta.offsetS2 !== undefined) setOffsetS2(lastFile.meta.offsetS2)
          if (lastFile.meta.offsetST !== undefined) setOffsetST(lastFile.meta.offsetST)
          if (lastFile.meta.timeUnit !== undefined) setTimeUnit(lastFile.meta.timeUnit)
        }
      }

      const rows = await parquetReadObjects({ file: await parquetResp.arrayBuffer() })

      if (!rows?.length) { setStatus({ text: 'Сессия пустая', type: 'error' }); return }

      const colMap = rowsToColMap(rows)
      const tCol = detectTimeCol(Object.keys(colMap))
      addDerivedSessionColumns(colMap, tCol, result.additional_info)
      setParquetData(colMap)
      setTimeCol(tCol)
      const drift = prepareYawDrift(colMap)

      const names = sortSensorNames(colMap)
      const insole = names.filter(n => n !== SPEED_TRACKER)
      const hasST = names.includes(SPEED_TRACKER)
      setSensorNames(names)

      const numCols = computeNumericColumns(colMap, tCol)
      setColumns(numCols)
      const nextSelectedCols = previousSelectedCols === null
        ? buildDefaultCols(numCols, hasST, colMap, insole)
        : previousSelectedCols.filter(col => numCols.includes(col))
      columnSelectionInitializedRef.current = true
      selectedColsRef.current = nextSelectedCols
      setSelectedCols(nextSelectedCols)
      setShowSpeedTracker(hasST)
      setOffsetST(hasST ? computeAutoOffsetST(colMap, tCol, insole) : 0)

      const tVals   = (colMap[tCol] || []).map(safeNum).filter(v => v !== null)
      const tMax    = tVals.length ? arrayMax(tVals) : 0
      const autoUnit = tMax > 3600 ? 'ms' : 's'
      setTimeUnit(autoUnit)
      timeUnitRef.current = autoUnit

      const gapStats = computeGapStats(colMap, tCol)
      const gapCount = Object.values(gapStats)
        .reduce((count, sensor) => count + sensor.gaps.length, 0)
      setCheckHzData(gapStats)
      setShowGaps(gapCount > 0)

      const stHint = hasST ? ' · SpeedTracker' : ''
      const gapHint = gapCount ? ` · ${gapCount} пропуск(ов)` : ' · без пропусков'
      const yawHint = drift.applied
        ? ` · дрейф ${drift.differentialDegS.toFixed(1)} °/с`
        : ' · дрейф не найден'
      setStatus({
        text: `✓ GCS · ${rows.length} строк · ${numCols.length} колонок · ${autoUnit}${stHint}${gapHint}${yawHint}`,
        type: 'ok',
      })
    } catch (err) {
      setStatus({ text: `Ошибка: ${err.message}`, type: 'error' })
    }
  }, [sessionId, token, prepareYawDrift, resetYawDrift])

  const totalGaps = useMemo(() => {
    if (!checkHzData) return 0
    return Object.values(checkHzData).reduce((n, s) => n + (s.gaps?.length || 0), 0)
  }, [checkHzData])

  // ── Contact + gap shapes ──────────────────────────────────────────────────
  const updateOverlayShapes = useCallback(() => {
    if (!chartDivRef.current || !plotInitRef.current) return

    const contactShapes = []
    const sm = selectedMarkupRef.current
    const nSubplots = selectedColsRef.current.length || 1

    const pushAcrossSubplots = (target, shape) => {
      for (let subplotIndex = 0; subplotIndex < nSubplots; subplotIndex++) {
        target.push({
          ...shape,
          xref: subplotIndex === 0 ? 'x' : `x${subplotIndex + 1}`,
          yref: subplotIndex === 0 ? 'y domain' : `y${subplotIndex + 1} domain`,
        })
      }
    }

    const pushContactShapes = (contacts, fillColor, lineColor, foot) => {
      const isSelectedFoot = sm?.foot === foot
      for (let i = 0; i + 1 < contacts.length; i += 2) {
        const x0 = Math.min(contacts[i], contacts[i + 1])
        const x1 = Math.max(contacts[i], contacts[i + 1])
        const isSel = isSelectedFoot && (sm.index === i || sm.index === i + 1)
        pushAcrossSubplots(contactShapes, {
          type: 'rect', x0, x1,
          y0: 0, y1: 1,
          fillcolor: isSel ? SEL_FILL : fillColor,
          line: { color: isSel ? SEL_LINE : lineColor, width: isSel ? 3 : 1.5 },
          layer: 'below',
        })
      }
      if (contacts.length % 2 === 1) {
        const i = contacts.length - 1
        const t = contacts[i]
        const isSel = isSelectedFoot && sm.index === i
        pushAcrossSubplots(contactShapes, {
          type: 'line', x0: t, x1: t,
          y0: 0, y1: 1,
          line: {
            color: isSel ? SEL_LINE : lineColor,
            width: isSel ? 3.5 : 2,
            dash: isSel ? 'solid' : 'dot',
          },
        })
      }
    }

    if (showLeftRef.current)  pushContactShapes(leftContactsRef.current,  L_FILL, L_LINE, 'left')
    if (showRightRef.current) pushContactShapes(rightContactsRef.current, R_FILL, R_LINE, 'right')

    if (sm) {
      const contacts = sm.foot === 'left' ? leftContactsRef.current : rightContactsRef.current
      const t = contacts[sm.index]
      if (t != null) contactShapes.push(...buildSelectedPointShapes(t, nSubplots))
    }
    contactShapesRef.current = contactShapes

    const calculatorShapes = []
    const timeScale = timeUnitRef.current === 'ms' ? 1000 : 1
    activeCalculatorsRef.current.forEach(calculatorId => {
      const style = CALCULATOR_BY_ID[calculatorId]
      const result = calculatorResultsRef.current[calculatorId]
      if (!style || !result?.contacts?.length) return

      result.contacts.forEach(contact => {
        if (contact.foot === 'left' && !showSensor1) return
        if (contact.foot === 'right' && !showSensor2) return
        const eventStyle = calculatorEventStyle(style, contact)
        const shift = contact.foot === 'right'
          ? offsetS2Ref.current
          : contact.foot === 'left'
            ? offsetS1Ref.current
            : 0
        const x0 = contact.start_time_s * timeScale + shift
        const x1 = contact.end_time_s * timeScale + shift
        if (!isFinite(x0) || !isFinite(x1) || x1 <= x0) return
        pushAcrossSubplots(calculatorShapes, {
          type: 'rect', x0, x1,
          y0: 0, y1: 1,
          fillcolor: eventStyle.fill,
          line: {
            color: eventStyle.color,
            width: eventStyle.width,
            dash: eventStyle.dash,
          },
          layer: 'below',
        })
      })
    })
    calculatorShapesRef.current = calculatorShapes

    const gapShapes = []
    if (showGapsRef.current && checkHzData) {
      const seen = new Set()
      const intervals = []
      const addSensorGaps = (name, shift) => {
        const gaps = checkHzData[name]?.gaps
        if (!gaps?.length) return
        for (const [startT, endT] of gaps) {
          // Gap timestamps use the same raw Time units as the plotted traces.
          // Dividing ms by 1000 here used to place every red band off-chart.
          const x0 = Math.min(startT, endT) + shift
          const x1 = Math.max(startT, endT) + shift
          const key = `${x0}|${x1}`
          if (seen.has(key)) continue
          seen.add(key)
          intervals.push([x0, x1])
        }
      }

      insoleSensorNames.forEach(name => {
        const foot = sensorFootForName(name, insoleSensorNames)
        const visible = foot === 'left' ? showSensor1 : showSensor2
        if (!visible) return
        const shift = foot === 'left' ? offsetS1Ref.current : offsetS2Ref.current
        addSensorGaps(name, shift)
      })
      if (showSpeedTracker) addSensorGaps(SPEED_TRACKER, offsetSTRef.current)

      const nSubplots = selectedColsRef.current.length || 1
      gapShapes.push(...buildGapBandShapes(intervals, nSubplots))
    }
    gapShapesRef.current = gapShapes

    Plotly.relayout(chartDivRef.current, {
      shapes: [
        ...gapShapesRef.current,
        ...calculatorShapesRef.current,
        ...contactShapesRef.current,
        ...cursorShapesRef.current,
      ],
    })
  }, [checkHzData, insoleSensorNames, showSensor1, showSensor2, showSpeedTracker])

  useEffect(() => {
    leftContactsRef.current  = leftContacts
    rightContactsRef.current = rightContacts
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [leftContacts, rightContacts, updateOverlayShapes])

  useEffect(() => {
    showLeftRef.current  = showLeftPatterns
    showRightRef.current = showRightPatterns
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [showLeftPatterns, showRightPatterns, updateOverlayShapes])

  useEffect(() => {
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [showGaps, checkHzData, showSensor1, showSensor2, showSpeedTracker, offsetS1, offsetS2, offsetST, timeUnit, selectedCols, selectedMarkup, calculatorResults, activeCalculators, updateOverlayShapes])

  useEffect(() => {
    if (!chartReady || !chartDivRef.current) return
    if (s1TraceIdxRef.current.length)
      Plotly.restyle(chartDivRef.current, { visible: showSensor1 }, s1TraceIdxRef.current)
    if (s2TraceIdxRef.current.length)
      Plotly.restyle(chartDivRef.current, { visible: showSensor2 }, s2TraceIdxRef.current)
  }, [showSensor1, showSensor2, chartReady])

  const undoContact = useCallback(() => {
    if (currentFootRef.current === 'left') setLeftContacts(p => p.slice(0, -1))
    else setRightContacts(p => p.slice(0, -1))
  }, [])

  const clearCurrentContacts = useCallback(() => {
    if (currentFootRef.current === 'left') setLeftContacts([])
    else setRightContacts([])
  }, [])

  const clearAllContacts = useCallback(() => {
    setLeftContacts([])
    setRightContacts([])
    setSelectedMarkup(null)
  }, [])

  const deleteSelectedMarkup = useCallback(() => {
    if (!selectedMarkup) return
    const { foot, index } = selectedMarkup
    const pairStart = getPairStartIndex(index)
    const removeInterval = (prev) => {
      if (pairStart + 1 < prev.length) {
        return prev.filter((_, i) => i !== pairStart && i !== pairStart + 1)
      }
      if (pairStart < prev.length) {
        return prev.filter((_, i) => i !== pairStart)
      }
      return prev
    }
    if (foot === 'left') setLeftContacts(removeInterval)
    else setRightContacts(removeInterval)
    setSelectedMarkup(null)
  }, [selectedMarkup])

  const generateCsvString = useCallback(() => {
    if (!parquetData) return ''
    const allCols = Object.keys(parquetData)
    const timeArr = parquetData[timeCol] || []
    const nameArr = parquetData['Name']  || []
    const n = timeArr.length
    const leftSensorNames = new Set(sensorGroups.left)
    const rightSensorNames = new Set(sensorGroups.right)

    const buildIv = (contacts, offset) => {
      const out = []
      for (let i = 0; i + 1 < contacts.length; i += 2)
        out.push([
          Math.min(contacts[i], contacts[i + 1]) - offset,
          Math.max(contacts[i], contacts[i + 1]) - offset,
        ])
      return out
    }
    const lIv = buildIv(leftContactsRef.current, offsetS1Ref.current)
    const rIv = buildIv(rightContactsRef.current, offsetS2Ref.current)
    const inIv = (t, ivs) => {
      const tv = safeNum(t); if (tv === null) return false
      return ivs.some(([a, b]) => tv >= a && tv <= b)
    }

    const hdr = [...allCols, 'Target'].join(',')
    const rows = []
    for (let i = 0; i < n; i++) {
      const name = nameArr[i] || ''
      const t    = timeArr[i]
      const target = name === SPEED_TRACKER
        ? ''
        : rightSensorNames.has(name)
          ? (inIv(t, rIv) ? 1 : 0)
          : leftSensorNames.has(name)
            ? (inIv(t, lIv) ? 1 : 0)
            : ''
      const vals = allCols.map(c => {
        const v = parquetData[c][i]
        return v == null ? '' : String(v)
      })
      vals.push(String(target))
      rows.push(vals.join(','))
    }

    return hdr + '\n' + rows.join('\n')
  }, [parquetData, timeCol, sensorGroups])

  const exportLabels = useCallback(() => {
    const csv = generateCsvString()
    if (!csv) return
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url
    a.download = (sessionLabel || 'session').replace(/\s+/g, '_') + '_labeled.csv'
    document.body.appendChild(a); a.click()
    document.body.removeChild(a); URL.revokeObjectURL(url)
  }, [generateCsvString, sessionLabel])

  const handleSelectMarkupFile = useCallback((id) => {
    setRelabelStep(null)
    setPendingImportFilename('')
    importedCsvTextRef.current = ''
    if (id === 'new' || !id) {
      setActiveMarkupFileId('new')
      setLeftContacts([])
      setRightContacts([])
      return
    }
    const file = markupFiles.find(f => f.id === id)
    if (file) {
      setActiveMarkupFileId(file.id)
      setLeftContacts(file.leftContacts || [])
      setRightContacts(file.rightContacts || [])
      importedCsvTextRef.current = file.csv || ''
      if (file.meta) {
        if (file.meta.offsetS1 !== undefined) setOffsetS1(file.meta.offsetS1)
        if (file.meta.offsetS2 !== undefined) setOffsetS2(file.meta.offsetS2)
        if (file.meta.offsetST !== undefined) setOffsetST(file.meta.offsetST)
        if (file.meta.timeUnit !== undefined) setTimeUnit(file.meta.timeUnit)
      }
    }
  }, [markupFiles])

  const saveMarkupToDb = useCallback(async () => {
    const sid = sessionId.trim()
    if (!sid) {
      setStatus({ text: 'Укажите ID сессии в поле слева', type: 'error' })
      return
    }
    if (!sessionRecordAvailable) {
      setStatus({
        text: 'Данные загружены из GCS, но записи сессии в БД нет — сохранить разметку в БД нельзя',
        type: 'error',
      })
      return
    }
    if (leftContactsRef.current.length === 0 && rightContactsRef.current.length === 0) {
      setStatus({ text: 'Нет разметки для сохранения', type: 'error' })
      return
    }

    setIsSaveLoading(true)
    try {
      const currentSession = await fetchSessionMarkupsFromDb(sid)
      const currentAdditionalInfo = currentSession.additional_info || {}
      const existingMarkupFiles = Array.isArray(currentAdditionalInfo.markup_files)
        ? [...currentAdditionalInfo.markup_files]
        : []

      const isNew = !activeMarkupFileId || activeMarkupFileId === 'new'
      const fileId = isNew ? `mf_${Date.now()}` : activeMarkupFileId
      const fileIndex = isNew ? -1 : existingMarkupFiles.findIndex(f => f.id === fileId)

      const csv = (isNew && importedCsvTextRef.current)
        ? importedCsvTextRef.current
        : generateCsvString()
      if (!csv) throw new Error('Не удалось сформировать CSV')

      const defaultFilename = pendingImportFilename
        || `markup_${sid}_v${existingMarkupFiles.length + 1}.csv`

      const newFile = {
        id: fileId,
        filename: !isNew && fileIndex >= 0
          ? existingMarkupFiles[fileIndex].filename
          : defaultFilename,
        type: 'contact_target_csv',
        updated_at: new Date().toISOString(),
        leftContacts: [...leftContactsRef.current],
        rightContacts: [...rightContactsRef.current],
        meta: {
          offsetS1: offsetS1Ref.current,
          offsetS2: offsetS2Ref.current,
          offsetST: offsetSTRef.current,
          timeUnit: timeUnitRef.current,
        },
        csv,
      }

      const updatedFiles = [...existingMarkupFiles]
      if (fileIndex >= 0) {
        updatedFiles[fileIndex] = newFile
      } else {
        updatedFiles.push(newFile)
      }

      const updatedAdditionalInfo = {
        ...currentAdditionalInfo,
        markup_files: updatedFiles,
      }

      const resp = await fetch(`${MARKUP_API}/sessions/${sid}/additional-info`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          additional_info: updatedAdditionalInfo,
        }),
      })

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }

      const updatedSession = await resp.json()

      const nextFiles = updatedSession.additional_info?.markup_files || updatedFiles
      setMarkupFiles(nextFiles)
      setActiveMarkupFileId(fileId)
      setPendingImportFilename('')
      importedCsvTextRef.current = csv

      setStatus({
        text: `✓ Разметка «${newFile.filename}» сохранена в БД (${nextFiles.length} верс.)`,
        type: 'ok',
      })
    } catch (err) {
      setStatus({ text: `Ошибка сохранения: ${err.message}`, type: 'error' })
    } finally {
      setIsSaveLoading(false)
    }
  }, [
    sessionId,
    sessionRecordAvailable,
    activeMarkupFileId,
    generateCsvString,
    token,
    pendingImportFilename,
    fetchSessionMarkupsFromDb,
  ])

  // Import works standalone: a CSV that carries sensor channels becomes the
  // dataset (no session number required, the previous chart is dropped), while
  // a Target-only CSV stays an overlay on the already loaded data. Target is
  // optional — an unlabeled CSV just loads as a session to mark up.
  const importLabeledCsv = useCallback(async (file) => {
    const previousSelectedCols = columnSelectionInitializedRef.current
      ? [...selectedColsRef.current]
      : null
    const sid = sessionId.trim()

    setStatus({ text: `Читаю ${file.name}…`, type: 'loading' })
    setRelabelStep(null)
    setSelectedMarkup(null)

    try {
      const text = await file.text()
      const { headers, rows } = parseCsvText(text)
      if (!rows.length) throw new Error('CSV пустой')

      const tCol = headers.find(c => c === 'Time')
        || headers.find(c => ['time', 'timestamp', 'Timestamp', 't'].includes(c))
      if (!tCol) throw new Error('Колонка Time не найдена в CSV')

      const hasTarget = headers.some(c => ['Target', 'target', 'Label', 'label'].includes(c))

      const colMap = rowsToColMap(rows)
      coerceCsvColumnsToNumbers(colMap)
      const isDataset = computeNumericColumns(colMap, tCol).length > 0

      let leftSensors = sensorGroups.left
      let rightSensors = sensorGroups.right
      let datasetHint = ''

      if (isDataset) {
        // ── Replace the current dataset with the CSV ──────────────────────
        setChartReady(false)
        plotInitRef.current = false
        if (chartDivRef.current) {
          if (chartNativeClickRef.current) {
            chartDivRef.current.removeEventListener('click', chartNativeClickRef.current, true)
            chartNativeClickRef.current = null
          }
          Plotly.purge(chartDivRef.current)
        }
        setColumnsPanelOpen(false)
        setLeftContacts([])
        setRightContacts([])
        setMarkupFiles([])
        setActiveMarkupFileId('')
        setPendingImportFilename('')
        setSessionRecordAvailable(false)
        setLoadedSessionId(null)
        setShowLeftPatterns(true)
        setShowRightPatterns(true)
        setShowSensor1(true)
        setShowSensor2(true)
        setShowSpeedTracker(false)
        setSpeedPredict(null)
        setShowSpeedPredict(false)
        setShowDistancePredict(false)
        calculatorDataVersionRef.current += 1
        setCalculatorResults({})
        setActiveCalculators([])
        setCalculatorLoading('')
        setSelectedCalculatorContact(null)
        setExtraCalculatorsOpen(false)
        setProtocolDetectorsOpen(false)
        setOffsetST(0)
        setShowGaps(false)
        setCheckHzData(null)
        anglesUnwrappedRef.current = false
        setAnglesUnwrapped(false)
        mirrorLeftRef.current = false
        setMirrorLeft(false)
        resetYawDrift()
        subplotRangesRef.current = {}
        importedCsvTextRef.current = ''
        imuOriginalRef.current = null
        setImuApplied(false)

        addDerivedSessionColumns(colMap, tCol, null)
        setParquetData(colMap)
        setTimeCol(tCol)
        prepareYawDrift(colMap)

        const names = sortSensorNames(colMap)
        const insole = names.filter(n => n !== SPEED_TRACKER)
        const hasST = names.includes(SPEED_TRACKER)
        setSensorNames(names)

        const numCols = computeNumericColumns(colMap, tCol)
        setColumns(numCols)
        const keptCols = previousSelectedCols === null
          ? []
          : previousSelectedCols.filter(col => numCols.includes(col))
        // Columns of an unrelated session may not exist here — fall back to
        // defaults so the new chart is never empty.
        const nextSelectedCols = keptCols.length
          ? keptCols
          : buildDefaultCols(numCols, hasST, colMap, insole)
        columnSelectionInitializedRef.current = true
        selectedColsRef.current = nextSelectedCols
        setSelectedCols(nextSelectedCols)
        setShowSpeedTracker(hasST)
        setOffsetST(hasST ? computeAutoOffsetST(colMap, tCol, insole) : 0)

        const tVals = (colMap[tCol] || []).map(safeNum).filter(v => v !== null)
        const tMax = tVals.length ? arrayMax(tVals) : 0
        const autoUnit = tMax > 3600 ? 'ms' : 's'
        setTimeUnit(autoUnit)
        timeUnitRef.current = autoUnit

        const gapStats = computeGapStats(colMap, tCol)
        const gapCount = Object.values(gapStats)
          .reduce((count, sensor) => count + sensor.gaps.length, 0)
        setCheckHzData(gapStats)
        setShowGaps(gapCount > 0)

        setSessionLabel(sid ? `Сессия #${sid} · ${file.name}` : file.name)

        const groups = groupSensorNamesByFoot(insole)
        leftSensors = groups.left
        rightSensors = groups.right

        const stHint = hasST ? ' · SpeedTracker' : ''
        const gapHint = gapCount ? ` · ${gapCount} пропуск(ов)` : ' · без пропусков'
        datasetHint = `${rows.length} строк · ${numCols.length} колонок · ${autoUnit}${stHint}${gapHint}`

        // The session number stays optional — it only unlocks saving to the DB.
        if (sid && token) {
          try {
            const sess = await fetchSessionMarkupsFromDb(sid)
            setSessionRecordAvailable(true)
            setSessionProtocolName(sess.protocol_name || sess.protocolName || '')
            setSessionDeviceId(
              hasSessionMetaValue(sess.device_id ?? sess.deviceId)
                ? (sess.device_id ?? sess.deviceId)
                : null
            )
            setSessionMemberName(normalizeMemberName(sess.member_name ?? sess.memberName))
            setSessionTitle(normalizeSessionTitle(sess.session_title ?? sess.sessionTitle))
            setLoadedSessionId(String(sid))
          } catch {
            setSessionRecordAvailable(false)
          }
        }
      } else {
        if (!parquetData) {
          throw new Error('В CSV нет колонок с данными сенсоров — сначала загрузите сессию или parquet')
        }
        if (!hasTarget) {
          throw new Error('В CSV нет ни данных сенсоров, ни колонки Target')
        }
        if (!sensorGroups.left.length && !sensorGroups.right.length) {
          throw new Error('В данных сессии нет сенсоров стельки')
        }
        if (sid && token) {
          try { await fetchSessionMarkupsFromDb(sid) } catch { /* markups load on save */ }
        }
      }

      if (!hasTarget) {
        setStatus({
          text: `✓ ${file.name}: ${datasetHint} · разметки (Target) в файле нет`,
          type: 'ok',
        })
        return
      }

      const csvNames = new Set(rows.map(r => r.Name || r.name).filter(Boolean))
      const resolveSensors = (preferred, fallback) => {
        const matched = preferred.filter(name => csvNames.has(name))
        if (matched.length) return matched
        return fallback && csvNames.has(fallback) ? [fallback] : preferred
      }
      const leftNames = resolveSensors(leftSensors, 'ESP32_Sensor_1')
      const rightNames = resolveSensors(rightSensors, 'ESP32_Sensor_2')

      const { leftContacts: importedLeft, rightContacts: importedRight, leftCount, rightCount } =
        extractContactsFromLabeledCsv(
          rows,
          tCol,
          leftNames,
          rightNames,
          offsetS1Ref.current,
          offsetS2Ref.current,
        )

      if (leftCount === 0 && rightCount === 0) {
        if (isDataset) {
          setStatus({
            text: `✓ ${file.name}: ${datasetHint} · интервалов с Target=1 нет`,
            type: 'ok',
          })
          return
        }
        throw new Error('В CSV нет интервалов с Target=1')
      }

      setLeftContacts(importedLeft)
      setRightContacts(importedRight)
      setActiveMarkupFileId('new')
      setPendingImportFilename(file.name.replace(/\.csv$/i, '') + '.csv')
      skipClearImportCsvRef.current = true
      importedCsvTextRef.current = text
      setLabelingMode(true)
      setShowLeftPatterns(true)
      setShowRightPatterns(true)

      const savePrompt = sid ? ' Нажмите «Сохранить в БД».' : ''
      setStatus({
        text: `✓ Импорт ${file.name}: S1 ${leftCount} · S2 ${rightCount}.${savePrompt}`,
        type: 'ok',
      })
    } catch (err) {
      setStatus({ text: `Ошибка импорта CSV: ${err.message}`, type: 'error' })
    }
  }, [
    parquetData,
    sensorGroups,
    sessionId,
    token,
    fetchSessionMarkupsFromDb,
    prepareYawDrift,
    resetYawDrift,
  ])

  // ── Raw IMU postprocessing ────────────────────────────────────────────────
  // Swap in a version of the session with rewritten IMU channels and drop
  // everything derived from the previous signals.
  const applyImuColMap = useCallback((colMap) => {
    const numCols = computeNumericColumns(colMap, timeCol)
    setParquetData(colMap)
    setColumns(numCols)
    const keptCols = selectedColsRef.current.filter(col => numCols.includes(col))
    if (keptCols.length && keptCols.length !== selectedColsRef.current.length) {
      selectedColsRef.current = keptCols
      setSelectedCols(keptCols)
    }
    // Cached calculator results were computed on the previous signals.
    calculatorDataVersionRef.current += 1
    setCalculatorResults({})
    setActiveCalculators([])
    setCalculatorLoading('')
    setSelectedCalculatorContact(null)
    anglesUnwrappedRef.current = false
    setAnglesUnwrapped(false)
    mirrorLeftRef.current = false
    setMirrorLeft(false)
    // XData was rewritten, so the cached drift estimate no longer matches it.
    prepareYawDrift(colMap)
  }, [timeCol, prepareYawDrift])


  // Older sessions ship accelerations with gravity still in them and the
  // gyroscope in dps. The backend converts those to the new firmware channels
  // (linear acceleration in the BLE basis + Heading/Roll/Pitch) so the chart
  // and the calculators see the same shape as a modern session.
  const handlePreprocessImu = useCallback(async () => {
    if (!parquetData || imuProcessing) return

    setImuProcessing(true)
    setStatus({ text: 'Постпроцессинг IMU…', type: 'loading' })
    try {
      const resp = await fetch(`${CALCULATOR_API}/markup/preprocess-imu`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
        body: JSON.stringify({
          columns: parquetData,
          target_sensor: imuTargetSensor,
        }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }

      const data = await resp.json()
      const rows = data.rows || []
      const colMapFromResp = data.columns || (rows.length ? rowsToColMap(rows) : null)
      if (!colMapFromResp) throw new Error('Сервис вернул пустой результат')

      const processed = data.processed_sensors || []
      if (!processed.length) {
        setStatus({
          text: 'Сырых IMU-данных не найдено — сессия уже в формате новой прошивки',
          type: 'ok',
        })
        return
      }

      // Snapshot the channels as loaded — the backend returns the rows in the
      // order they were sent, so the arrays stay index-aligned. Taken once, so
      // undo returns to the original data even after several runs.
      if (!imuOriginalRef.current) {
        imuOriginalRef.current = Object.fromEntries(
          IMU_SNAPSHOT_COLUMNS.map(col => [col, parquetData[col]]),
        )
      }

      const colMap = { ...colMapFromResp }
      // Accelerations changed, so the derived TKEO channel is rebuilt.
      delete colMap['acc_tkeo']
      TKEO_PLOT_COLS.forEach(col => { delete colMap[col] })
      addAccTkeoColumn(colMap, timeCol)
      addTkeoColumns(colMap, timeCol)
      applyImuColMap(colMap)
      setImuApplied(true)

      setStatus({
        text: `✓ Постпроцессинг IMU · обработано датчиков: ${processed.length} (${processed.join(', ')})`,
        type: 'ok',
      })
    } catch (err) {
      setStatus({ text: `Ошибка обработки IMU: ${err.message}`, type: 'error' })
    } finally {
      setImuProcessing(false)
    }
  }, [parquetData, imuProcessing, imuTargetSensor, timeCol, applyImuColMap])

  const handleRevertImu = useCallback(() => {
    const snapshot = imuOriginalRef.current
    if (!parquetData || !snapshot || imuProcessing) return

    const colMap = { ...parquetData }
    Object.entries(snapshot).forEach(([col, values]) => {
      if (values === undefined) delete colMap[col]
      else colMap[col] = values
    })
    applyImuColMap(colMap)
    imuOriginalRef.current = null
    setImuApplied(false)
    setStatus({ text: '✓ Постпроцессинг IMU откачен — данные сессии исходные', type: 'ok' })
  }, [parquetData, imuProcessing, applyImuColMap])

  // ── Video zoom helpers ────────────────────────────────────────────────────
  const clampPan = useCallback((z, px, py) => {
    const el = videoWrapRef.current
    if (!el) return [px, py]
    const maxX = el.clientWidth  * (z - 1) / (2 * z)
    const maxY = el.clientHeight * (z - 1) / (2 * z)
    return [
      Math.max(-maxX, Math.min(maxX, px)),
      Math.max(-maxY, Math.min(maxY, py)),
    ]
  }, [])

  const resetZoom = useCallback(() => { setZoom(1); setPanX(0); setPanY(0) }, [])

  const changeZoom = useCallback((factor) => {
    setZoom(prevZ => {
      const newZ = Math.max(1, Math.min(8, prevZ * factor))
      setPanX(px => {
        setPanY(py => {
          const [cx, cy] = clampPan(newZ, px, py)
          setPanX(cx); setPanY(cy); return cy
        })
        return px
      })
      if (newZ === 1) { setPanX(0); setPanY(0) }
      return newZ
    })
  }, [clampPan])

  const handleVideoWheel = useCallback((e) => {
    e.preventDefault()
    const rect   = videoWrapRef.current.getBoundingClientRect()
    const cx     = e.clientX - rect.left - rect.width  / 2
    const cy     = e.clientY - rect.top  - rect.height / 2
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
    setZoom(prevZ => {
      const newZ = Math.max(1, Math.min(8, prevZ * factor))
      if (newZ === 1) { setPanX(0); setPanY(0); return 1 }
      setPanX(px => {
        const npx  = px - cx * (1 / newZ - 1 / prevZ)
        const width = videoWrapRef.current?.clientWidth
        const maxX = width == null ? 9999 : width * (newZ - 1) / (2 * newZ)
        return Math.max(-maxX, Math.min(maxX, npx))
      })
      setPanY(py => {
        const npy  = py - cy * (1 / newZ - 1 / prevZ)
        const height = videoWrapRef.current?.clientHeight
        const maxY = height == null ? 9999 : height * (newZ - 1) / (2 * newZ)
        return Math.max(-maxY, Math.min(maxY, npy))
      })
      return newZ
    })
  }, [])

  const handleVideoPanStart = useCallback((e) => {
    if (zoom <= 1) return
    isVideoPan.current = true
    e.preventDefault()
  }, [zoom])

  useEffect(() => {
    const onMove = (e) => {
      if (!isVideoPan.current) return
      setPanX(px => {
        const newPx = px + e.movementX / zoom
        const width = videoWrapRef.current?.clientWidth
        const maxX = width == null ? 9999 : width * (zoom - 1) / (2 * zoom)
        return Math.max(-maxX, Math.min(maxX, newPx))
      })
      setPanY(py => {
        const newPy = py + e.movementY / zoom
        const height = videoWrapRef.current?.clientHeight
        const maxY = height == null ? 9999 : height * (zoom - 1) / (2 * zoom)
        return Math.max(-maxY, Math.min(maxY, newPy))
      })
    }
    const onUp = () => { isVideoPan.current = false }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
  }, [zoom])

  useEffect(() => {
    const el = videoWrapRef.current
    if (!el) return
    el.addEventListener('wheel', handleVideoWheel, { passive: false })
    return () => el.removeEventListener('wheel', handleVideoWheel)
  }, [handleVideoWheel])

  const startSidebarResize = useCallback((event) => {
    event.preventDefault()
    const startX = event.clientX
    const startWidth = sidebarWidth
    const onMove = moveEvent => {
      const nextWidth = Math.max(250, Math.min(460, startWidth + moveEvent.clientX - startX))
      setSidebarWidth(nextWidth)
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [sidebarWidth])

  const startVideoResize = useCallback((event) => {
    const videoSide = videoSideRef.current
    const content = videoSide?.parentElement
    if (!videoSide || !content) return
    event.preventDefault()
    const startX = event.clientX
    const startWidth = videoSide.getBoundingClientRect().width
    const contentWidth = content.getBoundingClientRect().width
    const onMove = moveEvent => {
      const maxWidth = Math.max(320, contentWidth - 420)
      const nextWidth = Math.max(280, Math.min(maxWidth, startWidth + moveEvent.clientX - startX))
      setVideoPanelWidth(nextWidth)
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [])

  const resizeSidebarWithKeyboard = useCallback((event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    setSidebarWidth(current => {
      if (event.key === 'Home') return 250
      if (event.key === 'End') return 460
      return Math.max(250, Math.min(460, current + (event.key === 'ArrowRight' ? 16 : -16)))
    })
  }, [])

  const resizeVideoWithKeyboard = useCallback((event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    const videoSide = videoSideRef.current
    const content = videoSide?.parentElement
    if (!videoSide || !content) return
    event.preventDefault()
    const current = videoSide.getBoundingClientRect().width
    const maxWidth = Math.max(320, content.getBoundingClientRect().width - 420)
    const next = event.key === 'Home'
      ? 280
      : event.key === 'End'
        ? maxWidth
        : current + (event.key === 'ArrowRight' ? 20 : -20)
    setVideoPanelWidth(Math.max(280, Math.min(maxWidth, next)))
  }, [])

  const toggleVideoPanel = useCallback(() => {
    if (videoPanelOpen && videoRef.current) videoRef.current.pause()
    setVideoPanelOpen(open => !open)
  }, [videoPanelOpen])

  // ── Video loader ──────────────────────────────────────────────────────────
  const loadVideo = useCallback((file) => {
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    const url = URL.createObjectURL(file)
    videoUrlRef.current = url
    setVideoUrl(url)
    setVideoName(file.name)
    setVideoPanelOpen(true)
    setCurrentTime(0)
    if (typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches) {
      setMobileTab('video')
    }
  }, [])

  // ── Parquet loader ────────────────────────────────────────────────────────
  const loadParquetFile = useCallback(async (file) => {
    const previousSelectedCols = columnSelectionInitializedRef.current
      ? [...selectedColsRef.current]
      : null
    setStatus({ text: `Читаю ${file.name}…`, type: 'loading' })
    setSessionLabel(file.name)
    setSessionProtocolName('')
    setSessionDeviceId(null)
    setChartReady(false)
    setColumnsPanelOpen(false)
    plotInitRef.current = false
    if (chartDivRef.current) {
      if (chartNativeClickRef.current) {
        chartDivRef.current.removeEventListener('click', chartNativeClickRef.current, true)
        chartNativeClickRef.current = null
      }
      Plotly.purge(chartDivRef.current)
    }
    setLeftContacts([])
    setRightContacts([])
    setShowLeftPatterns(true)
    setShowRightPatterns(true)
    setShowSensor1(true)
    setShowSensor2(true)
    setShowSpeedTracker(false)
    setSpeedPredict(null)
    setShowSpeedPredict(false)
    setShowDistancePredict(false)
    calculatorDataVersionRef.current += 1
    setCalculatorResults({})
    setActiveCalculators([])
    setCalculatorLoading('')
    setSelectedCalculatorContact(null)
    setExtraCalculatorsOpen(false)
    setProtocolDetectorsOpen(false)
    setOffsetST(0)
    setShowGaps(false)
    setCheckHzData(null)
    setSelectedMarkup(null)
    anglesUnwrappedRef.current = false
    setAnglesUnwrapped(false)
    mirrorLeftRef.current = false
    setMirrorLeft(false)
    resetYawDrift()
    subplotRangesRef.current = {}
    importedCsvTextRef.current = ''
    imuOriginalRef.current = null
    setImuApplied(false)
    setPendingImportFilename('')
    setActiveMarkupFileId('')
    setSessionRecordAvailable(false)
    setLoadedSessionId(null)

    try {
      const arrayBuffer = await file.arrayBuffer()
      const rows = await parquetReadObjects({ file: arrayBuffer })

      if (!rows?.length) { setStatus({ text: 'Файл пустой', type: 'error' }); return }

      const colMap = rowsToColMap(rows)
      const tCol = detectTimeCol(Object.keys(colMap))
      addDerivedSessionColumns(colMap, tCol, null)
      setParquetData(colMap)
      setTimeCol(tCol)
      const drift = prepareYawDrift(colMap)

      const names = sortSensorNames(colMap)
      const insole = names.filter(n => n !== SPEED_TRACKER)
      const hasST = names.includes(SPEED_TRACKER)
      setSensorNames(names)

      const numCols = computeNumericColumns(colMap, tCol)
      setColumns(numCols)
      const nextSelectedCols = previousSelectedCols === null
        ? buildDefaultCols(numCols, hasST, colMap, insole)
        : previousSelectedCols.filter(col => numCols.includes(col))
      columnSelectionInitializedRef.current = true
      selectedColsRef.current = nextSelectedCols
      setSelectedCols(nextSelectedCols)
      setShowSpeedTracker(hasST)
      setOffsetST(hasST ? computeAutoOffsetST(colMap, tCol, insole) : 0)

      const tVals   = (colMap[tCol] || []).map(safeNum).filter(v => v !== null)
      const tMax    = tVals.length ? arrayMax(tVals) : 0
      const autoUnit = tMax > 3600 ? 'ms' : 's'
      setTimeUnit(autoUnit)
      timeUnitRef.current = autoUnit

      const gapStats = computeGapStats(colMap, tCol)
      const gapCount = Object.values(gapStats)
        .reduce((count, sensor) => count + sensor.gaps.length, 0)
      setCheckHzData(gapStats)
      setShowGaps(gapCount > 0)

      const stHint = hasST ? ' · SpeedTracker' : ''
      const gapHint = gapCount ? ` · ${gapCount} пропуск(ов)` : ' · без пропусков'
      const yawHint = drift.applied
        ? ` · дрейф ${drift.differentialDegS.toFixed(1)} °/с`
        : ' · дрейф не найден'
      setStatus({
        text: `✓ ${rows.length} строк · ${numCols.length} колонок · ${autoUnit}${stHint}${gapHint}${yawHint}`,
        type: 'ok',
      })
      if (typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches) {
        setMobileTab('chart')
      }

      const sid = sessionId.trim()
      if (sid) {
        try {
          const sess = await fetchSessionMarkupsFromDb(sid)
          setSessionRecordAvailable(true)
          setSessionProtocolName(sess.protocol_name || sess.protocolName || '')
          setSessionDeviceId(hasSessionMetaValue(sess.device_id ?? sess.deviceId) ? (sess.device_id ?? sess.deviceId) : null)
          setSessionMemberName(normalizeMemberName(sess.member_name ?? sess.memberName))
          setSessionTitle(normalizeSessionTitle(sess.session_title ?? sess.sessionTitle))
          setLoadedSessionId(String(sid))
          // The parquet file carries no calibration; if the linked session does,
          // derive the normalized channels now and refresh the column list. The
          // weighted total was built from raw counts above, so it is rebuilt too.
          if (addNormalizedSensorColumns(colMap, sess?.additional_info).length) {
            delete colMap[INSOLE_TOTAL_COL]
            delete colMap[SENSOR_SUM_NORM_COL]
            addWeightedInsoleTotalColumn(colMap, tCol, sess?.additional_info)
            addSensorSumColumns(colMap)
            setParquetData({ ...colMap })
            setColumns(computeNumericColumns(colMap, tCol))
          }
          setSessionLabel(`Сессия #${sid} · ${file.name}`)
        } catch {
          setSessionRecordAvailable(false)
          // parquet loaded; markups will load on save
        }
      }
    } catch (err) {
      setStatus({ text: `Ошибка чтения parquet: ${err.message}`, type: 'error' })
    }
  }, [sessionId, fetchSessionMarkupsFromDb, prepareYawDrift, resetYawDrift])

  const handleFiles = useCallback((files) => {
    ;[...files].forEach(f => {
      if (f.type.startsWith('video/') || /\.(mp4|webm|mov|avi|mkv)$/i.test(f.name)) loadVideo(f)
      else if (/\.parquet$/i.test(f.name)) loadParquetFile(f)
      else if (/\.csv$/i.test(f.name)) importLabeledCsv(f)
    })
  }, [loadVideo, loadParquetFile, importLabeledCsv])

  // ── Speed/Distance predict (charts/sprint) ────────────────────────────────
  // Both overlays read the same fetched series (it carries speed AND
  // distance per point) — whichever button is clicked first fetches, the
  // other reuses the cached result. Visibility is toggled independently.
  const ensurePredictSeries = useCallback(async () => {
    if (speedPredict) return speedPredict

    const sid = (loadedSessionId || sessionId).trim()
    if (!sid) throw new Error('Укажите ID сессии — прогноз берётся по сессии')

    setPredictLoading(true)
    try {
      const resp = await fetch(`${MARKUP_API}/sessions/${sid}/charts/sprint`, {
        headers: { 'accept': 'application/json', 'Authorization': `Bearer ${token}` },
      })
      if (resp.status === 401) {
        setToken('')
        sessionStorage.removeItem('auth_token')
        throw new Error('Сессия авторизации истекла — войдите снова')
      }
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }
      const data = normaliseSpeedPrediction(await resp.json())
      if (!data.data_points.length) {
        throw new Error('В charts/sprint нет точек скорости для этой сессии')
      }
      setSpeedPredict(data)
      return data
    } finally {
      setPredictLoading(false)
    }
  }, [speedPredict, loadedSessionId, sessionId, token])

  const fetchSpeedPredict = useCallback(async () => {
    if (showSpeedPredict) {
      setShowSpeedPredict(false)
      if (!columns.some(column => SPEED_PRED_COLS.has(column))) {
        setSelectedCols(prev => prev.filter(column => !SPEED_PRED_COLS.has(column)))
      }
      return
    }
    try {
      const data = await ensurePredictSeries()
      // Make sure a speed subplot exists to overlay onto.
      const speedCol = columns.find(c => SPEED_PRED_COLS.has(c)) || 'Speed'
      setSelectedCols(prev => prev.includes(speedCol) ? prev : [...prev, speedCol])
      setShowSpeedPredict(true)
      const peak = data.stat?.peak_speed
      setStatus({
        text: `✓ speed predict: ${data.data_points.length} точек${peak != null ? ` · пик ${peak.toFixed(2)} m/s` : ''}`,
        type: 'ok',
        area: 'models',
      })
    } catch (err) {
      setStatus({ text: `Ошибка speed predict: ${err.message}`, type: 'error', area: 'models' })
    }
  }, [showSpeedPredict, ensurePredictSeries, columns])

  const fetchDistancePredict = useCallback(async () => {
    if (showDistancePredict) {
      setShowDistancePredict(false)
      if (!columns.some(column => DISTANCE_PRED_COLS.has(column))) {
        setSelectedCols(prev => prev.filter(column => !DISTANCE_PRED_COLS.has(column)))
      }
      return
    }
    try {
      const data = await ensurePredictSeries()
      // Make sure a distance subplot exists to overlay onto.
      const distCol = columns.find(c => DISTANCE_PRED_COLS.has(c)) || 'Distance'
      setSelectedCols(prev => prev.includes(distCol) ? prev : [...prev, distCol])
      setShowDistancePredict(true)
      const dist = data.stat?.distance_at_peak_speed
      setStatus({
        text: `✓ distance predict: ${data.data_points.length} точек${dist != null ? ` · на пике скорости ${dist.toFixed(1)} м` : ''}`,
        type: 'ok',
        area: 'models',
      })
    } catch (err) {
      setStatus({ text: `Ошибка distance predict: ${err.message}`, type: 'error', area: 'models' })
    }
  }, [showDistancePredict, ensurePredictSeries, columns])

  /**
   * What the chart plots: the raw columns, or the same columns with XData
   * swapped for the drift-corrected yaw. Swapping a cached array reference is
   * all the toggle costs — the estimate already ran at load time.
   */
  const chartData = useMemo(() => (
    parquetData && yawFixed && correctedXDataCol
      ? { ...parquetData, XData: correctedXDataCol }
      : parquetData
  ), [parquetData, yawFixed, correctedXDataCol])

  const toggleAdditionalCalculator = useCallback(async (calculatorId, options = {}) => {
    const force = Boolean(options.force)
    const requestedDetectionFoot = PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)
      ? (options.detectionFoot || turnDetectionFeet[calculatorId] || 'both')
      : 'both'

    if (activeCalculators.includes(calculatorId) && !force) {
      setActiveCalculators(prev => prev.filter(id => id !== calculatorId))
      return
    }

    const cachedResult = calculatorResults[calculatorId]
    const cacheHasSprintSteps = calculatorId !== 'protocol-sprint-detector'
      || cachedResult?.summary?.step_count != null
    const isJumpDetector = ['jump-metrics', 'protocol-jumping-detector'].includes(calculatorId)
    const cacheHasJumpHeights = !isJumpDetector
      || cachedResult?.contacts?.every(contact => contact.jump_height_cm != null)
    const cacheMatchesDetectionFoot = !PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)
      || (cachedResult?.summary?.detection_foot || 'both') === requestedDetectionFoot
    const cacheHasSeparateFootOverlay = !PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)
      || requestedDetectionFoot !== 'both'
      || (cachedResult?.summary?.left_turn_count != null && cachedResult?.summary?.right_turn_count != null)
    if (!force && cachedResult && cacheHasSprintSteps && cacheHasJumpHeights && cacheMatchesDetectionFoot && cacheHasSeparateFootOverlay) {
      setActiveCalculators(prev => prev.includes(calculatorId) ? prev : [...prev, calculatorId])
      return
    }

    if (!parquetData || calculatorLoading) return

    const parsedWeight = Number(weightKg)
    if (calculatorId === 'force-jump' && (!Number.isFinite(parsedWeight) || parsedWeight <= 0)) {
      setStatus({ text: 'Укажите положительный вес для Bilateral GRF', type: 'error', area: 'models' })
      return
    }

    const dataVersion = calculatorDataVersionRef.current
    setCalculatorLoading(calculatorId)
    try {
      const canUseSessionId = loadedSessionId
        && sessionRecordAvailable
        && !yawFixed
        && !imuApplied
        && !importedCsvTextRef.current

      const payload = canUseSessionId
        ? { session_id: Number(loadedSessionId) }
        : { columns: getCalculatorColumns(calculatorId, chartData) }

      if (calculatorId === 'force-jump') payload.weight_kg = parsedWeight
      if (PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)) {
        const selectedSensorName = requestedDetectionFoot === 'both'
          ? ''
          : sensorNameForFoot(insoleSensorNames, requestedDetectionFoot)
        if (requestedDetectionFoot !== 'both' && !selectedSensorName) {
          throw new Error(`В данных нет ${requestedDetectionFoot === 'left' ? 'левой' : 'правой'} ноги`)
        }
        payload.detection_foot = requestedDetectionFoot
        if (selectedSensorName) payload.sensor_name = selectedSensorName
      }
      const resp = await fetch(`/calculator-api/calculate/${calculatorId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }

      const data = await resp.json()
      if (dataVersion !== calculatorDataVersionRef.current) return

      setCalculatorResults(prev => ({ ...prev, [calculatorId]: data }))
      setActiveCalculators(prev => prev.includes(calculatorId) ? prev : [...prev, calculatorId])
      setSelectedCalculatorContact(prev => prev?.calculatorId === calculatorId ? null : prev)

      const left = data.summary?.left?.contact_count || 0
      const right = data.summary?.right?.contact_count || 0
      const cadence = data.summary?.cadence_spm
      const resultText = PROTOCOL_DETECTOR_BY_ID[calculatorId]
        ? protocolDetectorSummary(data)
        : calculatorId === 'jump-metrics'
          ? `${data.summary?.total_jump_count || 0} прыж. · высота ${formatMetric(data.summary?.mean_jump_height_cm, 1, ' см')}`
          : calculatorId === 'force-jump'
            ? `пик ${formatMetric(data.summary?.peak_force_n, 1, ' Н')} · ${formatMetric(data.summary?.peak_force_bw, 2, ' BW')}`
            : `L ${left} · R ${right}${cadence != null ? ` · ${cadence.toFixed(0)} spm` : ''}`
      setStatus({
        text: `✓ ${data.label}: ${resultText}`,
        type: 'ok',
        area: 'models',
      })
    } catch (err) {
      if (dataVersion !== calculatorDataVersionRef.current) return
      const localHint = err instanceof TypeError
        ? 'Локальный API калькуляторов недоступен — запустите npm run calculator-api'
        : err.message
      setStatus({ text: `Ошибка калькулятора: ${localHint}`, type: 'error', area: 'models' })
    } finally {
      if (dataVersion === calculatorDataVersionRef.current) setCalculatorLoading('')
    }
  }, [activeCalculators, calculatorResults, parquetData, calculatorLoading, weightKg, turnDetectionFeet, insoleSensorNames, loadedSessionId, sessionRecordAvailable, yawFixed, imuApplied, chartData])

  // ── Build Plotly chart ────────────────────────────────────────────────────
  const renderChart = useCallback(() => {
    if (!chartData || !selectedCols.length || !chartDivRef.current) return

    const nameArr = chartData['Name']

    const filterBySensors = (sensorNames) => {
      if (!nameArr) return chartData
      if (!sensorNames.length) return null
      const sensorSet = new Set(sensorNames)
      const mask = nameArr.map(v => sensorSet.has(v))
      const out  = {}
      Object.entries(chartData).forEach(([k, arr]) => {
        out[k] = arr.filter((_, i) => mask[i])
      })
      return out
    }

    const applyUnwrap = (d) => {
      if (!anglesUnwrappedRef.current || !d) return d
      const out = { ...d }
      selectedCols.forEach((col) => {
        if (UNWRAPPABLE_ANGLE_COLUMNS.has(col) && out[col]) {
          out[col] = unwrapAngleDegrees(out[col])
        }
      })
      return out
    }
    // Only the left foot is flipped: the point is to bring it into the right
    // foot's frame, so the right one stays as recorded.
    const applyMirror = (d) => {
      if (!mirrorLeftRef.current || !d) return d
      const out = { ...d }
      selectedCols.forEach((col) => {
        if (MIRRORED_LEFT_COLUMNS.has(col) && out[col]) {
          out[col] = out[col].map((v) => {
            const n = safeNum(v)
            return n === null ? v : -n
          })
        }
      })
      return out
    }
    const data1 = applyMirror(applyUnwrap(filterBySensors(sensorGroups.left))) || {}
    const data2 = sensorGroups.right.length
      ? applyUnwrap(filterBySensors(sensorGroups.right))
      : null
    const dataST = hasSpeedTracker ? filterBySensors([SPEED_TRACKER]) : null

    const shift1 = offsetS1Ref.current
    const shift2 = offsetS2Ref.current
    const shiftST = offsetSTRef.current
    const buildSeries = (data, col, shift) => {
      if (!data) return { x: [], y: [] }
      const times = data[timeCol] || []
      const values = data[col] || []
      const x = []
      const y = []
      for (let index = 0; index < Math.min(times.length, values.length); index++) {
        const t = safeNum(times[index])
        const value = safeNum(values[index])
        if (t === null || value === null) continue
        x.push(t + shift)
        y.push(value)
      }
      return { x, y }
    }
    const s1Series = Object.fromEntries(selectedCols.map(col => [col, buildSeries(data1, col, shift1)]))
    const s2Series = Object.fromEntries(selectedCols.map(col => [col, buildSeries(data2, col, shift2)]))
    const stSeries = Object.fromEntries(selectedCols.map(col => {
      const stCol = dataST ? resolveStDataCol(dataST, col) : col
      return [col, buildSeries(dataST, stCol, shiftST)]
    }))

    const allTVals = selectedCols.flatMap(col => [
      ...(ST_ONLY_COLS.has(col) || !showSensor1 ? [] : s1Series[col].x),
      ...(ST_ONLY_COLS.has(col) || !showSensor2 ? [] : s2Series[col].x),
      ...(showSpeedTrackerRef.current ? stSeries[col].x : []),
    ])
    if (!allTVals.length) {
      setStatus({ text: `Колонка "${timeCol}" пустая`, type: 'error' })
      return
    }
    const xMin = arrayMin(allTVals)
    const xMax = arrayMax(allTVals)

    const n    = selectedCols.length
    const gap  = 0.03
    const subH = (1 - gap * (n - 1)) / n

    const yRanges = {}
    selectedCols.forEach(col => {
      const stOnly = ST_ONLY_COLS.has(col)
      const vals1 = (stOnly || !showSensor1) ? [] : s1Series[col].y
      const vals2 = (stOnly || !showSensor2) ? [] : s2Series[col].y
      let vals  = [...vals1, ...vals2]
      if (dataST && showSpeedTrackerRef.current) {
        vals = [...vals, ...stSeries[col].y]
      }
      if (showSpeedPredict && SPEED_PRED_COLS.has(col) && speedPredict?.data_points?.length) {
        vals = [...vals, ...speedPredict.data_points.map(point => safeNum(point.speed)).filter(value => value !== null)]
      }
      if (showDistancePredict && DISTANCE_PRED_COLS.has(col) && speedPredict?.data_points?.length) {
        vals = [...vals, ...speedPredict.data_points.map(point => safeNum(point.distance)).filter(value => value !== null)]
      }
      if (!vals.length) { yRanges[col] = [-1, 1]; return }
      const mn = arrayMin(vals), mx = arrayMax(vals)
      const p  = Math.max((mx - mn) * 0.08, 0.1)
      yRanges[col] = [mn - p, mx + p]
    })

    const traces = []
    const s1Idx  = []
    const s2Idx  = []
    const stIdx  = []
    selectedCols.forEach((col, i) => {
      const yAxis = i === 0 ? 'y' : `y${i + 1}`
      const xAxis = `x${i === 0 ? '' : i + 1}`
      const stOnly = ST_ONLY_COLS.has(col)

      if (!stOnly) {
        if (s1Series[col].y.length) {
          s1Idx.push(traces.length)
          traces.push({
            x: s1Series[col].x,
            y: s1Series[col].y,
            name: data2 ? `${col} (S1)` : col,
            type: 'scatter', mode: 'lines',
            xaxis: xAxis, yaxis: yAxis,
            line: { color: PALETTE[(2 * i) % PALETTE.length], width: 1.5 },
            connectgaps: false,
            visible: showSensor1,
            hovertemplate: TRACE_HOVER_TEMPLATE,
          })
        }
        if (s2Series[col].y.length) {
          s2Idx.push(traces.length)
          traces.push({
            x: s2Series[col].x,
            y: s2Series[col].y,
            name: `${col} (S2)`,
            type: 'scatter', mode: 'lines',
            xaxis: xAxis, yaxis: yAxis,
            line: { color: PALETTE[(2 * i + 1) % PALETTE.length], width: 1.5 },
            connectgaps: false,
            visible: showSensor2,
            hovertemplate: TRACE_HOVER_TEMPLATE,
          })
        }
      }

      if (dataST) {
        const stCol = resolveStDataCol(dataST, col)
        const seriesST = stSeries[col]
        if (seriesST.y.length) {
          stIdx.push(traces.length)
          traces.push({
            x: seriesST.x,
            y: seriesST.y,
            name: `${col} (ST)`,
            type: 'scatter', mode: 'lines',
            xaxis: xAxis, yaxis: yAxis,
            line: { color: ST_COL_COLORS[stCol] ?? ST_COLOR, width: stOnly ? 2 : 1.5, dash: stOnly ? 'solid' : 'dot' },
            connectgaps: false,
            visible: showSpeedTrackerRef.current,
            hovertemplate: TRACE_HOVER_TEMPLATE,
          })
        }
      }

      // Speed-predict overlay (charts/sprint) on top of the speed subplot.
      // Backend time is seconds; convert it to the raw Time unit used by the chart.
      if (SPEED_PRED_COLS.has(col) && showSpeedPredict && speedPredict?.data_points?.length) {
        const shiftST = offsetSTRef.current
        const predictTimeScale = timeUnitRef.current === 'ms' ? 1000 : 1
        const toX = (tSec) => tSec * predictTimeScale + shiftST
        traces.push({
          x: speedPredict.data_points.map(p => toX(p.time)),
          y: speedPredict.data_points.map(p => p.speed),
          name: 'speed predict',
          type: 'scatter', mode: 'lines',
          xaxis: xAxis, yaxis: yAxis,
          line: { color: PRED_COLOR, width: 2 },
          connectgaps: false,
          hovertemplate: TRACE_HOVER_TEMPLATE,
        })
        const st = speedPredict.stat
        if (st && st.peak_speed != null && st.timestep_at_peak_speed != null) {
          traces.push({
            x: [toX(st.timestep_at_peak_speed)],
            y: [st.peak_speed],
            name: `пик ${st.peak_speed.toFixed(2)} m/s`,
            type: 'scatter', mode: 'markers',
            xaxis: xAxis, yaxis: yAxis,
            marker: { color: PRED_COLOR, size: 11, symbol: 'star', line: { color: '#fff', width: 1 } },
            hovertemplate: TRACE_HOVER_TEMPLATE,
          })
        }
      }

      // Distance-predict overlay (same fetched series, cumulative distance).
      if (DISTANCE_PRED_COLS.has(col) && showDistancePredict && speedPredict?.data_points?.length) {
        const shiftST = offsetSTRef.current
        const predictTimeScale = timeUnitRef.current === 'ms' ? 1000 : 1
        const toX = (tSec) => tSec * predictTimeScale + shiftST
        traces.push({
          x: speedPredict.data_points.map(p => toX(p.time)),
          y: speedPredict.data_points.map(p => p.distance),
          name: 'distance predict',
          type: 'scatter', mode: 'lines',
          xaxis: xAxis, yaxis: yAxis,
          line: { color: PRED_COLOR, width: 2 },
          connectgaps: false,
          hovertemplate: TRACE_HOVER_TEMPLATE,
        })
        const st = speedPredict.stat
        if (st && st.distance_at_peak_speed != null && st.timestep_at_peak_speed != null) {
          traces.push({
            x: [toX(st.timestep_at_peak_speed)],
            y: [st.distance_at_peak_speed],
            name: `${st.distance_at_peak_speed.toFixed(1)} м на пике скорости`,
            type: 'scatter', mode: 'markers',
            xaxis: xAxis, yaxis: yAxis,
            marker: { color: PRED_COLOR, size: 11, symbol: 'star', line: { color: '#fff', width: 1 } },
            hovertemplate: TRACE_HOVER_TEMPLATE,
          })
        }
      }

    })
    s1TraceIdxRef.current = s1Idx
    s2TraceIdxRef.current = s2Idx
    stTraceIdxRef.current = stIdx

    cursorShapesRef.current  = buildCursorShapes(xMin, n)
    contactShapesRef.current = []
    gapShapesRef.current     = []
    lastTRef.current         = null
    plotInitRef.current      = false

    const layout = {
      shapes: cursorShapesRef.current,
      xaxis: {},
      margin: { t: 12, l: 60, r: 16, b: 42 },
      plot_bgcolor: '#f8f9fa',
      paper_bgcolor: '#fff',
      font: { family: UI_FONT_FAMILY, color: '#334155', size: 11 },
      showlegend: true,
      dragmode: 'pan',
      hovermode: 'closest',
      hoverlabel: {
        bgcolor: '#ffffff',
        bordercolor: '#94a3b8',
        font: { family: UI_FONT_FAMILY, color: '#111827', size: 12 },
        namelength: -1,
      },
      legend: { orientation: 'h', y: -0.06, font: { family: UI_FONT_FAMILY, size: 11 } },
    }

    const shareX = chartsLockedRef.current && n > 1
    const sharedXRange = shareX
      ? (subplotRangesRef.current[selectedCols[0]]?.x || [xMin, xMax])
      : null

    selectedCols.forEach((col, i) => {
      const top    = 1 - i * (subH + gap)
      const bottom = top - subH
      const yKey   = i === 0 ? 'yaxis'  : `yaxis${i + 1}`
      const xKey   = i === 0 ? 'xaxis'  : `xaxis${i + 1}`

      layout[yKey] = {
        domain:    [Math.max(0, bottom), Math.min(1, top)],
        title:     { text: col, font: { size: 11 } },
        range:     subplotRangesRef.current[col]?.y || yRanges[col],
        showgrid:  true,
        gridcolor: '#e8e8e8',
        zeroline:  false,
        tickfont:  { size: 10 },
      }
      layout[xKey] = {
        anchor:         `y${i === 0 ? '' : i + 1}`,
        showgrid:        true,
        gridcolor:       '#e8e8e8',
        title:           i === n - 1 ? { text: 'Время', font: { size: 11 } } : undefined,
        tickfont:        { size: 10 },
        showticklabels:  i === n - 1,
        range:           sharedXRange || subplotRangesRef.current[col]?.x || [xMin, xMax],
      }
      if (shareX && i > 0) layout[xKey].matches = 'x'
    })

    Plotly.newPlot(chartDivRef.current, traces, layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      scrollZoom: true,
    }).then(() => {
      plotInitRef.current = true
      setChartReady(true)
      updateOverlayShapes()

      const findCalculatorContact = (x) => {
        const timeScale = timeUnitRef.current === 'ms' ? 1000 : 1
        const calculatorIds = [...new Set(['step-cadence', ...activeCalculatorsRef.current])]
        const candidates = []

        calculatorIds.forEach(calculatorId => {
          const result = calculatorResultsRef.current[calculatorId]
          if (!result?.contacts?.length) return
          result.contacts.forEach((contact, index) => {
            const shift = contact.foot === 'right'
              ? offsetS2Ref.current
              : contact.foot === 'left'
                ? offsetS1Ref.current
                : 0
            const start = Number(contact.start_time_s) * timeScale + shift
            const end = Number(contact.end_time_s) * timeScale + shift
            if (!Number.isFinite(start) || !Number.isFinite(end)) return
            const x0 = Math.min(start, end)
            const x1 = Math.max(start, end)
            if (x >= x0 && x <= x1) {
              candidates.push({ calculatorId, index, contact })
            }
          })
        })

        candidates.sort((a, b) => {
          if (a.calculatorId === 'step-cadence' && b.calculatorId !== 'step-cadence') return -1
          if (b.calculatorId === 'step-cadence' && a.calculatorId !== 'step-cadence') return 1
          return Number(a.contact.duration_ms || 0) - Number(b.contact.duration_ms || 0)
        })
        return candidates[0] || null
      }

      const selectCalculatorContactAtX = (x) => {
        const selectedContact = findCalculatorContact(Number(x))
        if (!selectedContact) return false
        setSelectedCalculatorContact(selectedContact)
        return true
      }

      chartDivRef.current.on('plotly_click', (d) => {
        if (!d?.points?.length) return
        const t = d.points[0].x
        if (relabelStepRef.current === 'start') {
          const sm = selectedMarkupRef.current
          if (sm) {
            const setter = sm.foot === 'left' ? setLeftContacts : setRightContacts
            setter(prev => {
              const next = [...prev]
              if (sm.index < next.length) next[sm.index] = t
              return next
            })
            setRelabelStep('end')
          }
        } else if (relabelStepRef.current === 'end') {
          const sm = selectedMarkupRef.current
          if (sm) {
            const setter = sm.foot === 'left' ? setLeftContacts : setRightContacts
            setter(prev => {
              const next = [...prev]
              if (sm.index + 1 < next.length) next[sm.index + 1] = t
              return next
            })
            setRelabelStep(null)
          }
        } else if (labelingRef.current) {
          if (currentFootRef.current === 'left') setLeftContacts(p => [...p, t])
          else setRightContacts(p => [...p, t])
        } else {
          if (!selectCalculatorContactAtX(t) && videoRef.current) {
            const scale = timeUnitRef.current === 'ms' ? 1000 : 1
            videoRef.current.currentTime = Math.max(0, t / scale)
          }
        }
      })

      if (chartNativeClickRef.current) {
        chartDivRef.current.removeEventListener('click', chartNativeClickRef.current, true)
      }
      const nativeChartClick = (event) => {
        if (event.target?.closest?.('.modebar')) return
        if (relabelStepRef.current || labelingRef.current) return

        const fullLayout = chartDivRef.current?._fullLayout
        const rect = chartDivRef.current?.getBoundingClientRect()
        if (!rect || !fullLayout) return

        const chartY = event.clientY - rect.top
        const subplotIndex = selectedCols.findIndex((_, index) => {
          const yAxisKey = index === 0 ? 'yaxis' : `yaxis${index + 1}`
          const yAxis = fullLayout[yAxisKey]
          const axisOffset = Number(yAxis?._offset)
          const axisLength = Number(yAxis?._length)
          return Number.isFinite(axisOffset)
            && Number.isFinite(axisLength)
            && chartY >= axisOffset
            && chartY <= axisOffset + axisLength
        })
        const xAxisKey = subplotIndex > 0 ? `xaxis${subplotIndex + 1}` : 'xaxis'
        const xAxis = fullLayout[xAxisKey]
        const axisOffset = Number(xAxis?._offset)
        const axisLength = Number(xAxis?._length)
        const range = xAxis?.range
        if (!xAxis || !Number.isFinite(axisOffset) || !Number.isFinite(axisLength) || axisLength <= 0 || !Array.isArray(range) || range.length < 2) return

        const axisPixel = event.clientX - rect.left - axisOffset
        if (axisPixel < 0 || axisPixel > axisLength) return

        const x = typeof xAxis.p2l === 'function'
          ? xAxis.p2l(axisPixel)
          : Number(range[0]) + (axisPixel / axisLength) * (Number(range[1]) - Number(range[0]))
        if (selectCalculatorContactAtX(x)) {
          event.preventDefault()
          event.stopPropagation()
        }
      }
      chartNativeClickRef.current = nativeChartClick
      chartDivRef.current.addEventListener('click', nativeChartClick, true)

      chartDivRef.current.on('plotly_relayout', (eventData) => {
        selectedCols.forEach((col, index) => {
          const xAxisKey = plotAxisKey(index, 'xaxis')
          const yAxisKey = plotAxisKey(index, 'yaxis')
          const current = subplotRangesRef.current[col] || {}
          const next = { ...current }
          let changed = false

          if (eventData[`${xAxisKey}.range[0]`] !== undefined && eventData[`${xAxisKey}.range[1]`] !== undefined) {
            next.x = [eventData[`${xAxisKey}.range[0]`], eventData[`${xAxisKey}.range[1]`]]
            changed = true
          } else if (eventData[`${xAxisKey}.range`] !== undefined) {
            next.x = eventData[`${xAxisKey}.range`]
            changed = true
          } else if (eventData[`${xAxisKey}.autorange`] === true) {
            delete next.x
            changed = true
          }

          if (eventData[`${yAxisKey}.range[0]`] !== undefined && eventData[`${yAxisKey}.range[1]`] !== undefined) {
            next.y = [eventData[`${yAxisKey}.range[0]`], eventData[`${yAxisKey}.range[1]`]]
            changed = true
          } else if (eventData[`${yAxisKey}.range`] !== undefined) {
            next.y = eventData[`${yAxisKey}.range`]
            changed = true
          } else if (eventData[`${yAxisKey}.autorange`] === true) {
            delete next.y
            changed = true
          }

          if (changed) subplotRangesRef.current[col] = next
        })

        if (chartsLockedRef.current) {
          let sharedX = null
          selectedCols.forEach((_, index) => {
            const x = readPlotRange(eventData, plotAxisKey(index, 'xaxis'))
            if (x) sharedX = x
          })
          if (sharedX) {
            selectedCols.forEach(col => {
              const current = subplotRangesRef.current[col] || {}
              subplotRangesRef.current[col] = { ...current, x: sharedX }
            })
          }
        }
      })
    })
  }, [chartData, selectedCols, timeCol, sensorGroups, hasSpeedTracker, updateOverlayShapes, showSensor1, showSensor2, speedPredict, showSpeedPredict, showDistancePredict])

  const toggleChartsLock = useCallback(() => {
    const next = !chartsLockedRef.current
    chartsLockedRef.current = next
    setChartsLocked(next)
    const gd = chartDivRef.current
    const cols = selectedColsRef.current
    if (!plotInitRef.current || !gd || cols.length < 2) return

    const xRange = currentAxisRange(gd, 'xaxis')
    const updates = {}
    cols.forEach((col, i) => {
      const xKey = plotAxisKey(i, 'xaxis')
      if (i > 0) updates[`${xKey}.matches`] = next ? 'x' : false
      if (next && xRange) {
        updates[`${xKey}.range`] = xRange
        const current = subplotRangesRef.current[col] || {}
        subplotRangesRef.current[col] = { ...current, x: xRange }
      }
    })
    if (next && xRange) updates['xaxis.range'] = xRange
    Plotly.relayout(gd, updates)
  }, [])

  const handleToggleYawDrift = useCallback(() => {
    if (!yawDrift?.applied) return
    const next = !yawFixed
    setYawFixed(next)
    setStatus(next
      ? {
        text: `✓ Дрейф убран · расхождение стоп ${yawDrift.divergenceStdBefore.toFixed(0)}° → ${yawDrift.divergenceStdAfter.toFixed(0)}°`,
        type: 'ok',
      }
      : { text: 'Показан исходный XData', type: 'ok' })
  }, [yawDrift, yawFixed])

  /** Hover copy for the toggle: why it is off, or what the estimate found. */
  const yawDriftTitle = useMemo(() => {
    if (!yawDrift) return 'Загрузите сессию — дрейф оценивается при загрузке'
    if (!yawDrift.applied) return `Корректировать нечего: ${yawDrift.reason}`
    return [
      yawFixed ? 'Вернуть исходный XData' : 'Убрать дрейф гироскопа из XData',
      `дифференциал ${yawDrift.differentialDegS.toFixed(2)} °/с (L ${yawDrift.leftDegS.toFixed(2)} / R ${yawDrift.rightDegS.toFixed(2)})`,
      `нелинейность ${yawDrift.nonlinearityDeg.toFixed(0)}°`,
      `расхождение стоп ${yawDrift.divergenceStdBefore.toFixed(0)}° → ${yawDrift.divergenceStdAfter.toFixed(0)}°`,
      `перекрытие ${yawDrift.spanS.toFixed(0)} с`,
    ].join(' · ')
  }, [yawDrift, yawFixed])

  const handleUnwrapAngles = useCallback(() => {
    if (!parquetData || !selectedCols.length) return
    anglesUnwrappedRef.current = !anglesUnwrappedRef.current
    setAnglesUnwrapped(anglesUnwrappedRef.current)
    renderChart()
  }, [parquetData, selectedCols, renderChart])

  const mirrorableCols = useMemo(
    () => selectedCols.filter(col => MIRRORED_LEFT_COLUMNS.has(col)),
    [selectedCols],
  )

  /** Flip AcX/AcY of the left foot. Plot only — the stored data is untouched. */
  const handleMirrorLeft = useCallback(() => {
    if (!parquetData || !mirrorableCols.length) return
    mirrorLeftRef.current = !mirrorLeftRef.current
    setMirrorLeft(mirrorLeftRef.current)
    renderChart()
    setStatus(mirrorLeftRef.current
      ? { text: `✓ Левая нога отражена · ${mirrorableCols.join(', ')} × (-1)`, type: 'ok' }
      : { text: 'Показаны исходные AcX/AcY левой ноги', type: 'ok' })
  }, [parquetData, mirrorableCols, renderChart])

  const mirrorLeftTitle = useMemo(() => {
    if (!mirrorableCols.length) {
      return 'Выберите AcX или AcY — отражаются только эти каналы'
    }
    return mirrorLeft
      ? `Вернуть исходные ${mirrorableCols.join(', ')} левой ноги`
      : `Умножить ${mirrorableCols.join(', ')} левой ноги на −1 (только график)`
  }, [mirrorLeft, mirrorableCols])

  useEffect(() => {
    if (!parquetData || !selectedCols.length) {
      const timeout = window.setTimeout(() => {
        if (plotInitRef.current && chartDivRef.current) Plotly.purge(chartDivRef.current)
        plotInitRef.current = false
        setChartReady(false)
      }, 0)
      return () => window.clearTimeout(timeout)
    }
    const timeout = window.setTimeout(renderChart, 140)
    return () => window.clearTimeout(timeout)
  }, [offsetS1, offsetS2, offsetST, timeUnit, showSpeedTracker, showSensor1, showSensor2, renderChart, parquetData, selectedCols])

  useEffect(() => {
    if (!chartReady || !chartDivRef.current) return undefined
    const delay = isMobile && mobileTab === 'chart' ? 80 : 30
    const timeout = window.setTimeout(() => {
      if (chartDivRef.current) Plotly.Plots.resize(chartDivRef.current)
    }, delay)
    return () => window.clearTimeout(timeout)
  }, [chartReady, sidebarWidth, videoPanelOpen, videoPanelWidth, mobileTab, isMobile])

  // ── Video timeupdate → move chart cursor ──────────────────────────────────
  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current) return
    const t = videoRef.current.currentTime
    setCurrentTime(t)

    if (vidLblRef.current) vidLblRef.current.textContent = formatTime(t)
    const scale = timeUnitRef.current === 'ms' ? 1000 : 1
    const imuT  = t * scale
    if (imuLblRef.current) imuLblRef.current.textContent =
      `IMU ${timeUnitRef.current === 'ms' ? imuT.toFixed(0) + 'ms' : imuT.toFixed(2) + 's'}`

    if (!plotInitRef.current || !chartDivRef.current) return
    if (lastTRef.current !== null && Math.abs(imuT - lastTRef.current) < 0.04) return
    lastTRef.current = imuT

    const n = selectedColsRef.current.length
    if (n === 0) return
    cursorShapesRef.current = buildCursorShapes(imuT, n)
    Plotly.relayout(chartDivRef.current, {
      shapes: [...gapShapesRef.current, ...contactShapesRef.current, ...cursorShapesRef.current],
    })
  }, [])

  // ── Timeline drag ─────────────────────────────────────────────────────────
  const seekFromX = useCallback((clientX) => {
    const rect = timelineRef.current?.getBoundingClientRect()
    if (!rect || !videoRef.current || videoDuration <= 0) return
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    videoRef.current.currentTime = (x / rect.width) * videoDuration
  }, [videoDuration])

  useEffect(() => {
    const onMove = (e) => { if (isDragging.current) seekFromX(e.clientX) }
    const onUp   = () => { isDragging.current = false }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
  }, [seekFromX])

  // ── Cleanup ───────────────────────────────────────────────────────────────
  useEffect(() => () => {
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current)
    if (chartReorderRef.current?.previewUrl) URL.revokeObjectURL(chartReorderRef.current.previewUrl)
    if (chartDivRef.current && chartNativeClickRef.current) {
      chartDivRef.current.removeEventListener('click', chartNativeClickRef.current, true)
    }
    if (chartDivRef.current) Plotly.purge(chartDivRef.current)
  }, [])

  // ── Column toggle ─────────────────────────────────────────────────────────
  const toggleCol = (col) =>
    setSelectedCols(p => p.includes(col) ? p.filter(c => c !== col) : [...p, col])

  const moveSelectedColumn = useCallback((fromIndex, toIndex) => {
    setSelectedCols((current) => {
      if (fromIndex === toIndex
        || fromIndex < 0
        || toIndex < 0
        || fromIndex >= current.length
        || toIndex >= current.length) return current
      const next = [...current]
      const [moved] = next.splice(fromIndex, 1)
      next.splice(toIndex, 0, moved)
      return next
    })
  }, [])

  const beginChartReorder = useCallback((event, fromIndex) => {
    event.preventDefault()
    event.stopPropagation()
    try {
      event.currentTarget.setPointerCapture?.(event.pointerId)
    } catch {
      // Pointer capture can be unavailable for synthetic/assistive input.
    }

    const chartArea = chartAreaRef.current
    const chartRect = chartArea?.getBoundingClientRect()
    const sourceMetrics = chartRect
      ? chartSubplotMetrics(fromIndex, selectedCols.length, chartRect.height)
      : { top: 0, height: 1 }
    let previewUrl = ''
    const plotSvg = chartDivRef.current?.querySelector('svg.main-svg')
    if (plotSvg) {
      try {
        const clone = plotSvg.cloneNode(true)
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        const svgText = new XMLSerializer().serializeToString(clone)
        previewUrl = URL.createObjectURL(new Blob([svgText], { type: 'image/svg+xml' }))
      } catch {
        previewUrl = ''
      }
    }

    const next = {
      fromIndex,
      targetIndex: fromIndex,
      pointerId: event.pointerId,
      pointerY: event.clientY,
      pointerOffsetY: chartRect
        ? Math.max(0, Math.min(sourceMetrics.height, event.clientY - chartRect.top - sourceMetrics.top))
        : sourceMetrics.height / 2,
      areaTop: chartRect?.top || 0,
      chartWidth: chartRect?.width || 0,
      chartHeight: chartRect?.height || 0,
      sourceTop: sourceMetrics.top,
      sourceHeight: sourceMetrics.height,
      col: selectedCols[fromIndex],
      previewUrl,
    }
    chartReorderRef.current = next
    setChartReorder(next)
  }, [selectedCols])

  const updateChartReorder = useCallback((event) => {
    const current = chartReorderRef.current
    const chartArea = chartAreaRef.current
    if (!current || current.pointerId !== event.pointerId || !chartArea || !selectedCols.length) return
    event.preventDefault()
    event.stopPropagation()

    const rect = chartArea.getBoundingClientRect()
    const plotTop = rect.top + 12
    const plotHeight = Math.max(1, rect.height - 54)
    const relativeY = Math.max(0, Math.min(plotHeight - 1, event.clientY - plotTop))
    const targetIndex = Math.min(selectedCols.length - 1, Math.floor((relativeY / plotHeight) * selectedCols.length))
    const next = { ...current, targetIndex, pointerY: event.clientY }
    chartReorderRef.current = next
    setChartReorder(next)
  }, [selectedCols.length])

  const finishChartReorder = useCallback((event, cancelled = false) => {
    const current = chartReorderRef.current
    if (!current || current.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopPropagation()
    try {
      if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }
    } catch {
      // The pointer may already have been released by the browser.
    }
    chartReorderRef.current = null
    setChartReorder(null)
    if (current.previewUrl) {
      window.setTimeout(() => URL.revokeObjectURL(current.previewUrl), 0)
    }
    if (!cancelled) moveSelectedColumn(current.fromIndex, current.targetIndex)
  }, [moveSelectedColumn])

  // ── Computed ──────────────────────────────────────────────────────────────
  const cursorPct = videoDuration > 0 ? (currentTime / videoDuration) * 100 : 0

  const ticks = []
  if (videoDuration > 0) {
    const step = videoDuration <= 30 ? 5 : videoDuration <= 120 ? 15 : videoDuration <= 600 ? 60 : 300
    for (let t = 0; t <= videoDuration; t += step)
      ticks.push({ t, pct: (t / videoDuration) * 100 })
  }

  const totalContacts = leftContacts.length + rightContacts.length

  // ── Login modal ───────────────────────────────────────────────────────────
  if (!token) {
    return (
      <div className="login-backdrop">
        <div className="login-card">
          <div className="login-logo">
            <span className="login-icon">🎬</span>
            <h1 className="login-title">Видео + IMU Viewer</h1>
            <p className="login-sub">MiraiTech Health</p>
          </div>
          <form className="login-form" onSubmit={handleLogin}>
            <label className="login-label">
              Email
              <input
                type="email"
                className="login-input"
                value={loginEmail}
                onChange={e => setLoginEmail(e.target.value)}
                placeholder="admin@miraitech.health"
                required
                autoFocus
              />
            </label>
            <label className="login-label">
              Пароль
              <input
                type="password"
                className="login-input"
                value={loginPassword}
                onChange={e => setLoginPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </label>
            {loginError && <p className="login-error">{loginError}</p>}
            <button type="submit" className="login-btn" disabled={authLoading}>
              {authLoading ? 'Вход…' : 'Войти'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  // ── Main UI ───────────────────────────────────────────────────────────────
  const activeModelCount = activeCalculators.length
    + Number(showSpeedPredict)
    + Number(showDistancePredict)
  const chartReorderTargetMetrics = chartReorder
    ? chartSubplotMetrics(chartReorder.targetIndex, selectedCols.length, chartReorder.chartHeight)
    : null

  return (
    <div
      className={`app${dragOver ? ' drag-over' : ''}${isMobile ? ` mobile-tab-${mobileTab}` : ''}`}
      onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
    >
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <UiIcon name="video" className="header-icon" />
          <h1 className="header-title">Видео + IMU Viewer</h1>
        </div>
        <div className="header-right">
          {videoName && (
            <FileBadge type="video"><UiIcon name="video" /> {videoName}</FileBadge>
          )}
          {sessionRecordAvailable && loadedSessionId && (
            <SessionTitleBadge
              key={`${loadedSessionId}:${sessionTitle}`}
              title={sessionTitle}
              expanded={sessionTitleExpanded}
              onToggle={() => setSessionTitleExpanded(v => !v)}
              onSave={saveSessionTitle}
            />
          )}
          {sessionMemberName && (
            <FileBadge type="member" title={sessionMemberName}>
              <UiIcon name="user" /> {sessionMemberName}
            </FileBadge>
          )}
          {sessionLabel && (
            <FileBadge type="parquet"><UiIcon name="database" /> {sessionLabel}</FileBadge>
          )}
          {sessionProtocolName && (
            <FileBadge type="protocol">{sessionProtocolName}</FileBadge>
          )}
          {hasSessionMetaValue(sessionDeviceId) && (
            <FileBadge type="device">device {sessionDeviceId}</FileBadge>
          )}
          <button className="logout-btn" onClick={handleLogout} title="Выйти">
            <UiIcon name="logout" /> <span className="logout-text">Выйти</span>
          </button>
        </div>
      </header>

      <div className="mobile-session-strip" aria-label="Сведения о сессии">
        <span className="mobile-session-chip">
          <span className="mobile-session-key">Протокол</span>
          <span className="mobile-session-val">{sessionProtocolName || '—'}</span>
        </span>
        <span className="mobile-session-chip mobile-session-device">
          <span className="mobile-session-key">Device</span>
          <span className="mobile-session-val">
            {hasSessionMetaValue(sessionDeviceId) ? sessionDeviceId : '—'}
          </span>
        </span>
        <span className={`mobile-session-chip${checkHzData && totalGaps > 0 ? ' mobile-session-gaps' : ''}`}>
          <span className="mobile-session-key">Пропуски</span>
          <span className="mobile-session-val">
            {!checkHzData ? '—' : totalGaps > 0 ? totalGaps : 'нет'}
          </span>
        </span>
      </div>

      <div className={`app-body${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
        <aside className="sidebar" style={sidebarCollapsed ? undefined : { width: sidebarWidth }}>
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={() => setSidebarCollapsed(v => !v)}
            title={sidebarCollapsed ? 'Развернуть панель' : 'Свернуть панель'}
            aria-label={sidebarCollapsed ? 'Развернуть боковую панель' : 'Свернуть боковую панель'}
            aria-expanded={!sidebarCollapsed}
          >
            {sidebarCollapsed ? '▶' : '◀'}
          </button>

          {(!sidebarCollapsed || isMobile) && (
            <div className="sidebar-scroll">
              <SidebarSection
                title="1. Данные"
                open={dataPanelOpen}
                onToggle={() => setDataPanelOpen(v => !v)}
              >
                <div className="sidebar-actions">
                  <div className="btn-group btn-group-block">
                    <UploadBtn accept="video/*,.mp4,.webm,.mov,.avi" onFile={loadVideo}>
                      <UiIcon name="video" /> Видео
                    </UploadBtn>
                    <UploadBtn accept=".parquet" onFile={loadParquetFile}>
                      <UiIcon name="database" /> Parquet
                    </UploadBtn>
                    <UploadBtn accept=".csv,text/csv" onFile={importLabeledCsv}>
                      <UiIcon name="file-table" /> CSV
                    </UploadBtn>
                  </div>

                  <SessionInfoCard
                    protocolName={sessionProtocolName}
                    deviceId={sessionDeviceId}
                    gapCount={totalGaps}
                    gapsKnown={Boolean(checkHzData)}
                  />

                  <div className="sidebar-field">
                    <span className="sidebar-field-lbl">Сессия</span>
                    <div className="session-group session-group-stack">
                      <div className="session-combo">
                        <input
                          ref={sessionInputRef}
                          type="text"
                          inputMode="numeric"
                          className="input-sm session-input session-input-wide"
                          value={sessionId}
                          onChange={e => { setSessionId(e.target.value); setShowSessionDropdown(true) }}
                          onFocus={() => setShowSessionDropdown(true)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') { setShowSessionDropdown(false); loadSession() }
                            if (e.key === 'Escape') setShowSessionDropdown(false)
                          }}
                          placeholder={sessionsListLoading ? 'Загрузка…' : '3421'}
                          autoComplete="off"
                        />
                        {showSessionDropdown && filteredSessions.length > 0 && (
                          <ul ref={dropdownRef} className="session-dropdown">
                            {filteredSessions.map(s => (
                              <li
                                key={s.id}
                                className={`session-dropdown-item${String(s.id) === sessionId ? ' selected' : ''}`}
                                onMouseDown={e => {
                                  e.preventDefault()
                                  setSessionId(String(s.id))
                                  setShowSessionDropdown(false)
                                }}
                              >
                                <span className="sdi-id">#{s.id}</span>
                                <span className="sdi-name">{s.member_name || '—'}</span>
                                {s.protocol_name && <span className="sdi-protocol">{s.protocol_name}</span>}
                                {s.date && <span className="sdi-date">{s.date.slice(0, 10)}</span>}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                      <button
                        type="button"
                        className="btn-primary btn-block"
                        onClick={() => { setShowSessionDropdown(false); loadSession() }}
                        disabled={!sessionId.trim() || status.type === 'loading'}
                      >
                        <UiIcon name="download" /> Загрузить сессию
                      </button>
                    </div>
                  </div>

                  {parquetData && (
                    <div className="sidebar-block imu-preprocess-block">
                      <span className="sidebar-block-lbl">Постпроцессинг сырых IMU</span>
                      <select
                        className="select-sm imu-target-select"
                        value={imuTargetSensor}
                        onChange={e => setImuTargetSensor(e.target.value)}
                        disabled={imuProcessing}
                        title="Какие датчики обрабатывать"
                      >
                        <option value="auto">Автодетекция (по гравитации)</option>
                        {insoleSensorNames.map(name => (
                          <option key={name} value={name}>
                            {sensorFootForName(name, insoleSensorNames) === 'left' ? 'Левая стопа' : 'Правая стопа'} ({name})
                          </option>
                        ))}
                        <option value="all">Все датчики</option>
                      </select>
                      <button
                        type="button"
                        className="btn-secondary btn-block"
                        onClick={handlePreprocessImu}
                        disabled={imuProcessing}
                        title="Удаляет гравитацию, согласует оси linX/linY и пересчитывает Heading/Roll/Pitch в формат новой прошивки"
                      >
                        <UiIcon name={imuProcessing ? 'loader' : 'rotate'} />
                        {imuProcessing ? 'Обработка IMU…' : 'Применить к сессии'}
                      </button>
                      {imuApplied && (
                        <button
                          type="button"
                          className="btn-secondary btn-block"
                          onClick={handleRevertImu}
                          disabled={imuProcessing}
                          title="Вернуть исходные каналы IMU, как они были загружены"
                        >
                          <UiIcon name="undo" /> Откатить
                        </button>
                      )}
                    </div>
                  )}

                  {status.text && status.area !== 'models' && (
                    <span
                      className={`status-pill status-${status.type} status-block`}
                      role={status.type === 'error' ? 'alert' : 'status'}
                      aria-live={status.type === 'error' ? 'assertive' : 'polite'}
                    >
                      {status.text}
                    </span>
                  )}
                </div>
              </SidebarSection>

              {columns.length > 0 && (
                <>
                  <SidebarSection
                  title="2. График"
                  open={chartPanelOpen}
                  onToggle={() => setChartPanelOpen(v => !v)}
                >
                  <div className="sidebar-actions">
                    {(insoleSensorNames.length > 0 || hasSpeedTracker) && (
                      <div className="sidebar-block">
                        <span className="sidebar-block-lbl">Сенсоры</span>
                        <div className="sidebar-chip-list sensor-list">
                          {insoleSensorNames.map(name => {
                            const foot      = sensorFootForName(name, insoleSensorNames) || 'right'
                            const isLeft    = foot === 'left'
                            const isVisible = isLeft ? showSensor1 : showSensor2
                            const toggle    = () => isLeft ? setShowSensor1(v => !v) : setShowSensor2(v => !v)
                            const color     = isLeft ? PALETTE[0] : '#ff7f0e'
                            const stats     = checkHzData?.[name]
                            return (
                              <div key={name} className="sensor-group sensor-group-stack">
                                <button
                                  type="button"
                                  className={`btn-toggle sensor-badge${isVisible ? '' : ' sensor-badge-off'}`}
                                  style={{ '--sensor-color': color }}
                                  onClick={toggle}
                                  aria-pressed={isVisible}
                                  title={`${isVisible ? 'Скрыть' : 'Показать'} ${isLeft ? 'левую' : 'правую'} ногу · ${name}${stats
                                    ? ` · интервал ${formatInterval(stats.time_diff_mean)} ${timeUnit} · максимум ${formatInterval(stats.time_diff_max)} ${timeUnit} · пропусков ${stats.gaps?.length || 0}`
                                    : ''}`}
                                >
                                  <span className="sensor-list-dot" style={{ background: color }} />
                                  <span className="sensor-list-copy">
                                    <span className="sensor-list-name">{name.replace('ESP32_', '')}</span>
                                    {stats && (
                                      <span className="sensor-list-metrics">
                                        <span>Δt <b>{formatInterval(stats.time_diff_mean)} {timeUnit}</b></span>
                                        <span>макс. <b>{formatInterval(stats.time_diff_max)} {timeUnit}</b></span>
                                        <span className={(stats.gaps?.length || 0) > 0 ? 'has-gaps' : ''}>
                                          пропуски <b>{stats.gaps?.length || 0}</b>
                                        </span>
                                      </span>
                                    )}
                                  </span>
                                  <span className="sensor-side-badge">{isLeft ? 'L' : 'R'}</span>
                                </button>
                              </div>
                            )
                          })}
                          {hasSpeedTracker && (
                            <div className="sensor-group sensor-group-stack">
                              <button
                                type="button"
                                className={`btn-toggle sensor-badge sensor-badge-st${showSpeedTracker ? '' : ' sensor-badge-off'}`}
                                style={{ '--sensor-color': ST_COLOR }}
                                onClick={() => setShowSpeedTracker(v => !v)}
                                aria-pressed={showSpeedTracker}
                                title={`${showSpeedTracker ? 'Скрыть' : 'Показать'} SpeedTracker${checkHzData?.[SPEED_TRACKER]
                                  ? ` · интервал ${formatInterval(checkHzData[SPEED_TRACKER].time_diff_mean)} ${timeUnit} · максимум ${formatInterval(checkHzData[SPEED_TRACKER].time_diff_max)} ${timeUnit} · пропусков ${checkHzData[SPEED_TRACKER].gaps?.length || 0}`
                                  : ''}`}
                              >
                                <span className="sensor-list-dot" style={{ background: ST_COLOR }} />
                                <span className="sensor-list-copy">
                                  <span className="sensor-list-name">SpeedTracker</span>
                                  {checkHzData?.[SPEED_TRACKER] && (
                                    <span className="sensor-list-metrics">
                                      <span>Δt <b>{formatInterval(checkHzData[SPEED_TRACKER].time_diff_mean)} {timeUnit}</b></span>
                                      <span>макс. <b>{formatInterval(checkHzData[SPEED_TRACKER].time_diff_max)} {timeUnit}</b></span>
                                      <span className={(checkHzData[SPEED_TRACKER].gaps?.length || 0) > 0 ? 'has-gaps' : ''}>
                                        пропуски <b>{checkHzData[SPEED_TRACKER].gaps?.length || 0}</b>
                                      </span>
                                    </span>
                                  )}
                                </span>
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="sidebar-block columns-picker">
                      <button
                        type="button"
                        className={`columns-picker-toggle${columnsPanelOpen ? ' open' : ''}`}
                        onClick={() => setColumnsPanelOpen(open => !open)}
                        aria-expanded={columnsPanelOpen}
                        aria-controls="graph-columns-picker"
                      >
                        <span className="columns-picker-title">Колонки</span>
                        <span className="columns-picker-count">Выбрано: {selectedCols.length}</span>
                        <span className="columns-picker-chevron">⌄</span>
                      </button>

                      {columnsPanelOpen && (
                        <div id="graph-columns-picker" className="columns-picker-body">
                          <div className="sidebar-chip-list">
                            {columns.map(col => (
                              <button
                                type="button"
                                key={col}
                                className={`btn-toggle col-chip${selectedCols.includes(col) ? ' active' : ''}`}
                                style={selectedCols.includes(col)
                                  ? { '--c': PALETTE[selectedCols.indexOf(col) % PALETTE.length] } : {}}
                                onClick={() => toggleCol(col)}
                              >{col}</button>
                            ))}
                          </div>
                          <div className="btn-group btn-group-sm">
                            <button type="button" className="btn-toggle col-chip ghost" onClick={() => setSelectedCols([...columns])}>все</button>
                            <button type="button" className="btn-toggle col-chip ghost" onClick={() => setSelectedCols([])}>сброс</button>
                          </div>
                        </div>
                      )}
                    </div>

                  </div>
                  </SidebarSection>

                  <SidebarSection
                    title={`3. Модели и анализ${activeModelCount > 0 ? ` · ${activeModelCount}` : ''}`}
                    open={modelsPanelOpen}
                    onToggle={() => setModelsPanelOpen(v => !v)}
                  >
                    <div className="sidebar-actions models-sidebar-actions">
                    {(hasSpeedTracker || insoleSensorNames.length > 0) && (
                      <div className="calculator-panel">
                        <div className="models-panel-intro">
                          <span>Прогнозы и детекторы</span>
                          <span>{activeModelCount > 0 ? `Активно: ${activeModelCount}` : 'Выберите модель'}</span>
                        </div>
                        {status.text && status.area === 'models' && (
                          <span
                            className={`status-pill status-${status.type} status-block model-status`}
                            role={status.type === 'error' ? 'alert' : 'status'}
                            aria-live={status.type === 'error' ? 'assertive' : 'polite'}
                          >
                            {status.text}
                          </span>
                        )}
                        <div className="sidebar-block-row calculator-primary-row">
                        <div className="sidebar-block">
                          <span className="sidebar-block-lbl">Скорость · CausalSpeedTCN</span>
                          <button
                            type="button"
                            className={`btn-secondary btn-speed-predict${showSpeedPredict ? ' active' : ''}`}
                            onClick={fetchSpeedPredict}
                            disabled={predictLoading || !sessionId.trim()}
                            title={!sessionId.trim()
                              ? 'Укажите ID сессии — прогноз берётся по сессии (charts/sprint)'
                              : showSpeedPredict
                                ? 'Убрать прогноз скорости с графика'
                                : `Загрузить charts/sprint${hasSpeedTracker ? ' и наложить поверх колонки Speed' : ' для этой сессии'}`}
                          >
                            <UiIcon name={predictLoading ? 'loader' : showSpeedPredict ? 'x' : 'bolt'} />
                            {predictLoading ? 'Загрузка…' : showSpeedPredict ? 'Убрать speed predict' : 'Speed predict'}
                          </button>
                          {showSpeedPredict && speedPredict?.stat && (
                            <span className="hz-stats hz-stats-compact" style={{ '--hzc': PRED_COLOR }}>
                              {speedPredict.stat.peak_speed != null && (
                                <span className="hz-stat-item" title="пиковая скорость">
                                  <span className="hz-stat-key">пик</span>
                                  <span className="hz-stat-val">{speedPredict.stat.peak_speed.toFixed(2)}</span>
                                </span>
                              )}
                              {speedPredict.stat.average_speed != null && (
                                <>
                                  <span className="hz-stat-sep" />
                                  <span className="hz-stat-item" title="средняя скорость на участке 30 м">
                                    <span className="hz-stat-key">ср</span>
                                    <span className="hz-stat-val">{speedPredict.stat.average_speed.toFixed(2)}</span>
                                  </span>
                                </>
                              )}
                              {speedPredict.stat.duration != null && (
                                <>
                                  <span className="hz-stat-sep" />
                                  <span className="hz-stat-item" title="время прохождения 30 м, с">
                                    <span className="hz-stat-key">30м</span>
                                    <span className="hz-stat-val">{speedPredict.stat.duration.toFixed(2)}с</span>
                                  </span>
                                </>
                              )}
                            </span>
                          )}
                        </div>

                        <div className="sidebar-block">
                          <span className="sidebar-block-lbl">Дистанция · CausalSpeedTCN</span>
                          <button
                            type="button"
                            className={`btn-secondary btn-distance-predict${showDistancePredict ? ' active' : ''}`}
                            onClick={fetchDistancePredict}
                            disabled={predictLoading || !sessionId.trim()}
                            title={!sessionId.trim()
                              ? 'Укажите ID сессии — прогноз берётся по сессии (charts/sprint)'
                              : showDistancePredict
                                ? 'Убрать прогноз дистанции с графика'
                                : `Загрузить charts/sprint${hasSpeedTracker ? ' и наложить поверх колонки Distance' : ' для этой сессии'}`}
                          >
                            <UiIcon name={predictLoading ? 'loader' : showDistancePredict ? 'x' : 'ruler'} />
                            {predictLoading ? 'Загрузка…' : showDistancePredict ? 'Убрать distance predict' : 'Distance predict'}
                          </button>
                          {showDistancePredict && speedPredict?.stat && (
                            <span className="hz-stats hz-stats-compact" style={{ '--hzc': PRED_COLOR }}>
                              {speedPredict.stat.distance_at_peak_speed != null && (
                                <span className="hz-stat-item" title="дистанция на момент пика скорости">
                                  <span className="hz-stat-key">на пике</span>
                                  <span className="hz-stat-val">{speedPredict.stat.distance_at_peak_speed.toFixed(1)}м</span>
                                </span>
                              )}
                              {speedPredict.stat.duration != null && (
                                <>
                                  <span className="hz-stat-sep" />
                                  <span className="hz-stat-item" title="время прохождения 30 м, с">
                                    <span className="hz-stat-key">30м</span>
                                    <span className="hz-stat-val">{speedPredict.stat.duration.toFixed(2)}с</span>
                                  </span>
                                </>
                              )}
                            </span>
                          )}
                        </div>
                          </div>

                        <button
                          type="button"
                          className={`calculator-expand${protocolDetectorsOpen ? ' open' : ''}`}
                          onClick={() => setProtocolDetectorsOpen(open => !open)}
                          aria-expanded={protocolDetectorsOpen}
                          aria-controls="protocol-detectors"
                        >
                          <span>
                            Детекторы протоколов
                            {activeCalculators.some(id => PROTOCOL_SECTION_CALCULATOR_IDS.has(id)) && (
                              <span className="calculator-active-count">
                                {activeCalculators.filter(id => PROTOCOL_SECTION_CALCULATOR_IDS.has(id)).length}
                              </span>
                            )}
                          </span>
                          <span className="calculator-expand-chevron">⌄</span>
                        </button>

                        {protocolDetectorsOpen && (
                          <div id="protocol-detectors" className="calculator-options">
                            {PROTOCOL_DETECTORS.map(detector => {
                              const active = activeCalculators.includes(detector.id)
                              const loading = calculatorLoading === detector.id
                              const result = calculatorResults[detector.id]
                              const eventLegend = calculatorEventLegend(detector, result)
                              const supportsPerFootDetection = PER_FOOT_TURN_DETECTOR_IDS.has(detector.id)
                              const selectedDetectionFoot = turnDetectionFeet[detector.id] || 'both'
                              return (
                                <div key={detector.id} className="calculator-option">
                                  <button
                                    type="button"
                                    className={`btn-secondary btn-calculator${active ? ' active' : ''}`}
                                    style={{ '--calculator-color': detector.color }}
                                    disabled={!parquetData || !!calculatorLoading}
                                    onClick={() => toggleAdditionalCalculator(detector.id)}
                                    title={active
                                      ? `Убрать события «${detector.label}» с графика`
                                      : `Запустить «${detector.label}» на загруженных данных`}
                                  >
                                    <span className="calculator-dot" />
                                    {loading ? 'Детектирую…' : active ? `Убрать ${detector.label}` : detector.label}
                                  </button>
                                  <span className="calculator-description">{detector.description}</span>
                                  {supportsPerFootDetection && (
                                    <div className="turn-foot-selector" role="radiogroup" aria-label={`Нога для детекции поворотов: ${detector.label}`}>
                                      <span className="turn-foot-selector-label">Источник:</span>
                                      {TURN_DETECTION_FOOT_OPTIONS.map(option => {
                                        const sensorName = option.value === 'both'
                                          ? ''
                                          : sensorNameForFoot(insoleSensorNames, option.value)
                                        const available = option.value === 'both' || Boolean(sensorName)
                                        return (
                                          <button
                                            key={option.value}
                                            type="button"
                                            className={`turn-foot-choice${selectedDetectionFoot === option.value ? ' active' : ''}`}
                                            role="radio"
                                            aria-checked={selectedDetectionFoot === option.value}
                                            disabled={!available || !!calculatorLoading}
                                            title={available
                                              ? `${option.title}${sensorName ? ` · ${sensorName}` : ''}`
                                              : 'Сенсор этой ноги отсутствует в данных'}
                                            onClick={() => {
                                              if (selectedDetectionFoot === option.value) return
                                              setTurnDetectionFeet(prev => ({ ...prev, [detector.id]: option.value }))
                                              if (active) {
                                                void toggleAdditionalCalculator(detector.id, {
                                                  force: true,
                                                  detectionFoot: option.value,
                                                })
                                              }
                                            }}
                                          >
                                            {option.label}
                                          </button>
                                        )
                                      })}
                                    </div>
                                  )}
                                  {result?.model && (
                                    <span className="calculator-model">
                                      Детектор: {result.model}{result.model_file ? ` · ${result.model_file}` : ''}
                                    </span>
                                  )}
                                  {result && (
                                    <span className="calculator-summary" style={{ '--calculator-color': detector.color }}>
                                      {protocolDetectorSummary(result)}
                                    </span>
                                  )}
                                  {active && eventLegend.length > 0 && (
                                    <div className="calculator-event-legend" aria-label="Легенда событий">
                                      {eventLegend.map(item => (
                                        <span
                                          key={item.key}
                                          className="calculator-event-key"
                                          style={{ '--event-color': item.color, '--event-fill': item.fill }}
                                        >
                                          <span className="calculator-event-swatch" />
                                          {item.label}
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        )}

                        {protocolDetectorsOpen && (
                          <div className="calculator-options calculator-featured-options">
                            {FEATURED_EXTRA_CALCULATORS.map(calculator => {
                            const active = activeCalculators.includes(calculator.id)
                            const loading = calculatorLoading === calculator.id
                            const result = calculatorResults[calculator.id]
                            const summary = result?.summary
                            const leftCount = summary?.left?.contact_count || 0
                            const rightCount = summary?.right?.contact_count || 0
                            const eventLegend = calculatorEventLegend(calculator, result)
                            return (
                              <div
                                key={calculator.id}
                                className="calculator-option calculator-option-featured"
                                style={{ '--calculator-color': calculator.color }}
                              >
                                <button
                                  type="button"
                                  className={`btn-secondary btn-calculator${active ? ' active' : ''}`}
                                  style={{ '--calculator-color': calculator.color }}
                                  disabled={!parquetData || !!calculatorLoading}
                                  onClick={() => toggleAdditionalCalculator(calculator.id)}
                                  title={active
                                    ? `Убрать ${calculator.label} с графика`
                                    : `Запустить ${calculator.label} для загруженных данных`}
                                >
                                  <span className="calculator-dot" />
                                  {loading ? 'Детектирую…' : active ? `Убрать ${calculator.label}` : calculator.label}
                                </button>
                                <span className="calculator-description">{calculator.description}</span>
                                {result?.model && (
                                  <span className="calculator-model">
                                    Детектор: {result.model}{result.model_file ? ` · ${result.model_file}` : ''}
                                  </span>
                                )}
                                {result && (
                                  <span className="calculator-summary" style={{ '--calculator-color': calculator.color }}>
                                    <span>L {leftCount} · R {rightCount}</span>
                                    <br />
                                    <span>
                                      GCT L {formatMetric(summary?.left?.mean_contact_duration_s != null
                                        ? summary.left.mean_contact_duration_s * 1000 : null, 0, ' ms')}
                                      {' · '}
                                      GCT R {formatMetric(summary?.right?.mean_contact_duration_s != null
                                        ? summary.right.mean_contact_duration_s * 1000 : null, 0, ' ms')}
                                    </span>
                                    <br />
                                    <span>
                                      step L {formatMetric(summary?.left?.mean_step_interval_s, 3, ' s')}
                                      {' · '}
                                      step R {formatMetric(summary?.right?.mean_step_interval_s, 3, ' s')}
                                    </span>
                                  </span>
                                )}
                                {active && eventLegend.length > 0 && (
                                  <div className="calculator-event-legend" aria-label="Легенда шагов T-теста">
                                    {eventLegend.map(item => (
                                      <span
                                        key={item.key}
                                        className="calculator-event-key"
                                        style={{ '--event-color': item.color, '--event-fill': item.fill }}
                                      >
                                        <span className="calculator-event-swatch" />
                                        {item.label}
                                      </span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )
                            })}
                          </div>
                        )}

                        {selectedCalculatorContact && (() => {
                          const detail = selectedCalculatorContact.contact
                          const calculator = CALCULATOR_BY_ID[selectedCalculatorContact.calculatorId]
                          const detailStyle = calculatorEventStyle(calculator, detail)
                          const foot = detail.foot === 'left' ? 'L' : detail.foot === 'right' ? 'R' : 'ALL'
                          const durationLabel = {
                            flight: 'Flight',
                            contact: 'GCT',
                            step: 'Контакт шага',
                            turn: 'Поворот',
                            run: 'Беговая фаза',
                            sprint: 'Спринт',
                          }[detail.kind] || 'Событие'
                          return (
                            <div className="calculator-contact-detail" style={{ '--calculator-color': detailStyle.color }}>
                              <div className="calculator-contact-head">
                                <span>{calculator?.label || 'ML-контакт'} · {foot} · #{selectedCalculatorContact.index + 1}</span>
                                <button
                                  type="button"
                                  className="calculator-contact-close"
                                  onClick={() => setSelectedCalculatorContact(null)}
                                  aria-label="Закрыть показатели контакта"
                                >×</button>
                              </div>
                              <div className="calculator-contact-metrics">
                                <span><b>{durationLabel}</b> {formatMetric(detail.duration_ms, 0, ' ms')}</span>
                                <span>начало {formatMetric(detail.start_time_s, 3, ' s')}</span>
                                <span>конец {formatMetric(detail.end_time_s, 3, ' s')}</span>
                                {detail.confidence != null && (
                                  <span>confidence {formatMetric(Number(detail.confidence) * 100, 0, '%')}</span>
                                )}
                                {detail.direction && <span>направление {detail.direction}</span>}
                                {detail.angle_deg != null && <span>угол {formatMetric(detail.angle_deg, 0, '°')}</span>}
                                {detail.jump_height_cm != null && (
                                  <span><b>высота</b> {formatMetric(detail.jump_height_cm, 1, ' см')}</span>
                                )}
                                {detail.step_length_m != null && <span>step length {formatMetric(detail.step_length_m, 3, ' м')}</span>}
                                {detail.stride_length_m != null && <span>stride length {formatMetric(detail.stride_length_m, 3, ' м')}</span>}
                                {detail.distance_m != null && detail.kind === 'sprint' && (
                                  <span><b>дистанция</b> {formatMetric(detail.distance_m, 2, ' м')}</span>
                                )}
                                {detail.distance_m != null && detail.kind === 'step' && (
                                  <span>дистанция {formatMetric(detail.distance_m, 2, ' м')}</span>
                                )}
                              </div>
                            </div>
                          )
                        })()}

                        <button
                          type="button"
                          className={`calculator-expand${extraCalculatorsOpen ? ' open' : ''}`}
                          onClick={() => setExtraCalculatorsOpen(open => !open)}
                          aria-expanded={extraCalculatorsOpen}
                          aria-controls="extra-calculators"
                        >
                          <span>
                            Другие калькуляторы
                            {activeCalculators.some(id => COLLAPSIBLE_EXTRA_CALCULATORS.some(calculator => calculator.id === id)) && (
                              <span className="calculator-active-count">
                                {activeCalculators.filter(id => COLLAPSIBLE_EXTRA_CALCULATORS.some(calculator => calculator.id === id)).length}
                              </span>
                            )}
                          </span>
                          <span className="calculator-expand-chevron">⌄</span>
                        </button>

                        {extraCalculatorsOpen && (
                          <div id="extra-calculators" className="calculator-options">
                            <div className="calculator-advanced-settings">
                              <div className="calculator-weight-row">
                                <label htmlFor="calculator-weight">Вес для GRF, кг</label>
                                <input
                                  id="calculator-weight"
                                  type="number"
                                  min="1"
                                  max="300"
                                  step="0.1"
                                  value={weightKg}
                                  onChange={event => setWeightKg(event.target.value)}
                                />
                                <span>для Bilateral GRF</span>
                              </div>
                              <div className="calculator-model-note">
                                Модели: <b>step_gc_model.pt</b>, <b>jump_bilstm.pt</b>, <b>jump_force_total.pt</b>
                              </div>
                            </div>
                            {COLLAPSIBLE_EXTRA_CALCULATORS.map(calculator => {
                              const active = activeCalculators.includes(calculator.id)
                              const loading = calculatorLoading === calculator.id
                              const result = calculatorResults[calculator.id]
                              const summary = result?.summary
                              const leftCount = summary?.left?.contact_count || 0
                              const rightCount = summary?.right?.contact_count || 0
                              const eventLegend = calculatorEventLegend(calculator, result)
                              return (
                                <div key={calculator.id} className="calculator-option">
                                  <button
                                    type="button"
                                    className={`btn-secondary btn-calculator${active ? ' active' : ''}`}
                                    style={{ '--calculator-color': calculator.color }}
                                    disabled={!parquetData || !!calculatorLoading}
                                    onClick={() => toggleAdditionalCalculator(calculator.id)}
                                    title={active
                                      ? `Убрать ${calculator.label} с графика`
                                      : `Запустить ${calculator.label} для загруженных данных`}
                                  >
                                    <span className="calculator-dot" />
                                    {loading ? 'Считаю…' : active ? `Убрать ${calculator.label}` : calculator.label}
                                  </button>
                                  <span className="calculator-description">{calculator.description}</span>
                                  {result?.model && (
                                    <span className="calculator-model">
                                      Модель: {result.model}{result.model_file ? ` · ${result.model_file}` : ''}
                                    </span>
                                  )}
                                  {result && (
                                    <span className="calculator-summary" style={{ '--calculator-color': calculator.color }}>
                                      {calculator.id === 'jump-metrics'
                                        ? `${summary?.total_jump_count || 0} прыж. · высота ${formatMetric(summary?.mean_jump_height_cm, 1, ' см')} · flight ${formatMetric(summary?.left_mean_flight_time_ms, 0, ' мс')}`
                                        : calculator.id === 'force-jump'
                                          ? `пик ${formatMetric(summary?.peak_force_n, 1, ' Н')} · ${formatMetric(summary?.peak_force_bw, 2, ' BW')}`
                                          : <>
                                            <span>
                                              L {leftCount} · R {rightCount}
                                              {summary?.cadence_spm != null && ` · ${summary.cadence_spm.toFixed(0)} spm`}
                                            </span>
                                            <br />
                                            <span>
                                              GCT L {formatMetric(summary?.left?.mean_contact_duration_s != null
                                                ? summary.left.mean_contact_duration_s * 1000 : null, 0, ' ms')}
                                              {' · '}
                                              GCT R {formatMetric(summary?.right?.mean_contact_duration_s != null
                                                ? summary.right.mean_contact_duration_s * 1000 : null, 0, ' ms')}
                                            </span>
                                            <br />
                                            <span>
                                              step L {formatMetric(summary?.left?.mean_step_interval_s, 3, ' s')}
                                              {' · '}
                                              step R {formatMetric(summary?.right?.mean_step_interval_s, 3, ' s')}
                                            </span>
                                            {(summary?.left?.mean_confidence != null || summary?.right?.mean_confidence != null) && (
                                              <>
                                                <br />
                                                <span>
                                                  {summary?.left?.mean_confidence != null
                                                    && `conf L ${(summary.left.mean_confidence * 100).toFixed(0)}%`}
                                                  {summary?.right?.mean_confidence != null
                                                    && ` · conf R ${(summary.right.mean_confidence * 100).toFixed(0)}%`}
                                                </span>
                                              </>
                                            )}
                                          </>}
                                    </span>
                                  )}
                                  {active && eventLegend.length > 0 && (
                                    <div className="calculator-event-legend" aria-label="Легенда событий">
                                      {eventLegend.map(item => (
                                        <span
                                          key={item.key}
                                          className="calculator-event-key"
                                          style={{ '--event-color': item.color, '--event-fill': item.fill }}
                                        >
                                          <span className="calculator-event-swatch" />
                                          {item.label}
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    )}

                    </div>
                  </SidebarSection>
                </>
              )}
            </div>
          )}
        </aside>

        {!sidebarCollapsed && (
          <div
            className="panel-resizer sidebar-panel-resizer"
            onMouseDown={startSidebarResize}
            onKeyDown={resizeSidebarWithKeyboard}
            tabIndex={0}
            role="separator"
            aria-orientation="vertical"
            aria-label="Изменить ширину боковой панели"
            aria-valuemin={250}
            aria-valuemax={460}
            aria-valuenow={Math.round(sidebarWidth)}
          />
        )}

        <div className="main-area">
      {/* ── Content ── */}
      <div className="content">

        {/* Left: video */}
        <div
          ref={videoSideRef}
          className={`video-side${videoPanelOpen ? '' : ' video-side-hidden'}`}
          style={videoPanelWidth == null ? undefined : { width: videoPanelWidth }}
        >
          <div
            ref={videoWrapRef}
            className={`video-wrap${zoom > 1 ? ' zoomed' : ''}`}
            onMouseDown={handleVideoPanStart}
          >
            {videoUrl ? (
              <div
                className="video-transform"
                style={{
                  transform: `scale(${zoom}) translate(${panX}px, ${panY}px)`,
                  transformOrigin: 'center center',
                  cursor: zoom > 1 ? 'grab' : 'default',
                }}
              >
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  className="video-el"
                  onLoadedMetadata={e => setVideoDuration(e.target.duration)}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
            ) : (
              <div className="drop-hint">
                <span>📹</span>
                <p>Перетащите видео или загрузите в панели слева</p>
              </div>
            )}

            {videoUrl && (
              <div className="zoom-overlay">
                <button className="zoom-btn" onClick={() => changeZoom(1.25)} title="Приблизить" aria-label="Приблизить"><UiIcon name="plus" /></button>
                <span className="zoom-label">{zoom.toFixed(1)}×</span>
                <button className="zoom-btn" onClick={() => changeZoom(1 / 1.25)} title="Отдалить" aria-label="Отдалить"><UiIcon name="minus" /></button>
                {zoom > 1 && (
                  <button className="zoom-btn zoom-reset" onClick={resetZoom} title="Сбросить масштаб" aria-label="Сбросить масштаб"><UiIcon name="maximize" /></button>
                )}
              </div>
            )}
          </div>

          <div className="time-bar">
            <span ref={vidLblRef} className="time-lbl">0:00.0</span>
            <span ref={imuLblRef} className="time-lbl imu-lbl">IMU 0.00s</span>
            <span className="time-lbl muted">S1:{offsetS1} S2:{offsetS2}{hasSpeedTracker ? ` ST:${offsetST}` : ''}{timeUnit === 'ms' ? 'мс' : 'с'}</span>
            <span className="time-lbl muted dur">{formatTime(videoDuration)}</span>
          </div>

          {videoDuration > 0 && (
            <div
              className="timeline"
              ref={timelineRef}
              onMouseDown={e => { isDragging.current = true; seekFromX(e.clientX) }}
            >
              <div className="tl-played" style={{ width: `${cursorPct}%` }} />
              {ticks.map(({ t, pct }) => (
                <div key={t} className="tl-tick" style={{ left: `${pct}%` }}>
                  <div className="tl-tick-line" />
                  <span className="tl-tick-lbl">{formatTime(t)}</span>
                </div>
              ))}
              <div className="tl-cursor" style={{ left: `${cursorPct}%` }}>
                <div className="tl-cursor-head" />
                <div className="tl-cursor-line" />
              </div>
            </div>
          )}
        </div>

        {videoPanelOpen && (
          <div
            className="panel-resizer video-panel-resizer"
            onMouseDown={startVideoResize}
            onKeyDown={resizeVideoWithKeyboard}
            tabIndex={0}
            role="separator"
            aria-orientation="vertical"
            aria-label="Изменить ширину видеопанели"
            aria-valuemin={280}
            aria-valuemax={1200}
            aria-valuenow={Math.round(videoPanelWidth || 480)}
          />
        )}

        {/* Right: labeling + chart */}
        <div className="chart-side">
          <div className="label-panel">
            <div className="label-toolbar-row">
              <div className="label-toolbar-group">
                <button
                  type="button"
                  className={`btn-toggle charts-lock-btn${chartsLocked ? ' active' : ''}`}
                  onClick={toggleChartsLock}
                  disabled={!chartReady || selectedCols.length < 2}
                  aria-pressed={chartsLocked}
                  title={
                    selectedCols.length < 2
                      ? 'Выберите несколько графиков, чтобы двигать их вместе'
                      : chartsLocked
                        ? 'Открепить: каждый график двигается отдельно'
                        : 'Закрепить: масштабирование и сдвиг по времени на всех графиках сразу'
                  }
                >
                  <UiIcon name="pin" /> Закрепить
                </button>

                <button
                  type="button"
                  className={`btn-toggle video-panel-toggle${videoPanelOpen ? ' active' : ''}`}
                  onClick={toggleVideoPanel}
                  aria-pressed={videoPanelOpen}
                  title={videoPanelOpen ? 'Скрыть видеопанель и расширить график' : 'Показать видеопанель'}
                >
                  <UiIcon name="video" />
                  <span>{videoPanelOpen ? 'Скрыть видео' : videoUrl ? 'Показать видео' : 'Видео'}</span>
                </button>

                <button
                  type="button"
                  className={`btn-toggle lab-mode-btn${labelingMode ? ' active' : ''}`}
                  onClick={() => setLabelingMode(m => !m)}
                  aria-pressed={labelingMode}
                  title={labelingMode ? 'Выключить режим разметки' : 'Включить режим разметки'}
                >
                  <UiIcon name="pencil" /> {labelingMode ? 'Разметка вкл' : 'Разметка'}
                </button>

                <button
                  type="button"
                  className={`btn-toggle gap-vis-btn${showGaps ? ' vis-on' : ''}`}
                  onClick={() => setShowGaps(v => !v)}
                  aria-pressed={showGaps}
                  disabled={!checkHzData || totalGaps === 0 || !chartReady}
                  title={
                    !checkHzData
                      ? 'Загрузите сессию для анализа пропусков'
                      : totalGaps === 0
                        ? 'Пропусков в данных не обнаружено'
                        : showGaps
                          ? 'Скрыть пропуски на графике'
                          : `Показать ${totalGaps} пропуск(ов) красными отрезками`
                  }
                >
                  <UiIcon name="gaps" /> Пропуски{totalGaps > 0 ? ` (${totalGaps})` : ''}
                </button>

                {parquetData && (
                  <div className="chart-sync-controls" aria-label="Сдвиги времени и углы">
                    <span className="chart-sync-title">Сдвиг</span>
                    <span className="offset-pair">
                      <span className="offset-lbl offset-lbl-s1">S1</span>
                      <OffsetInput
                        value={offsetS1}
                        step={timeUnit === 'ms' ? 100 : 0.05}
                        title="Сдвиг Sensor 1 (левая нога)"
                        onChange={setOffsetS1}
                      />
                    </span>
                    <span className="offset-pair">
                      <span className="offset-lbl offset-lbl-s2">S2</span>
                      <OffsetInput
                        value={offsetS2}
                        step={timeUnit === 'ms' ? 100 : 0.05}
                        title="Сдвиг Sensor 2 (правая нога)"
                        onChange={setOffsetS2}
                      />
                    </span>
                    {hasSpeedTracker && (
                      <span className="offset-pair">
                        <span className="offset-lbl offset-lbl-st">ST</span>
                        <OffsetInput
                          value={offsetST}
                          step={timeUnit === 'ms' ? 100 : 0.05}
                          title="Сдвиг SpeedTracker"
                          onChange={setOffsetST}
                        />
                      </span>
                    )}
                    <div className="btn-group chart-sync-units" aria-label="Единицы времени">
                      <button type="button" className={`btn-toggle unit-btn${timeUnit === 's' ? ' active' : ''}`} aria-pressed={timeUnit === 's'} onClick={() => setTimeUnit('s')}>с</button>
                      <button type="button" className={`btn-toggle unit-btn${timeUnit === 'ms' ? ' active' : ''}`} aria-pressed={timeUnit === 'ms'} onClick={() => setTimeUnit('ms')}>мс</button>
                    </div>
                    <button
                      type="button"
                      className={`btn-secondary btn-unwrap${anglesUnwrapped ? ' active' : ''}`}
                      disabled={!selectedCols.length}
                      onClick={handleUnwrapAngles}
                      title={anglesUnwrapped ? 'Вернуть исходные углы' : 'Развернуть углы'}
                    >
                      <UiIcon name="rotate" /> Углы
                    </button>
                    {/* The span carries the tooltip so the explanation is still
                        reachable when the button is disabled. */}
                    <span className="yaw-drift-ctl" title={yawDriftTitle}>
                      <button
                        type="button"
                        className={`btn-secondary btn-unwrap${yawFixed ? ' active' : ''}`}
                        disabled={!yawDrift?.applied}
                        onClick={handleToggleYawDrift}
                        aria-pressed={yawFixed}
                        aria-label={yawDriftTitle}
                      >
                        <UiIcon name="rotate" /> Дрейф
                      </button>
                      {yawDrift && !yawDrift.applied && (
                        <span className="yaw-drift-note">{yawDrift.reason}</span>
                      )}
                    </span>
                    <button
                      type="button"
                      className={`btn-secondary btn-unwrap${mirrorLeft ? ' active' : ''}`}
                      disabled={!mirrorableCols.length}
                      onClick={handleMirrorLeft}
                      aria-pressed={mirrorLeft}
                      title={mirrorLeftTitle}
                      aria-label={mirrorLeftTitle}
                    >
                      <UiIcon name="mirror" /> Отразить
                    </button>
                  </div>
                )}

                {totalContacts > 0 && (
                  <div className="btn-group">
                    <button
                      type="button"
                      className={`btn-toggle pattern-vis-btn${showLeftPatterns ? ' vis-on' : ''}`}
                      style={{ '--pc': L_LINE }}
                      onClick={() => setShowLeftPatterns(v => !v)}
                      title={showLeftPatterns ? 'Скрыть паттерны Sensor 1' : 'Показать паттерны Sensor 1'}
                    >
                      S1
                    </button>
                    <button
                      type="button"
                      className={`btn-toggle pattern-vis-btn${showRightPatterns ? ' vis-on' : ''}`}
                      style={{ '--pc': R_LINE }}
                      onClick={() => setShowRightPatterns(v => !v)}
                      title={showRightPatterns ? 'Скрыть паттерны Sensor 2' : 'Показать паттерны Sensor 2'}
                    >
                      S2
                    </button>
                  </div>
                )}
              </div>

              <div className="label-toolbar-group label-toolbar-actions" aria-label="Действия с разметкой">
                {(markupFiles.length > 0 || activeMarkupFileId === 'new' || pendingImportFilename) && (
                  <select
                    className="select-sm markup-file-select"
                    value={activeMarkupFileId || 'new'}
                    onChange={e => handleSelectMarkupFile(e.target.value)}
                    title="Выберите версию разметки из БД"
                  >
                    {markupFiles.map(f => (
                      <option key={f.id} value={f.id}>
                        {f.filename} ({new Date(f.updated_at).toLocaleString()})
                      </option>
                    ))}
                    <option value="new">
                      {pendingImportFilename ? `⬆ ${pendingImportFilename}` : '+ Новая разметка'}
                    </option>
                  </select>
                )}
                {labelingMode && (
                  <span className="lab-stat lab-stat-inline">
                    <span className="lab-stat-l">S1: {Math.floor(leftContacts.length / 2)}</span>
                    <span className="lab-stat-sep">·</span>
                    <span className="lab-stat-r">S2: {Math.floor(rightContacts.length / 2)}</span>
                  </span>
                )}
                <button
                  type="button"
                  className="btn-primary lab-btn save-db"
                  onClick={saveMarkupToDb}
                  disabled={isSaving || !sessionId.trim() || !sessionRecordAvailable || totalContacts === 0}
                  title={!sessionId.trim()
                    ? 'Укажите ID сессии слева (например 4102)'
                    : !sessionRecordAvailable
                      ? 'Parquet найден в GCS, но запись этой сессии отсутствует в БД'
                      : 'Сохранить текущую разметку в БД (сессия #' + sessionId.trim() + ')'}
                >
                  <UiIcon name={isSaving ? 'loader' : 'database-check'} />
                  {isSaving ? 'Сохранение…' : 'Сохранить в БД'}
                </button>
                <UploadBtn
                  accept=".csv,text/csv"
                  onFile={importLabeledCsv}
                  className="btn-secondary lab-btn import"
                  title="Загрузить CSV: данные сессии и, если в файле есть Target, интервалы разметки. Номер сессии не нужен"
                >
                  <UiIcon name="upload" /> Импорт CSV
                </UploadBtn>
                <button
                  type="button"
                  className="btn-secondary lab-btn export"
                  onClick={exportLabels}
                  disabled={!parquetData}
                  title={parquetData
                    ? 'Скачать CSV данных с колонкой Target (0, если разметки нет)'
                    : 'Сначала загрузите сессию, parquet или CSV'}
                >
                  <UiIcon name="download" /> Скачать CSV
                </button>
              </div>
            </div>

            {labelingMode && (
              <div className="label-toolbar-row label-toolbar-row-secondary">
                <div className="label-toolbar-group">
                  <div className="btn-group foot-toggle">
                    <button
                      type="button"
                      className={`foot-btn${currentFoot === 'left' ? ' active left-active' : ''}`}
                      onClick={() => setCurrentFoot('left')}
                    >
                      ◀ S1&nbsp;<span className="foot-count">{leftContacts.length}</span>
                    </button>
                    <button
                      type="button"
                      className={`foot-btn${currentFoot === 'right' ? ' active right-active' : ''}`}
                      onClick={() => setCurrentFoot('right')}
                    >
                      S2&nbsp;<span className="foot-count">{rightContacts.length}</span>&nbsp;▶
                    </button>
                  </div>

                  <button type="button" className="btn-secondary lab-btn" onClick={undoContact} title="Отменить последний клик">
                    <UiIcon name="undo" /> Отмена
                  </button>

                  <div className="lab-menu-wrap" ref={labMenuRef}>
                    <button
                      type="button"
                      className="btn-secondary lab-btn lab-menu-trigger"
                      onClick={() => setLabMenuOpen(v => !v)}
                      title="Дополнительные действия"
                    >
                      ⋯
                    </button>
                    {labMenuOpen && (
                      <div className="lab-menu">
                        <button
                          type="button"
                          className="lab-menu-item"
                          onClick={() => { clearCurrentContacts(); setLabMenuOpen(false) }}
                        >
                          Очистить текущую ногу
                        </button>
                        <button
                          type="button"
                          className="lab-menu-item danger"
                          onClick={() => { clearAllContacts(); setLabMenuOpen(false) }}
                        >
                          Очистить всё
                        </button>
                        <button
                          type="button"
                          className="lab-menu-item danger"
                          disabled={!selectedMarkup}
                          onClick={() => { deleteSelectedMarkup(); setLabMenuOpen(false) }}
                        >
                          Удалить выбранный интервал
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {selectedMarkup && (() => {
                  const contacts = selectedMarkup.foot === 'left' ? leftContacts : rightContacts
                  const pairStart = getPairStartIndex(selectedMarkup.index)
                  const t0 = contacts[pairStart]
                  if (t0 == null) return null
                  const footLabel = selectedMarkup.foot === 'left' ? 'S1' : 'S2'
                  const intervalNum = Math.floor(pairStart / 2) + 1
                  const t1 = contacts[pairStart + 1]
                  const hasPair = pairStart + 1 < contacts.length
                  const fmt = (t) => timeUnit === 'ms' ? `${t.toFixed(0)} мс` : `${t.toFixed(3)} с`
                  const chartCursorTime = currentTime * (timeUnit === 'ms' ? 1000 : 1)

                  const setStartToCursor = () => {
                    const setter = selectedMarkup.foot === 'left' ? setLeftContacts : setRightContacts
                    setter(prev => {
                      const next = [...prev]
                      if (pairStart < next.length) next[pairStart] = chartCursorTime
                      return next
                    })
                  }
                  const setEndToCursor = () => {
                    const setter = selectedMarkup.foot === 'left' ? setLeftContacts : setRightContacts
                    setter(prev => {
                      const next = [...prev]
                      if (pairStart + 1 < next.length) next[pairStart + 1] = chartCursorTime
                      return next
                    })
                  }

                  return (
                    <div className="selected-edit-controls">
                      <span className="lab-stat lab-stat-selected">
                        {footLabel} #{intervalNum}
                        {hasPair ? `: ${fmt(t0)} → ${fmt(t1)}` : `: ${fmt(t0)} (1 точка)`}
                      </span>
                      <div className="btn-group edit-btns">
                        <button
                          type="button"
                          className="btn-secondary btn-xs-edit"
                          onClick={setStartToCursor}
                          title={`Установить начало интервала на текущее время курсора (${fmt(chartCursorTime)})`}
                        >
                          ⏱ Старт в маркер
                        </button>
                        {hasPair && (
                          <button
                            type="button"
                            className="btn-secondary btn-xs-edit"
                            onClick={setEndToCursor}
                            title={`Установить конец интервала на текущее время курсора (${fmt(chartCursorTime)})`}
                          >
                            ⏱ Конец в маркер
                          </button>
                        )}
                        <button
                          type="button"
                          className={`btn-secondary btn-xs-edit${relabelStep ? ' active-relabel' : ''}`}
                          onClick={() => setRelabelStep(relabelStep ? null : 'start')}
                          title="Изменить границы интервала двумя последовательными кликами на графике"
                        >
                          {relabelStep 
                            ? (relabelStep === 'start' ? '📍 Кликните начало...' : '📍 Кликните конец...') 
                            : '🖱 Переразметить кликами'
                          }
                        </button>
                      </div>
                    </div>
                  )
                })()}
              </div>
            )}

            {labelingMode && (leftContacts.length > 0 || rightContacts.length > 0) && (
              <div className="zone-dur-block">
                {[
                  { contacts: leftContacts, cls: 'zone-dur-s1', label: 'S1', foot: 'left' },
                  { contacts: rightContacts, cls: 'zone-dur-s2', label: 'S2', foot: 'right' },
                ].map(({ contacts, cls, label, foot }) => contacts.length > 0 && (
                  <div key={label} className="zone-dur-row">
                    <span className={`zone-dur-label ${cls}`}>{label}</span>
                    {Array.from({ length: Math.floor(contacts.length / 2) }, (_, i) => {
                      const t0 = contacts[i * 2]
                      const t1 = contacts[i * 2 + 1]
                      const dur = Math.abs(t1 - t0)
                      const sel = selectedMarkup?.foot === foot && selectedMarkup.index === i * 2
                      return (
                        <span
                          key={i}
                          className={`zone-dur-chip ${cls}${sel ? ' zone-dur-selected' : ''}`}
                          title={`${t0.toFixed(2)} → ${t1.toFixed(2)}`}
                          onClick={() => {
                            if (selectedMarkup?.foot === foot && selectedMarkup.index === i * 2) {
                              setSelectedMarkup(null)
                            } else {
                              setSelectedMarkup({ foot, index: i * 2 })
                            }
                          }}
                        >
                          #{i + 1}&thinsp;{formatDuration(dur, timeUnit)}
                        </span>
                      )
                    })}
                    {contacts.length % 2 === 1 && (
                      <span
                        className={`zone-dur-chip zone-dur-pending${
                          selectedMarkup?.foot === foot && selectedMarkup.index === contacts.length - 1
                            ? ' zone-dur-selected' : ''
                        }`}
                        onClick={() => {
                          if (selectedMarkup?.foot === foot && selectedMarkup.index === contacts.length - 1) {
                            setSelectedMarkup(null)
                          } else {
                            setSelectedMarkup({ foot, index: contacts.length - 1 })
                          }
                        }}
                      >
                        …2-я точка
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="chart-area" ref={chartAreaRef}>
            <div ref={chartDivRef} style={{ width: '100%', height: '100%' }} />
            {chartReady && selectedCols.length > 1 && (
              <div className="chart-reorder-layer" aria-label="Изменение порядка графиков">
                {selectedCols.map((col, index) => (
                  <button
                    key={col}
                    type="button"
                    className={`chart-reorder-handle${chartReorder?.fromIndex === index ? ' dragging' : ''}${chartReorder?.targetIndex === index ? ' drop-target' : ''}`}
                    style={{ top: chartSubplotCenterTop(index, selectedCols.length) }}
                    onPointerDown={(event) => beginChartReorder(event, index)}
                    onPointerMove={updateChartReorder}
                    onPointerUp={(event) => finishChartReorder(event)}
                    onPointerCancel={(event) => finishChartReorder(event, true)}
                    onKeyDown={(event) => {
                      if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
                      event.preventDefault()
                      const target = event.key === 'ArrowUp'
                        ? Math.max(0, index - 1)
                        : Math.min(selectedCols.length - 1, index + 1)
                      moveSelectedColumn(index, target)
                    }}
                    aria-label={`Перетащить график ${col}`}
                    title={`Перетащить график ${col} вверх или вниз`}
                  >
                    <span className="chart-reorder-arrows" aria-hidden="true">↕</span>
                    <UiIcon name="grip" />
                    <span className="chart-reorder-hint" aria-hidden="true">Перетащить</span>
                  </button>
                ))}
                {chartReorder && chartReorderTargetMetrics && (
                  <>
                    <div
                      className="chart-reorder-origin"
                      style={{ top: chartReorder.sourceTop, height: chartReorder.sourceHeight }}
                    />
                    {chartReorder.targetIndex !== chartReorder.fromIndex && (
                      <div
                        className="chart-reorder-drop-zone"
                        style={{ top: chartReorderTargetMetrics.top, height: chartReorderTargetMetrics.height }}
                      />
                    )}
                    <div
                      className="chart-reorder-ghost"
                      style={{
                        top: chartReorder.pointerY - chartReorder.areaTop - chartReorder.pointerOffsetY,
                        height: chartReorder.sourceHeight,
                      }}
                    >
                      {chartReorder.previewUrl && (
                        <img
                          src={chartReorder.previewUrl}
                          alt=""
                          draggable={false}
                          className="chart-reorder-ghost-image"
                          style={{
                            top: -chartReorder.sourceTop,
                            width: chartReorder.chartWidth,
                            height: chartReorder.chartHeight,
                          }}
                        />
                      )}
                      <span className="chart-reorder-ghost-label">
                        <UiIcon name="grip" /> {chartReorder.col}
                      </span>
                    </div>
                  </>
                )}
              </div>
            )}
            {!chartReady && (
              <div className="chart-empty">
                {parquetData
                  ? <><span>📊</span><p>Выберите хотя бы одну колонку — график построится автоматически</p></>
                  : <><span>📊</span><p>Загрузите <b>.parquet</b>-файл или введите номер сессии</p></>
                }
              </div>
            )}
          </div>
        </div>
      </div>
        </div>
      </div>

      {dragOver && (
        <div className="drag-overlay">
          <div className="drag-box">⬇<p>Видео, .parquet или размеченный .csv</p></div>
        </div>
      )}

      <nav className="mobile-tabbar" aria-label="Разделы">
        <button
          type="button"
          className={mobileTab === 'data' ? 'active' : ''}
          onClick={() => setMobileTab('data')}
        >
          <UiIcon name="database" />
          <span>Данные</span>
        </button>
        <button
          type="button"
          className={mobileTab === 'video' ? 'active' : ''}
          onClick={() => setMobileTab('video')}
        >
          <UiIcon name="video" />
          <span>Видео</span>
        </button>
        <button
          type="button"
          className={mobileTab === 'chart' ? 'active' : ''}
          onClick={() => setMobileTab('chart')}
        >
          <UiIcon name="chart" />
          <span>График</span>
        </button>
      </nav>
    </div>
  )
}

function SidebarSection({ title, open, onToggle, children }) {
  return (
    <section className="sidebar-section">
      <button type="button" className="sidebar-section-head" onClick={onToggle}>
        <span className="sidebar-section-title">{title}</span>
        <span className="sidebar-section-chevron">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className="sidebar-section-body">{children}</div>}
    </section>
  )
}

function UploadBtn({ accept, onFile, children, className = 'btn-upload btn-secondary', disabled = false, title }) {
  return (
    <label
      className={`${className}${disabled ? ' disabled' : ''}`}
      title={title}
      style={disabled ? { opacity: 0.4, pointerEvents: 'none', cursor: 'not-allowed' } : undefined}
    >
      {children}
      <input
        type="file"
        accept={accept}
        hidden
        disabled={disabled}
        onChange={e => {
          if (e.target.files[0]) onFile(e.target.files[0])
          e.target.value = ''
        }}
      />
    </label>
  )
}

function UiIcon({ name, className = '' }) {
  let artwork

  switch (name) {
    case 'video':
      artwork = <><rect x="3" y="6" width="13" height="12" rx="2" /><path d="m16 10 5-3v10l-5-3z" /></>
      break
    case 'database':
      artwork = <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" /></>
      break
    case 'database-check':
      artwork = <><ellipse cx="10" cy="5" rx="7" ry="3" /><path d="M3 5v6c0 1.7 3.1 3 7 3h1" /><path d="M3 11v6c0 1.6 2.8 2.8 6.4 3" /><path d="m14 17 2 2 5-6" /></>
      break
    case 'file-table':
      artwork = <><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v5h5" /><path d="M9 12h6M9 16h6M12 11v6" /></>
      break
    case 'download':
      artwork = <><path d="M12 3v12" /><path d="m7 10 5 5 5-5" /><path d="M5 21h14" /></>
      break
    case 'upload':
      artwork = <><path d="M12 21V9" /><path d="m7 14 5-5 5 5" /><path d="M5 3h14" /></>
      break
    case 'pencil':
      artwork = <><path d="m4 20 4.2-1 10.9-10.9a2.1 2.1 0 0 0-3-3L5.2 16z" /><path d="m14.8 6.4 3 3" /></>
      break
    case 'gaps':
      artwork = <><path d="M3 12h6M15 12h6" /><path d="m10 8 4 8M14 8l-4 8" /></>
      break
    case 'undo':
      artwork = <><path d="m8 7-5 5 5 5" /><path d="M3 12h10a6 6 0 0 1 6 6v1" /></>
      break
    case 'rotate':
      artwork = <><path d="M20 7v5h-5" /><path d="M19 12a7 7 0 1 0-2 5" /></>
      break
    case 'grip':
      artwork = <><circle cx="9" cy="6" r="1" fill="currentColor" stroke="none" /><circle cx="15" cy="6" r="1" fill="currentColor" stroke="none" /><circle cx="9" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="15" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="9" cy="18" r="1" fill="currentColor" stroke="none" /><circle cx="15" cy="18" r="1" fill="currentColor" stroke="none" /></>
      break
    case 'pin':
      artwork = <><path d="M12 17v5" /><path d="M9 3h6l1 7 3 3H5l3-3z" /></>
      break
    case 'chart':
      artwork = <><path d="M4 19h16" /><path d="M7 16v-6" /><path d="M12 16V8" /><path d="M17 16v-9" /></>
      break
    case 'check':
      artwork = <path d="m5 13 4 4 10-11" />
      break
    case 'tag':
      artwork = <><path d="M11 3H4v7l10 10 7-7z" /><circle cx="7.5" cy="6.5" r="1.2" /></>
      break
    case 'user':
      artwork = <><circle cx="12" cy="8" r="4" /><path d="M5 21v-1.5A4.5 4.5 0 0 1 9.5 15h5a4.5 4.5 0 0 1 4.5 4.5V21" /></>
      break
    case 'logout':
      artwork = <><path d="M10 5H5v14h5" /><path d="M13 8l4 4-4 4M8 12h9" /></>
      break
    case 'plus':
      artwork = <path d="M12 5v14M5 12h14" />
      break
    case 'minus':
      artwork = <path d="M5 12h14" />
      break
    case 'maximize':
      artwork = <><path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5" /><path d="m4 9 5-5M20 9l-5-5M4 15l5 5M20 15l-5 5" /></>
      break
    case 'bolt':
      artwork = <path d="m13 2-8 12h7l-1 8 8-12h-7z" />
      break
    case 'ruler':
      artwork = <><path d="m4 15 11-11 5 5-11 11H4z" /><path d="m12 7 2 2M9 10l2 2M6 13l2 2" /></>
      break
    case 'mirror':
      artwork = <><path d="M12 3v18" strokeDasharray="3 3" /><path d="M9 8 4 12l5 4z" /><path d="m15 8 5 4-5 4z" /></>
      break
    case 'x':
      artwork = <path d="m6 6 12 12M18 6 6 18" />
      break
    case 'loader':
      artwork = <><circle cx="12" cy="12" r="9" opacity=".25" /><path d="M21 12a9 9 0 0 0-9-9" /></>
      break
    default:
      artwork = <circle cx="12" cy="12" r="8" />
  }

  return (
    <svg
      className={`ui-icon${name === 'loader' ? ' ui-icon-spin' : ''}${className ? ` ${className}` : ''}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      {artwork}
    </svg>
  )
}

// The title is free text an athlete typed on their phone, so it can be far
// wider than the header allows: it stays clamped to one line until clicked,
// and the expanded form doubles as the editor. The caller keys it on the
// session and the title, so a save or a session switch remounts it and the
// draft never survives into a different title.
function SessionTitleBadge({ title, expanded, onToggle, onSave }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(title)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing])

  const startEditing = () => {
    setDraft(title)
    setError('')
    setEditing(true)
  }

  const commit = async () => {
    const next = draft.trim().slice(0, 255)
    if (next === title) { setEditing(false); return }
    setSaving(true)
    setError('')
    try {
      await onSave(next)
      setEditing(false)
    } catch (err) {
      setError(err.message || 'Не удалось сохранить')
    } finally {
      setSaving(false)
    }
  }

  const cancel = () => {
    setDraft(title)
    setError('')
    setEditing(false)
  }

  if (editing) {
    return (
      <span className="file-badge badge-title badge-expanded badge-title-edit">
        <UiIcon name="tag" />
        <input
          ref={inputRef}
          className="badge-title-input"
          value={draft}
          maxLength={255}
          placeholder="Название сессии"
          disabled={saving}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') { e.preventDefault(); commit() }
            if (e.key === 'Escape') { e.preventDefault(); cancel() }
          }}
        />
        <button
          type="button"
          className="badge-title-act"
          onClick={commit}
          disabled={saving}
          title="Сохранить (Enter)"
        >
          <UiIcon name={saving ? 'loader' : 'check'} />
        </button>
        <button
          type="button"
          className="badge-title-act"
          onClick={cancel}
          disabled={saving}
          title="Отмена (Esc)"
        >
          <UiIcon name="x" />
        </button>
        {error && <span className="badge-title-error">{error}</span>}
      </span>
    )
  }

  return (
    <span
      className={`file-badge badge-title${expanded ? ' badge-expanded' : ''}`}
    >
      <button
        type="button"
        className="badge-title-text"
        onClick={onToggle}
        title={expanded ? 'Свернуть' : (title || 'Название не задано')}
        aria-expanded={expanded}
      >
        <UiIcon name="tag" />
        <span className={title ? '' : 'badge-title-empty'}>{title || 'Пусто'}</span>
      </button>
      <button
        type="button"
        className="badge-title-act"
        onClick={startEditing}
        title="Изменить название"
      >
        <UiIcon name="pencil" />
      </button>
    </span>
  )
}

function FileBadge({ type, title, children }) {
  return <span className={`file-badge badge-${type}`} title={title}>{children}</span>
}

function SessionInfoCard({ protocolName, deviceId, gapCount, gapsKnown }) {
  const gapsLabel = !gapsKnown ? '—' : gapCount > 0 ? String(gapCount) : 'нет'
  return (
    <div className="session-info-card" aria-label="Данные сессии">
      <div className="session-info-row">
        <span className="session-info-key">Протокол</span>
        <span className="session-info-val">{protocolName || '—'}</span>
      </div>
      <div className="session-info-row">
        <span className="session-info-key">Device ID</span>
        <span className="session-info-val">{hasSessionMetaValue(deviceId) ? String(deviceId) : '—'}</span>
      </div>
      <div className="session-info-row">
        <span className="session-info-key">Пропуски</span>
        <span className={`session-info-val${gapsKnown && gapCount > 0 ? ' session-info-warn' : ''}`}>
          {gapsLabel}
        </span>
      </div>
    </div>
  )
}

function OffsetInput({ value, step, title, onChange }) {
  const [draft, setDraft] = useState(String(value))
  const committed = useRef(value)

  useEffect(() => {
    if (committed.current !== value) {
      committed.current = value
      setDraft(String(value))
    }
  }, [value])

  const commit = (raw) => {
    const trimmed = raw.trim()
    const n = Number(trimmed)
    if (trimmed !== '' && isFinite(n)) {
      committed.current = n
      onChange(n)
      setDraft(String(n))
    } else {
      setDraft(String(committed.current))
    }
  }

  const nudge = (dir) => {
    const base = isFinite(Number(draft)) ? Number(draft) : committed.current
    const next = Math.round((base + dir * step) * 1e9) / 1e9
    committed.current = next
    onChange(next)
    setDraft(String(next))
  }

  return (
    <div className="offset-input-wrap">
      <input
        type="text"
        inputMode="numeric"
        className="input-sm offset-input-field"
        value={draft}
        title={title}
        aria-label={title}
        onChange={e => setDraft(e.target.value)}
        onBlur={e => commit(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter')     { e.preventDefault(); commit(draft) }
          if (e.key === 'ArrowUp')   { e.preventDefault(); nudge(+1) }
          if (e.key === 'ArrowDown') { e.preventDefault(); nudge(-1) }
        }}
      />
      <div className="offset-spinners">
        <button
          type="button"
          className="offset-spin-btn"
          aria-label={`${title}: увеличить`}
          onMouseDown={e => e.preventDefault()}
          onClick={() => nudge(+1)}
        >▲</button>
        <button
          type="button"
          className="offset-spin-btn"
          aria-label={`${title}: уменьшить`}
          onMouseDown={e => e.preventDefault()}
          onClick={() => nudge(-1)}
        >▼</button>
      </div>
    </div>
  )
}
