// Sensor naming, foot mapping and the column-map shape of a loaded session.
import { ST_COL_NAMES } from './chart.js'
import { safeNum } from './utils.js'

export const NON_DATA_COLS = new Set([
  'Name', 'Time', 'time', 'timestamp', 'Timestamp', 't',
  'target', 'Target', 'label', 'Label',
])

export const PREFERRED_COLS = ['AcX', 'AcY', 'AcZ', 'XData', 'YData', 'ZData', 'GravityZ']

export const SPEED_TRACKER = 'ESP32_SpeedTracker'

// Insole pressure channels and the device-name → foot mapping used for
// per-foot calibration/normalization (mirrors the backend: ESP32_Sensor_1 is
// the left insole, ESP32_Sensor_2 the right).
export const SENSOR_COLS = ['Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']

export const SENSOR_NAME_TO_FOOT = { ESP32_Sensor_1: 'left', ESP32_Sensor_2: 'right' }

export function inferSensorFoot(name) {
  if (!name) return null
  if (SENSOR_NAME_TO_FOOT[name]) return SENSOR_NAME_TO_FOOT[name]

  const normalized = String(name).trim().toLowerCase()
  if (/(^|[_\s-])left($|[_\s-])/.test(normalized)) return 'left'
  if (/(^|[_\s-])right($|[_\s-])/.test(normalized)) return 'right'
  if (/^(?:esp32_)?sensor[_\s-]*1(?:\D|$)/.test(normalized)) return 'left'
  if (/^(?:esp32_)?sensor[_\s-]*2(?:\D|$)/.test(normalized)) return 'right'
  return null
}

export function groupSensorNamesByFoot(names) {
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

export function sensorFootForName(name, names) {
  const inferred = inferSensorFoot(name)
  if (inferred) return inferred
  const groups = groupSensorNamesByFoot(names)
  if (groups.left.includes(name)) return 'left'
  if (groups.right.includes(name)) return 'right'
  return null
}

export function sensorNameForFoot(names, foot) {
  return groupSensorNamesByFoot(names)[foot]?.[0] || ''
}

export function rowsToColMap(rows) {
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

export function detectTimeCol(allCols) {
  return allCols.find(c => c === 'Time')
    || allCols.find(c => ['time', 'timestamp', 'Timestamp', 't'].includes(c))
    || allCols[0]
}

export function computeNumericColumns(colMap, tCol) {
  return Object.keys(colMap).filter(c => {
    if (NON_DATA_COLS.has(c) || c === tCol) return false
    return (colMap[c] || []).some(v => safeNum(v) !== null)
  })
}

export function sortSensorNames(colMap) {
  if (!colMap['Name']) return []
  return [...new Set(colMap['Name'].filter(v => v != null && v !== ''))]
    .sort((a, b) => a.localeCompare(b))
}

export function computeAutoOffsetST(colMap, timeCol, insoleNames) {
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

export function resolveStDataCol(data, col) {
  if ((data[col] || []).some(v => safeNum(v) !== null)) return col
  const alt = { Distance: 'DistanceM', Speed: 'VelocityMs', DistanceM: 'Distance', VelocityMs: 'Speed' }[col]
  if (alt && (data[alt] || []).some(v => safeNum(v) !== null)) return alt
  return col
}

export function buildDefaultCols(numCols, hasSpeedTracker, colMap, sensorNames) {
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
