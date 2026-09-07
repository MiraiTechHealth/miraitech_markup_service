// Markup CSV parsing: Target runs -> contact pairs.
import { safeNum } from './utils.js'

export function getPairStartIndex(index) {
  return index - (index % 2)
}

export function parseCsvRow(line) {
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

export function parseCsvText(text) {
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
export function coerceCsvColumnsToNumbers(colMap) {
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

export function isTargetOne(v) {
  if (v === 1 || v === '1' || v === 1.0) return true
  const s = String(v ?? '').trim()
  return s === '1' || s === '1.0'
}

export function extractContactPairsFromTargetRuns(times, targets, offset = 0) {
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

export function extractContactsFromLabeledCsv(rows, timeCol, leftSensors, rightSensors, offsetS1, offsetS2) {
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
