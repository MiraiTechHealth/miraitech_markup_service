// Plotly layout constants and shape builders for the session chart.

export const UI_FONT_FAMILY = 'Inter, "Segoe UI Variable", "Segoe UI", Arial, sans-serif'

export const PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
  '#9467bd', '#8c564b', '#e377c2', '#17becf',
]

export const ST_COLOR = '#2ca02c'

export const ST_COL_NAMES = ['Distance', 'Speed', 'DistanceM', 'VelocityMs']

export const ST_ONLY_COLS = new Set(ST_COL_NAMES)

// Distinct colours per SpeedTracker column so Distance and Speed don't look alike.
export const ST_COL_COLORS = {
  Speed: '#2ca02c',       // green — keeps the SpeedTracker brand colour
  VelocityMs: '#2ca02c',
  Distance: '#9467bd',    // purple
  DistanceM: '#9467bd',
}

// Speed/Distance-predict overlays (charts/sprint): drawn on top of their
// respective subplots. Both read from the same fetched series.
export const SPEED_PRED_COLS = new Set(['Speed', 'VelocityMs'])

export const DISTANCE_PRED_COLS = new Set(['Distance', 'DistanceM'])

export const PRED_COLOR = '#d62728'

export const TRACE_HOVER_TEMPLATE = '<b>%{fullData.name}</b><br>Время: %{x}<br>Значение: %{y:.4g}<extra></extra>'

export const L_FILL = 'rgba(31,119,180,0.35)'

export const R_FILL = 'rgba(255,127,14,0.35)'

export const L_LINE = 'rgba(31,119,180,0.9)'

export const R_LINE = 'rgba(255,127,14,0.9)'

export const GAP_FILL = 'rgba(220,53,69,0.35)'

export const GAP_LINE = 'rgba(220,53,69,0.92)'

export const SEL_FILL = 'rgba(234,179,8,0.5)'

export const SEL_LINE = '#ca8a04'

export function buildGapBandShapes(intervals, nSubplots) {
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

export function buildCursorShapes(x, n) {
  return Array.from({ length: n }, (_, i) => ({
    type: 'line',
    x0: x, x1: x,
    y0: 0, y1: 1,
    xref: i === 0 ? 'x' : `x${i + 1}`,
    yref: i === 0 ? 'y domain' : `y${i + 1} domain`,
    line: { color: 'rgba(220,40,40,0.85)', width: 2, dash: 'dot' },
  }))
}

export function buildSelectedPointShapes(x, n) {
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

export function chartSubplotCenterTop(index, total) {
  if (total <= 0) return '50%'
  const gap = 0.03
  const subplotHeight = (1 - gap * (total - 1)) / total
  const centerDomain = 1 - index * (subplotHeight + gap) - subplotHeight / 2
  return `calc(12px + (100% - 54px) * ${1 - centerDomain})`
}

export function chartSubplotMetrics(index, total, chartHeight) {
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

export function plotAxisKey(index, axis) {
  return index === 0 ? axis : `${axis}${index + 1}`
}

export function readPlotRange(eventData, axisKey) {
  if (!eventData) return null
  const start = eventData[`${axisKey}.range[0]`]
  const end = eventData[`${axisKey}.range[1]`]
  if (start !== undefined && end !== undefined) return [Number(start), Number(end)]
  const nested = eventData[`${axisKey}.range`] ?? eventData[axisKey]?.range
  if (Array.isArray(nested) && nested.length >= 2) return [Number(nested[0]), Number(nested[1])]
  return null
}

export function currentAxisRange(gd, axisKey) {
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
