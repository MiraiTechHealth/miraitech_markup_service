// The model / detector cards: ids, columns they read, event styles, summaries.

// Virtual subplot for the total GRF model. Its prediction has no raw column to
// live on, so a panel is created for it the way `Speed` is created for speed
// predict: the name goes into selectedCols and the traces are drawn onto it.
export const GRF_PRED_COL = 'GRF %BW'

// Weight is a conditioning input of the force model, so it cannot be defaulted.
export const WEIGHT_REQUIRED_CALCULATORS = new Set(['grf-split'])

export const EXTRA_CALCULATORS = [
  {
    id: 'step-detector-ttest',
    label: 'T-тест · Step detector',
    description: 'Детектирует шаги левой и правой ноги по давлению Sensor 1 + Sensor 2',
    color: '#7c3aed',
    fill: 'rgba(124,58,237,0.10)',
  },
  {
    id: 'jump-events',
    label: 'Jump events · по платформам',
    description: 'Отрыв и приземление по всему телу (не по ногам) · TCN, обучен по силовым платформам',
    color: '#0d9488',
    fill: 'rgba(13,148,136,0.10)',
  },
  {
    id: 'grf-split',
    label: 'Total GRF · по платформам',
    description: 'Кривая суммарной вертикальной силы (обе стопы) в %BW · обучена по двум '
      + 'силовым платформам · цель — плита с low-pass 20 Гц, сравнивать с колонкой '
      + 'Plate_Fz_total_lp20_pctBW; против сырой плиты пик приземления ниже ~10% по определению',
    color: '#7c3aed',
    fill: 'rgba(124,58,237,0.10)',
  },
]

export const PROTOCOL_DETECTORS = [
  {
    id: 'protocol-walking-detector',
    label: 'Тест ходьбы · Step detector',
    description: 'Определяет интервалы контакта стоп при ходьбе',
    color: '#2563eb',
    fill: 'rgba(37,99,235,0.10)',
  },
  {
    id: 'protocol-shuttle-detector',
    label: 'Челночный бег · Turn detector',
    description: 'Определяет повороты и беговые отрезки',
    color: '#d97706',
    fill: 'rgba(217,119,6,0.10)',
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

export const EXTRA_CALCULATOR_BY_ID = Object.fromEntries(EXTRA_CALCULATORS.map(calc => [calc.id, calc]))

export const PROTOCOL_DETECTOR_BY_ID = Object.fromEntries(PROTOCOL_DETECTORS.map(detector => [detector.id, detector]))

// Every card in the «Детекторы и модели» section - protocol detectors and the
// model calculators together. The plate ground truth is the one calculator that
// lives elsewhere (chart toolbar), so it is not counted here.
export const MODEL_SECTION_CALCULATOR_IDS = new Set([
  ...PROTOCOL_DETECTORS.map(detector => detector.id),
  ...EXTRA_CALCULATORS.map(calculator => calculator.id),
])

// Force-plate ground truth of the jump detector. Not a calculator card: it lives
// as a toggle in the chart toolbar (beside «Углы» / «Дрейф»), because it is a
// property of the loaded data, not a model - it exists only when the session
// carries the plate force. Registered here so the shared overlay / click-to-
// inspect machinery treats its bilateral segments like any calculator's.
export const PLATE_FLIGHT_ID = 'plate-flight'

export const PLATE_FORCE_COLUMNS = ['Plate_Fz_N', '1:Fz', '2:Fz']

export const PLATE_FLIGHT_CALCULATOR = {
  id: PLATE_FLIGHT_ID,
  label: 'Полёт по плитам · разметка v7',
  description: 'Ground truth прыжков по силовым платформам: обе плиты < 20 Н, '
    + 'гейт «удар ИЛИ свободное падение», прыжки через край плиты — маска; '
    + 'та же разметка, на которой обучается Jump events',
  color: '#15803d',
  fill: 'rgba(21,128,61,0.16)',
}

export const CALCULATOR_BY_ID = {
  ...EXTRA_CALCULATOR_BY_ID,
  ...PROTOCOL_DETECTOR_BY_ID,
  [PLATE_FLIGHT_ID]: PLATE_FLIGHT_CALCULATOR,
}

export const PER_FOOT_TURN_DETECTOR_IDS = new Set([
  'protocol-shuttle-detector',
  'protocol-beep-detector',
  'protocol-ttest-detector',
])

// Columns a calculator actually reads, for the ones where the rest of the export
// is dead weight in the request body. A force-plate session carries 39 columns
// and jump-events reads 12 of them, so sending everything tripled the upload
// (35.5 MB -> 10.3 MB on a 121k-row session). Keep in step with what the server
// reads: _foot_frames in new_jump_model_byAdil_calculator.py (ACC_COLS,
// ANGLE_COLS, PRESS_COLS) plus Name/Time. A column listed here but absent from
// the session is simply left out, and the server logs the channels it missed.
export const JUMP_MODEL_COLUMNS = [
  'Name', 'Time',
  'AcX', 'AcY', 'AcZ',
  'XData', 'YData', 'ZData',
  'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4',
]

export const CALCULATOR_COLUMNS = {
  'jump-events': JUMP_MODEL_COLUMNS,
  // The GRF regressor builds its features from the same channels (_foot_frames in
  // jump_grf_split_calculator.py: ACC_COLS, ANG_COLS, PRESS_COLS) and calls the
  // jump detector on the same frame.
  'grf-split': JUMP_MODEL_COLUMNS,
  // The labeler reads |a| (free fall + impact), the insole sum (foot on the
  // floor beside the plate) and the plate force itself, in newtons.
  [PLATE_FLIGHT_ID]: [
    'Name', 'Time',
    'AcX', 'AcY', 'AcZ',
    'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4',
    ...PLATE_FORCE_COLUMNS,
  ],
}

// The whole colMap unless the calculator declared a narrower set above.
export function columnsForCalculator(calculatorId, colMap) {
  const wanted = CALCULATOR_COLUMNS[calculatorId]
  if (!wanted || !colMap) return colMap
  const out = {}
  wanted.forEach(column => { if (colMap[column]) out[column] = colMap[column] })
  return out
}

// The calculator inputs as a query string, for the request that carries the
// session as parquet bytes instead of JSON. Lists (jump_pairs) go as JSON text.
export function calculatorQuery(options) {
  const query = new URLSearchParams()
  Object.entries(options).forEach(([key, value]) => {
    if (value == null || value === '') return
    query.set(key, Array.isArray(value) ? JSON.stringify(value) : String(value))
  })
  return query.toString()
}

export const EVENT_STYLE_BY_KIND = {
  run: {
    label: 'Беговая фаза', color: '#16a34a', fill: 'rgba(22,163,74,0.10)', dash: 'dash', width: 1.5,
  },
  turn: {
    label: 'Поворот', color: '#f97316', fill: 'rgba(249,115,22,0.18)', dash: 'solid', width: 2.25,
  },
  // Plate ground truth: a jump the labeler accepted, and a segment it refused to
  // call either flight or ground (no impact, no free fall, or an insole loaded
  // beside the plate) - the model is neither taught nor scored on the grey ones.
  plate_flight: {
    label: 'Полёт по плитам', color: '#15803d', fill: 'rgba(21,128,61,0.16)', dash: 'solid', width: 2,
  },
  plate_mask: {
    label: 'Маска: неизвестно', color: '#6b7280', fill: 'rgba(107,114,128,0.16)', dash: 'dot', width: 1.25,
  },
}

export const TURN_EVENT_STYLE_BY_FOOT = {
  left: {
    run: { color: '#2563eb', fill: 'rgba(37,99,235,0.11)', dash: 'dash', width: 1.75 },
    turn: { color: '#db2777', fill: 'rgba(219,39,119,0.20)', dash: 'solid', width: 2.5 },
  },
  right: {
    run: { color: '#0f766e', fill: 'rgba(15,118,110,0.11)', dash: 'dash', width: 1.75 },
    turn: { color: '#f97316', fill: 'rgba(249,115,22,0.20)', dash: 'solid', width: 2.5 },
  },
}

export const FOOT_EVENT_STYLE = {
  left: { color: '#2563eb', fill: 'rgba(37,99,235,0.13)', dash: 'dash', width: 1.5 },
  right: { color: '#f97316', fill: 'rgba(249,115,22,0.13)', dash: 'dot', width: 1.5 },
}

// ── Target steps ──────────────────────────────────────────────────────────
// Target is the model's step markup after the operator has been over it: every
// detected contact except the ones struck out by hand, plus the intervals
// placed by hand in markup mode. A deletion is remembered by the step's own
// times rather than its index, so re-running the same detector does not
// resurrect a step the operator already threw away.
const STEP_CONTACT_KINDS = new Set(['contact', 'step'])
export const isStepContact = (contact) => STEP_CONTACT_KINDS.has(contact?.kind)

export function stepKey(calculatorId, contact) {
  const start = Number(contact?.start_time_s)
  const end = Number(contact?.end_time_s)
  return [
    calculatorId,
    contact?.foot || 'all',
    Number.isFinite(start) ? start.toFixed(4) : 'na',
    Number.isFinite(end) ? end.toFixed(4) : 'na',
  ].join('|')
}

/** Struck-out steps stay on the chart as a grey ghost so they can be put back. */
export const DELETED_STEP_STYLE = {
  color: '#94a3b8', fill: 'rgba(148,163,184,0.04)', dash: 'dot', width: 1,
}

export function calculatorEventStyle(calculator, contact) {
  const turnFootStyle = TURN_EVENT_STYLE_BY_FOOT[contact?.foot]?.[contact?.kind]
  if (turnFootStyle) return turnFootStyle
  const semanticStyle = EVENT_STYLE_BY_KIND[contact?.kind]
  if (semanticStyle) return semanticStyle
  if (FOOT_EVENT_STYLE[contact?.foot]) return FOOT_EVENT_STYLE[contact.foot]
  return {
    color: calculator?.color || '#64748b',
    fill: calculator?.fill || 'rgba(100,116,139,0.10)',
    dash: 'dash',
    width: 1.25,
  }
}

export function calculatorEventLegend(calculator, result) {
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
        : kind === 'flight'
            ? `Прыжок${footSuffix}`
            : kind === 'plate_flight'
              ? 'Полёт по плитам'
            : kind === 'plate_mask'
              ? 'Маска: неизвестно'
            : kind === 'step'
              ? `Шаг ${foot}`
              : ['contact', 'plateau'].includes(kind)
                ? `Контакт ${foot}`
                : `Событие ${foot}`.trim()
    items.set(key, { key, label, ...style })
  })
  return [...items.values()]
}

// Движение — это входной канал модели (one-hot), поэтому одна и та же сессия
// под разным протоколом даёт разные события. Оператор выбирает его сам;
// «вертикальный» по умолчанию — самый частый случай разметки и протокол, на
// котором модель сильнее всего (F1@20мс 0.963/0.973 против 0.896/0.852 в среднем).
export const JUMP_EVENT_PROTOCOL_OPTIONS = [
  { value: 'vert', label: 'Вертикальный' },
  { value: 'fwd_sl', label: 'SL вперёд' },
  { value: 'side_sl', label: 'SL вбок' },
  { value: 'sl_hop', label: 'SL hopping' },
  { value: 'mv3', label: 'Движение 3' },
  { value: 'mv5', label: 'Движение 5' },
  { value: 'mv6', label: 'Движение 6' },
]

export const TURN_DETECTION_FOOT_OPTIONS = [
  { value: 'both', label: 'L+R', title: 'Наложить независимые детекции левой и правой ног' },
  { value: 'left', label: 'L', title: 'Детектировать только по левой ноге' },
  { value: 'right', label: 'R', title: 'Детектировать только по правой ноге' },
]

export function protocolDetectorSummary(result) {
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
  return `контакты ${summary.contact_count || 0} · L ${summary.left_count || 0} · R ${summary.right_count || 0}`
}

export function normaliseSpeedPrediction(data) {
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
