import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import Plotly from 'plotly.js-basic-dist-min'
import { parquetReadObjects } from 'hyparquet'
import './App.css'
import { FileBadge, OffsetInput, SessionInfoCard, SessionTitleBadge, SidebarSection, UiIcon, UploadBtn } from './components/ui.jsx'
import { ACTIVITY_BY_ID, ACTIVITY_FILE_TYPE, ACTIVITY_KINDS, ACTIVITY_SNAP_FRACTION, DEFAULT_ACTIVITY, activitySpanAt, activitySpansFromFiles, closeActivitySpan, contactMarkupFiles, insertActivitySpan, resizeActivitySpan, snapActivityEdge } from './lib/activity.js'
import { API_BASE, CALCULATOR_API, MARKUP_API, parseApiError } from './lib/api.js'
import { CALCULATOR_BY_ID, DELETED_STEP_STYLE, EXTRA_CALCULATORS, GRF_PRED_COL, JUMP_EVENT_PROTOCOL_OPTIONS, MODEL_SECTION_CALCULATOR_IDS, PER_FOOT_TURN_DETECTOR_IDS, PLATE_FLIGHT_ID, PLATE_FORCE_COLUMNS, PROTOCOL_DETECTORS, PROTOCOL_DETECTOR_BY_ID, TURN_DETECTION_FOOT_OPTIONS, WEIGHT_REQUIRED_CALCULATORS, calculatorEventLegend, calculatorEventStyle, calculatorQuery, columnsForCalculator, isStepContact, normaliseSpeedPrediction, protocolDetectorSummary, stepKey } from './lib/calculators.js'
import { DISTANCE_PRED_COLS, L_FILL, L_LINE, PALETTE, PRED_COLOR, R_FILL, R_LINE, SEL_FILL, SEL_LINE, SPEED_PRED_COLS, ST_COLOR, ST_COL_COLORS, ST_ONLY_COLS, TRACE_HOVER_TEMPLATE, UI_FONT_FAMILY, buildCursorShapes, buildGapBandShapes, buildSelectedPointShapes, chartSubplotCenterTop, chartSubplotMetrics, currentAxisRange, plotAxisKey, readPlotRange } from './lib/chart.js'
import { coerceCsvColumnsToNumbers, extractContactsFromLabeledCsv, getPairStartIndex, parseCsvText } from './lib/csv.js'
import { formatDuration, formatInterval, formatMetric, formatTime, formatTimeOffset, hasSessionMetaValue, normalizeMemberName, normalizeSessionTitle, timeOffsetAsShift } from './lib/format.js'
import { SPEED_TRACKER, buildDefaultCols, computeAutoOffsetST, computeNumericColumns, detectTimeCol, groupSensorNamesByFoot, resolveStDataCol, rowsToColMap, sensorFootForName, sensorNameForFoot, sortSensorNames } from './lib/sensors.js'
import { IMU_SNAPSHOT_COLUMNS, INSOLE_TOTAL_COL, MIRRORED_LEFT_COLUMNS, SENSOR_SUM_NORM_COL, TKEO_PLOT_COLS, UNWRAPPABLE_ANGLE_COLUMNS, addAccTkeoColumn, addDerivedSessionColumns, addNormalizedSensorColumns, addSensorSumColumns, addTkeoColumns, addWeightedInsoleTotalColumn, computeGapStats } from './lib/signal.js'
import { arrayMax, arrayMin, safeNum, unwrapAngleDegrees } from './lib/utils.js'
import { correctedXData, estimateYawDrift, turnAngleDeg, withTimeInMs } from './lib/yawDrift.js'

// Video transport. Phone footage is 30 or 60 fps; one 1/30 s step is one frame
// of the former and two of the latter, which is fine for placing a contact.
const FRAME_STEP_S = 1 / 30
const FINE_SEEK_S = 0.1
const COARSE_SEEK_S = 1
const PLAYBACK_RATES = [0.25, 0.5, 1, 1.5, 2]

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
  const [sessionLabel, setSessionLabel] = useState('')
  const [sessionProtocolName, setSessionProtocolName] = useState('')
  const [sessionDeviceId, setSessionDeviceId] = useState(null)
  const [sessionTimeOffset, setSessionTimeOffset] = useState(null)
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
  const [modelCardsOpen, setModelCardsOpen] = useState(false)
  const [calculatorResults, setCalculatorResults] = useState({})
  const [activeCalculators, setActiveCalculators] = useState([])
  const [calculatorLoading, setCalculatorLoading] = useState('')
  const [selectedCalculatorContact, setSelectedCalculatorContact] = useState(null)
  // Steps the operator struck out of Target, by stepKey(). Held apart from the
  // detector results so a re-run keeps the edits instead of wiping them.
  const [deletedStepKeys, setDeletedStepKeys] = useState(() => new Set())
  const [turnDetectionFeet, setTurnDetectionFeet] = useState({
    'protocol-shuttle-detector': 'both',
    'protocol-beep-detector': 'both',
    'protocol-ttest-detector': 'both',
  })
  const [weightKg, setWeightKg] = useState('70')
  const [jumpEventProtocol, setJumpEventProtocol] = useState('vert')
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
  const [isPlaying, setIsPlaying]         = useState(false)
  const [playbackRate, setPlaybackRate]   = useState(1)
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
  const [activityMode, setActivityMode]           = useState(false)
  const [activitySpans, setActivitySpans]         = useState([])
  // The first click of a pair, waiting for its closing click. null when idle.
  const [pendingActivityFrom, setPendingActivityFrom] = useState(null)
  const [currentActivity, setCurrentActivity]     = useState(DEFAULT_ACTIVITY)
  const [selectedActivityIdx, setSelectedActivityIdx] = useState(null)
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
  const selectedCalculatorContactRef = useRef(null)
  const deletedStepKeysRef = useRef(new Set())
  const activeCalculatorsRef = useRef([])
  const calculatorDataVersionRef = useRef(0)
  const cursorShapesRef  = useRef([])
  const selectedColsRef  = useRef([])
  const columnSelectionInitializedRef = useRef(false)
  const anglesUnwrappedRef = useRef(false)
  const mirrorLeftRef      = useRef(false)
  const isDragging       = useRef(false)
  const isVideoPan      = useRef(false)
  const tlHoverRef      = useRef(null)
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
  const activityModeRef   = useRef(false)
  // The end of the plotted data, so the ribbon's last segment knows where to
  // stop. Written by renderChart, which is the only place the extent is known.
  const xMaxRef           = useRef(null)
  // markup_files as loaded, held until the shift settles: the stored ribbon is
  // in raw sample time and needs the offset added to land on the plot.
  const savedActivityFilesRef = useRef([])
  const activitySpansRef  = useRef([])
  const pendingActivityFromRef = useRef(null)
  // Snap tolerance in data units, refreshed from the visible x-range so it
  // tracks the zoom: a pixel of slack is a different number of ms at each scale.
  const activitySnapToleranceRef = useRef(0)
  const correctedXDataColRef = useRef(null)
  const currentActivityRef = useRef(DEFAULT_ACTIVITY)
  const selectedActivityIdxRef = useRef(null)
  const relabelStepRef   = useRef(null)
  const subplotRangesRef = useRef({})
  const chartsLockedRef  = useRef(false)
  const chartReorderRef  = useRef(null)
  const importedCsvTextRef = useRef('')
  const skipClearImportCsvRef = useRef(false)
  // Raw IMU channels as loaded, kept until the postprocessing is undone.
  const imuOriginalRef = useRef(null)
  // The parquet file the session was read from, exactly as fetched. Sent to the
  // calculators as-is instead of an 8 MB JSON rebuilt from the columns: the
  // server parses it in ~10 ms and caches both the frame and the results by
  // content hash. Null for a CSV dataset (nothing to send but the columns), and
  // ignored while IMU post-processing has rewritten the channels in the browser.
  const parquetBytesRef = useRef(null)

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
  useEffect(() => { activityModeRef.current = activityMode }, [activityMode])
  useEffect(() => { currentActivityRef.current = currentActivity }, [currentActivity])
  useEffect(() => { selectedActivityIdxRef.current = selectedActivityIdx }, [selectedActivityIdx])
  useEffect(() => { pendingActivityFromRef.current = pendingActivityFrom }, [pendingActivityFrom])
  useEffect(() => { correctedXDataColRef.current = correctedXDataCol }, [correctedXDataCol])
  useEffect(() => { relabelStepRef.current = relabelStep }, [relabelStep])
  useEffect(() => { calculatorResultsRef.current = calculatorResults }, [calculatorResults])
  useEffect(() => { selectedCalculatorContactRef.current = selectedCalculatorContact }, [selectedCalculatorContact])
  useEffect(() => { deletedStepKeysRef.current = deletedStepKeys }, [deletedStepKeys])
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
    setSessionTimeOffset(null)
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
    const allFiles = result.additional_info?.markup_files || []
    const files = contactMarkupFiles(allFiles)
    setMarkupFiles(files)
    setActivitySpans(activitySpansFromFiles(allFiles, offsetS1Ref.current))

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
    setActivitySpans([])
    setPendingActivityFrom(null)
    setSelectedActivityIdx(null)
    savedActivityFilesRef.current = []
    setSessionProtocolName('')
    setSessionDeviceId(null)
    setSessionTimeOffset(null)
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
    parquetBytesRef.current = null
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
    // Deletions key off the old detector output, so they mean nothing here.
    setDeletedStepKeys(new Set())
    setModelCardsOpen(false)
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
      const timeOffset = result.time_offset ?? result.timeOffset ?? null
      setSessionRecordAvailable(metadataAvailable)
      setSessionProtocolName(protocolName)
      setSessionDeviceId(hasSessionMetaValue(deviceId) ? deviceId : null)
      setSessionTimeOffset(hasSessionMetaValue(timeOffset) ? timeOffset : null)
      setSessionMemberName(normalizeMemberName(result.member_name ?? result.memberName))
      setSessionTitle(normalizeSessionTitle(result.session_title ?? result.sessionTitle))
      setLoadedSessionId(metadataAvailable ? String(sid) : '')
      if (typeof window !== 'undefined' && window.matchMedia('(max-width: 768px)').matches) {
        setMobileTab('chart')
      }

      const allMarkupFiles = result.additional_info?.markup_files || []
      const initialMarkupFiles = contactMarkupFiles(allMarkupFiles)
      setMarkupFiles(initialMarkupFiles)
      savedActivityFilesRef.current = allMarkupFiles
      // A saved markup carries the shift its contacts were drawn against, so it
      // outranks the value derived from sessions.time_offset below.
      let markupSuppliedShift = false
      if (initialMarkupFiles.length > 0) {
        const lastFile = initialMarkupFiles[initialMarkupFiles.length - 1]
        setActiveMarkupFileId(lastFile.id)
        setLeftContacts(lastFile.leftContacts || [])
        setRightContacts(lastFile.rightContacts || [])
        importedCsvTextRef.current = lastFile.csv || ''
        if (lastFile.meta) {
          markupSuppliedShift = lastFile.meta.offsetS1 !== undefined
            || lastFile.meta.offsetS2 !== undefined
          if (lastFile.meta.offsetS1 !== undefined) setOffsetS1(lastFile.meta.offsetS1)
          if (lastFile.meta.offsetS2 !== undefined) setOffsetS2(lastFile.meta.offsetS2)
          if (lastFile.meta.offsetST !== undefined) setOffsetST(lastFile.meta.offsetST)
          if (lastFile.meta.timeUnit !== undefined) setTimeUnit(lastFile.meta.timeUnit)
        }
      }

      const parquetBuffer = await parquetResp.arrayBuffer()
      const rows = await parquetReadObjects({ file: parquetBuffer })

      if (!rows?.length) { setStatus({ text: 'Сессия пустая', type: 'error' }); return }

      const colMap = rowsToColMap(rows)
      const tCol = detectTimeCol(Object.keys(colMap))
      addDerivedSessionColumns(colMap, tCol, result.additional_info)
      setParquetData(colMap)
      parquetBytesRef.current = parquetBuffer
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

      // Pre-fill the video sync from sessions.time_offset. Applied here rather
      // than beside the other session metadata because the shift has to be
      // expressed in autoUnit, which is only settled at this point.
      const autoShift = timeOffsetAsShift(timeOffset, autoUnit)
      if (autoShift !== null && !markupSuppliedShift) {
        setOffsetS1(autoShift)
        setOffsetS2(autoShift)
        offsetS1Ref.current = autoShift
      }

      // Restored after the shift above, not beside the other markup files: the
      // points are stored in raw sample time, so whatever shift they will be
      // drawn against has to be settled before they can be placed. The ref is
      // written by hand there because setState has not flushed yet.
      setActivitySpans(activitySpansFromFiles(savedActivityFilesRef.current, offsetS1Ref.current))
      setSelectedActivityIdx(null)

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

    // The activity ribbon is drawn first so the contact zones stay readable on
    // top of it. Only its own mode shows it — the two markups are edited apart.
    if (activityModeRef.current) {
      const selIdx = selectedActivityIdxRef.current
      activitySpansRef.current.forEach((span, i) => {
        const kind = ACTIVITY_BY_ID[span.activity]
        if (!kind) return
        const isSel = selIdx === i
        pushAcrossSubplots(contactShapes, {
          type: 'rect',
          x0: span.from, x1: span.to,
          y0: 0, y1: 1,
          fillcolor: kind.color + (isSel ? '55' : '2e'),
          line: { color: kind.color, width: isSel ? 3 : 1 },
          layer: 'below',
        })
      })
      // The opening click of a pair, drawn dashed until its end is placed.
      const pending = pendingActivityFromRef.current
      if (pending !== null) {
        const kind = ACTIVITY_BY_ID[currentActivityRef.current]
        pushAcrossSubplots(contactShapes, {
          type: 'line',
          x0: pending, x1: pending,
          y0: 0, y1: 1,
          line: { color: kind ? kind.color : '#334155', width: 2.5, dash: 'dot' },
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

      const selected = selectedCalculatorContactRef.current
      result.contacts.forEach((contact, index) => {
        if (contact.foot === 'left' && !showSensor1) return
        if (contact.foot === 'right' && !showSensor2) return
        // A step the operator deleted keeps its place as a grey ghost rather
        // than vanishing, so a mis-click can be seen and taken back.
        const isDeleted = isStepContact(contact)
          && deletedStepKeysRef.current.has(stepKey(calculatorId, contact))
        const isSelected = selected?.calculatorId === calculatorId && selected?.index === index
        const eventStyle = isDeleted
          ? DELETED_STEP_STYLE
          : calculatorEventStyle(style, contact)
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
          fillcolor: isSelected ? SEL_FILL : eventStyle.fill,
          line: {
            color: isSelected ? SEL_LINE : eventStyle.color,
            width: isSelected ? 3 : eventStyle.width,
            dash: isSelected ? 'solid' : eventStyle.dash,
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
    activitySpansRef.current = activitySpans
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [activitySpans, updateOverlayShapes])

  // Entering or leaving the mode, and moving the selection, both change what the
  // ribbon looks like, so the overlay is rebuilt for those too.
  useEffect(() => {
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [activityMode, selectedActivityIdx, pendingActivityFrom, currentActivity, updateOverlayShapes])

  useEffect(() => {
    showLeftRef.current  = showLeftPatterns
    showRightRef.current = showRightPatterns
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [showLeftPatterns, showRightPatterns, updateOverlayShapes])

  useEffect(() => {
    if (plotInitRef.current && chartDivRef.current) updateOverlayShapes()
  }, [showGaps, checkHzData, showSensor1, showSensor2, showSpeedTracker, offsetS1, offsetS2, offsetST, timeUnit, selectedCols, selectedMarkup, calculatorResults, activeCalculators, selectedCalculatorContact, deletedStepKeys, updateOverlayShapes])

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

  // ── Target step actions ───────────────────────────────────────────────────
  /**
   * Strike a detected step out of Target, or put a struck-out one back. Only
   * steps: a turn, a sprint or a flight is an event the model reports, not
   * something the Target column ever marks.
   */
  const toggleDeletedStep = useCallback((entry) => {
    if (!entry || !isStepContact(entry.contact)) return
    const key = stepKey(entry.calculatorId, entry.contact)
    setDeletedStepKeys(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }, [])

  const restoreDeletedSteps = useCallback(() => setDeletedStepKeys(new Set()), [])

  // ── Activity ribbon actions ───────────────────────────────────────────────
  /** Take back the half-placed span first, then the last completed one. */
  const undoActivitySpan = useCallback(() => {
    if (pendingActivityFromRef.current !== null) {
      setPendingActivityFrom(null)
      return
    }
    setActivitySpans(prev => prev.slice(0, -1))
    setSelectedActivityIdx(null)
  }, [])

  const clearActivitySpans = useCallback(() => {
    setActivitySpans([])
    setPendingActivityFrom(null)
    setSelectedActivityIdx(null)
  }, [])

  const deleteSelectedActivity = useCallback(() => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      if (idx == null || idx >= prev.length) return prev
      return prev.filter((_, i) => i !== idx)
    })
    setSelectedActivityIdx(null)
  }, [])

  /**
   * The measured turn for each span, in the same order. Only 'turn' spans get
   * one — a heading change during a run is not a turn anyone marked — and a
   * hand-entered angleDeg on the span wins over the measurement.
   */
  const spanAngles = useMemo(() => activitySpans.map(span => {
    if (span.activity !== 'turn') return null
    if (span.angleDeg !== undefined) return span.angleDeg
    if (!parquetData) return null
    const offset = offsetS1
    return turnAngleDeg(parquetData, span.from - offset, span.to - offset, correctedXDataCol)
  }), [activitySpans, parquetData, offsetS1, correctedXDataCol])

  /** Pin an angle onto the selected span, overriding what was measured. */
  const setActivityAngle = useCallback((deg) => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      if (idx == null || !prev[idx]) return prev
      const next = [...prev]
      next[idx] = { ...next[idx], angleDeg: deg }
      return next
    })
  }, [])

  /** Drop a hand-entered angle so the measurement takes over again. */
  const clearActivityAngle = useCallback(() => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      if (idx == null || !prev[idx]) return prev
      const next = [...prev]
      const cleared = { ...next[idx] }
      delete cleared.angleDeg
      next[idx] = cleared
      return next
    })
  }, [])

  /** Nudge one edge of the selected span by a step in the current time unit. */
  const nudgeActivityEdge = useCallback((edge, deltaMs) => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      if (idx == null || !prev[idx]) return prev
      const step = timeUnitRef.current === 'ms' ? deltaMs : deltaMs / 1000
      return resizeActivitySpan(prev, idx, edge, prev[idx][edge] + step)
    })
  }, [])

  /** Set one edge of the selected span to an absolute time. */
  const setActivityEdge = useCallback((edge, value) => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      if (idx == null || !prev[idx]) return prev
      return resizeActivitySpan(prev, idx, edge, value)
    })
  }, [])

  /**
   * Close the gap between the selected span and the neighbour on that side, so
   * two activities meet exactly instead of leaving a sliver of unlabelled data.
   */
  const snapActivityToNeighbour = useCallback((edge) => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      const span = idx == null ? null : prev[idx]
      if (!span) return prev
      const neighbour = edge === 'from'
        ? prev.reduce((acc, o, i) => (
          i !== idx && o.to <= span.from && (!acc || o.to > acc.to) ? o : acc), null)
        : prev.reduce((acc, o, i) => (
          i !== idx && o.from >= span.to && (!acc || o.from < acc.from) ? o : acc), null)
      if (!neighbour) return prev
      return resizeActivitySpan(
        prev, idx, edge, edge === 'from' ? neighbour.to : neighbour.from)
    })
  }, [])

  /** Relabel the selected span in place, leaving its boundaries alone. */
  const relabelSelectedActivity = useCallback((activity) => {
    setActivitySpans(prev => {
      const idx = selectedActivityIdxRef.current
      if (idx == null || idx >= prev.length) return prev
      const next = [...prev]
      next[idx] = { ...next[idx], activity }
      return next
    })
  }, [])

  // Number keys pick the activity to paint with, and relabel the selected
  // segment if there is one. Ignored while typing so the session-id box and the
  // offset inputs keep working.
  useEffect(() => {
    if (!activityMode) return undefined
    const onKey = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const tag = event.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || event.target?.isContentEditable) return
      if (event.key === 'Escape') {
        event.preventDefault()
        setPendingActivityFrom(null)
        setSelectedActivityIdx(null)
        return
      }
      // Arrows nudge the selected span's edges: plain moves the end, Shift the
      // start, so both can be trimmed without leaving the keyboard.
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        if (selectedActivityIdxRef.current === null) return
        event.preventDefault()
        const delta = event.key === 'ArrowLeft' ? -100 : 100
        nudgeActivityEdge(event.shiftKey ? 'from' : 'to', delta)
        return
      }
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (selectedActivityIdxRef.current === null) return
        event.preventDefault()
        deleteSelectedActivity()
        return
      }
      const kind = ACTIVITY_KINDS.find(k => k.key === event.key)
      if (!kind) return
      event.preventDefault()
      setCurrentActivity(kind.id)
      if (selectedActivityIdxRef.current !== null) relabelSelectedActivity(kind.id)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activityMode, relabelSelectedActivity, deleteSelectedActivity, nudgeActivityEdge])


  // Delete strikes the selected step out of Target; pressing it again on the
  // ghost that is left puts the step back. Activity mode has its own Delete,
  // and no calculator contact can be selected while it is on.
  useEffect(() => {
    if (activityMode) return undefined
    if (!selectedCalculatorContact || !isStepContact(selectedCalculatorContact.contact)) return undefined
    const onKey = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const tag = event.target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || event.target?.isContentEditable) return
      if (event.key !== 'Delete' && event.key !== 'Backspace') return
      event.preventDefault()
      toggleDeletedStep(selectedCalculatorContact)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activityMode, selectedCalculatorContact, toggleDeletedStep])

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

    // Detected steps are Target too: the model's contacts, minus the ones the
    // operator struck out on the chart, unioned with the hand-placed zones
    // above - which are how a step the model missed gets added. Model times
    // come back in raw sample time, so unlike the hand-placed zones they need
    // no offset removed.
    const timeScale = timeUnitRef.current === 'ms' ? 1000 : 1
    const detectedL = []
    const detectedR = []
    const deleted = deletedStepKeysRef.current
    // Only the detectors currently switched on: Target has to be what the
    // operator can see and edit on the chart. A result left in the cache by a
    // detector they switched off draws nothing, so it must not write 1s either.
    const calculatorIds = activeCalculatorsRef.current
    calculatorIds.forEach(calculatorId => {
      const contacts = calculatorResultsRef.current[calculatorId]?.contacts
      if (!contacts?.length) return
      contacts.forEach(contact => {
        if (!isStepContact(contact)) return
        if (deleted.has(stepKey(calculatorId, contact))) return
        const start = Number(contact.start_time_s) * timeScale
        const end = Number(contact.end_time_s) * timeScale
        if (!Number.isFinite(start) || !Number.isFinite(end)) return
        const iv = [Math.min(start, end), Math.max(start, end)]
        if (contact.foot === 'left') detectedL.push(iv)
        else if (contact.foot === 'right') detectedR.push(iv)
      })
    })

    const hdr = [...allCols, 'Target'].join(',')
    const rows = []
    for (let i = 0; i < n; i++) {
      const name = nameArr[i] || ''
      const t    = timeArr[i]
      const isRight = rightSensorNames.has(name)
      const isLeft  = leftSensorNames.has(name)
      const target = name === SPEED_TRACKER
        ? ''
        : isRight
          ? (inIv(t, rIv) || inIv(t, detectedR) ? 1 : 0)
          : isLeft
            ? (inIv(t, lIv) || inIv(t, detectedL) ? 1 : 0)
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

  /**
   * Every sensor row plus the activity it falls in — the training set for an
   * activity classifier. The ribbon is drawn against the shifted plot, so the
   * same offset the traces carry is removed to get back to raw sample times;
   * S1's offset stands in for the athlete, whose activity is not per-foot.
   */
  const generateActivityCsvString = useCallback(() => {
    if (!parquetData) return ''
    const spans = activitySpansRef.current
    if (!spans.length) return ''

    const allCols = Object.keys(parquetData)
    const timeArr = parquetData[timeCol] || []
    const offset = offsetS1Ref.current
    const shifted = spans.map(span => {
      const from = span.from - offset
      const to = span.to - offset
      const angle = span.activity !== 'turn'
        ? null
        : span.angleDeg !== undefined
          ? span.angleDeg
          : turnAngleDeg(parquetData, from, to, correctedXDataColRef.current)
      return { from, to, activity: span.activity, angle }
    })
    // Spans are inclusive at both ends — they are placed by eye, so a sample
    // landing exactly on a boundary belongs to that span. Unmarked stretches
    // export a blank Activity rather than being guessed at.

    // TurnAngleDeg repeats on every row of a turn span and is blank elsewhere,
    // so a windowed model can read it off any sample it trains on.
    const hdr = [...allCols, 'Activity', 'TurnAngleDeg'].join(',')
    const rows = []
    for (let i = 0; i < timeArr.length; i++) {
      const vals = allCols.map(c => {
        const v = parquetData[c][i]
        return v == null ? '' : String(v)
      })
      const idx = activitySpanAt(shifted, safeNum(timeArr[i]) ?? NaN)
      vals.push(idx >= 0 ? shifted[idx].activity : '')
      vals.push(idx >= 0 && shifted[idx].angle !== null && shifted[idx].angle !== undefined
        ? shifted[idx].angle.toFixed(1)
        : '')
      rows.push(vals.join(','))
    }
    return hdr + '\n' + rows.join('\n')
  }, [parquetData, timeCol])

  const exportActivities = useCallback(() => {
    const csv = generateActivityCsvString()
    if (!csv) {
      setStatus({ text: 'Нет размеченных активностей', type: 'error' })
      return
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url
    a.download = (sessionLabel || 'session').replace(/\s+/g, '_') + '_activity.csv'
    document.body.appendChild(a); a.click()
    document.body.removeChild(a); URL.revokeObjectURL(url)
  }, [generateActivityCsvString, sessionLabel])

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

  /**
   * The ribbon is stored as its own entry in markup_files, beside the contact
   * markups, so the two can be edited and saved without disturbing each other.
   * Points are written in raw sample time — the plot offset is removed here and
   * added back on load, so a later change to the shift does not move them.
   */
  const saveActivitiesToDb = useCallback(async () => {
    const sid = sessionId.trim()
    if (!sid) {
      setStatus({ text: 'Укажите ID сессии в поле слева', type: 'error' })
      return
    }
    if (!sessionRecordAvailable) {
      setStatus({ text: 'Записи сессии в БД нет — сохранить активности нельзя', type: 'error' })
      return
    }
    if (activitySpansRef.current.length === 0) {
      setStatus({ text: 'Нет размеченных активностей', type: 'error' })
      return
    }

    setIsSaveLoading(true)
    try {
      const currentSession = await fetchSessionMarkupsFromDb(sid)
      const currentAdditionalInfo = currentSession.additional_info || {}
      const existingFiles = Array.isArray(currentAdditionalInfo.markup_files)
        ? [...currentAdditionalInfo.markup_files]
        : []

      const offset = offsetS1Ref.current
      const existingIndex = existingFiles.findIndex(f => f.type === ACTIVITY_FILE_TYPE)
      const fileId = existingIndex >= 0 ? existingFiles[existingIndex].id : `af_${Date.now()}`
      const newFile = {
        id: fileId,
        filename: existingIndex >= 0
          ? existingFiles[existingIndex].filename
          : `activities_${sid}.json`,
        type: ACTIVITY_FILE_TYPE,
        updated_at: new Date().toISOString(),
        activitySpans: activitySpansRef.current.map(sp => ({
          from: sp.from - offset,
          to: sp.to - offset,
          activity: sp.activity,
          // Only a hand-entered angle is stored; a measured one is recomputed on
          // load, so it follows any later change to the drift correction.
          ...(sp.angleDeg === undefined ? {} : { angleDeg: sp.angleDeg }),
        })),
        meta: { offsetS1: offset, timeUnit: timeUnitRef.current },
      }

      const updatedFiles = [...existingFiles]
      if (existingIndex >= 0) updatedFiles[existingIndex] = newFile
      else updatedFiles.push(newFile)

      const resp = await fetch(`${MARKUP_API}/sessions/${sid}/additional-info`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          additional_info: { ...currentAdditionalInfo, markup_files: updatedFiles },
        }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }
      const updatedSession = await resp.json()
      setMarkupFiles(updatedSession.additional_info?.markup_files || updatedFiles)
      setStatus({
        text: `✓ Активности сохранены · ${activitySpansRef.current.length} отрезк(ов)`,
        type: 'ok',
      })
    } catch (err) {
      setStatus({ text: `Ошибка сохранения активностей: ${err.message}`, type: 'error' })
    } finally {
      setIsSaveLoading(false)
    }
  }, [sessionId, sessionRecordAvailable, fetchSessionMarkupsFromDb, token])

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
    const hasManualMarkup = leftContactsRef.current.length > 0 || rightContactsRef.current.length > 0
    const hasDetectedSteps = activeCalculatorsRef.current.some(id =>
      calculatorResultsRef.current[id]?.contacts?.some(c =>
        isStepContact(c) && !deletedStepKeysRef.current.has(stepKey(id, c))
      )
    )
    if (!hasManualMarkup && !hasDetectedSteps) {
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
        // Deletions key off the old detector output, so they mean nothing here.
        setDeletedStepKeys(new Set())
        setModelCardsOpen(false)
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
        parquetBytesRef.current = null   // a CSV: the columns are all there is
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
            setSessionTimeOffset(
              hasSessionMetaValue(sess.time_offset ?? sess.timeOffset)
                ? (sess.time_offset ?? sess.timeOffset)
                : null
            )
            // No saved markup is read on this path, so nothing outranks the DB.
            const csvShift = timeOffsetAsShift(sess.time_offset ?? sess.timeOffset, autoUnit)
            if (csvShift !== null) { setOffsetS1(csvShift); setOffsetS2(csvShift) }
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
    // Deletions key off the old detector output, so they mean nothing here.
    setDeletedStepKeys(new Set())
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

  // The zoom buttons sit over the video. A mousedown on them must neither start
  // a pan (it would bubble to the wrap) nor move focus onto the button - with
  // focus there, the next Space re-fires the zoom instead of play / pause.
  const swallowOverlayMouseDown = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

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
    setIsPlaying(false)
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
    // Deletions key off the old detector output, so they mean nothing here.
    setDeletedStepKeys(new Set())
    setModelCardsOpen(false)
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

    try {
      const arrayBuffer = await file.arrayBuffer()
      const rows = await parquetReadObjects({ file: arrayBuffer })

      if (!rows?.length) { setStatus({ text: 'Файл пустой', type: 'error' }); return }

      const colMap = rowsToColMap(rows)
      const tCol = detectTimeCol(Object.keys(colMap))
      addDerivedSessionColumns(colMap, tCol, null)
      setParquetData(colMap)
      parquetBytesRef.current = arrayBuffer
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
          setSessionTimeOffset(hasSessionMetaValue(sess.time_offset ?? sess.timeOffset) ? (sess.time_offset ?? sess.timeOffset) : null)
          // No saved markup is read on this path, so nothing outranks the DB.
          const parquetShift = timeOffsetAsShift(sess.time_offset ?? sess.timeOffset, autoUnit)
          if (parquetShift !== null) { setOffsetS1(parquetShift); setOffsetS2(parquetShift) }
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

    const sid = sessionId.trim()
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
  }, [speedPredict, sessionId, token])

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

  const toggleAdditionalCalculator = useCallback(async (calculatorId, options = {}) => {
    const force = Boolean(options.force)
    const requestedDetectionFoot = PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)
      ? (options.detectionFoot || turnDetectionFeet[calculatorId] || 'both')
      : 'both'

    if (activeCalculators.includes(calculatorId) && !force) {
      setActiveCalculators(prev => prev.filter(id => id !== calculatorId))
      // The GRF panel exists only for the prediction, so it goes away with it -
      // unless the session really has a column by that name.
      if (calculatorId === 'grf-split' && !columns.includes(GRF_PRED_COL)) {
        setSelectedCols(prev => prev.filter(col => col !== GRF_PRED_COL))
      }
      return
    }

    const cachedResult = calculatorResults[calculatorId]
    // Both jump-conditioned models take the movement as an input channel, so a
    // result computed under another protocol is not the same result.
    const cacheMatchesJumpProtocol = !['jump-events', 'grf-split'].includes(calculatorId)
      || (cachedResult?.summary?.protocol || 'vert') === jumpEventProtocol
    // The server echoes the weight rounded to 0.1 kg, so compare with that slack.
    const cacheMatchesWeight = calculatorId !== 'grf-split'
      || Math.abs(Number(cachedResult?.summary?.weight_kg) - Number(weightKg)) < 0.1
    const cacheMatchesDetectionFoot = !PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)
      || (cachedResult?.summary?.detection_foot || 'both') === requestedDetectionFoot
    const cacheHasSeparateFootOverlay = !PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)
      || requestedDetectionFoot !== 'both'
      || (cachedResult?.summary?.left_turn_count != null && cachedResult?.summary?.right_turn_count != null)
    if (!force && cachedResult && cacheMatchesJumpProtocol && cacheMatchesWeight
      && cacheMatchesDetectionFoot && cacheHasSeparateFootOverlay) {
      setActiveCalculators(prev => prev.includes(calculatorId) ? prev : [...prev, calculatorId])
      return
    }

    if (!parquetData || calculatorLoading) return

    const parsedWeight = Number(weightKg)
    // The force model takes body weight as a model input, not just as a unit
    // conversion, so a missing weight is refused rather than defaulted here.
    if (WEIGHT_REQUIRED_CALCULATORS.has(calculatorId)
      && (!Number.isFinite(parsedWeight) || parsedWeight <= 0)) {
      setStatus({ text: 'Укажите положительный вес для Total GRF', type: 'error', area: 'models' })
      return
    }

    const dataVersion = calculatorDataVersionRef.current
    setCalculatorLoading(calculatorId)
    try {
      const options = {}
      if (WEIGHT_REQUIRED_CALCULATORS.has(calculatorId)) options.weight_kg = parsedWeight
      if (calculatorId === 'jump-events') options.protocol = jumpEventProtocol
      if (calculatorId === 'grf-split') {
        // The GRF model reads its take-off / landing instants from the plate-trained
        // jump detector, which is conditioned on the movement, so pass the same
        // protocol the Jump events calculator uses - and, when that calculator has
        // already run on this data under this movement, its flights themselves, so
        // the detector is not run a second time (it costs as much as the force model).
        options.protocol = jumpEventProtocol
        const jumpEvents = calculatorResults['jump-events']
        if (jumpEvents?.contacts?.length
          && (jumpEvents.summary?.protocol || 'vert') === jumpEventProtocol) {
          options.jump_pairs = jumpEvents.contacts.map(c => [c.start_time_s, c.end_time_s])
        }
      }
      if (PER_FOOT_TURN_DETECTOR_IDS.has(calculatorId)) {
        const selectedSensorName = requestedDetectionFoot === 'both'
          ? ''
          : sensorNameForFoot(insoleSensorNames, requestedDetectionFoot)
        if (requestedDetectionFoot !== 'both' && !selectedSensorName) {
          throw new Error(`В данных нет ${requestedDetectionFoot === 'left' ? 'левой' : 'правой'} ноги`)
        }
        options.detection_foot = requestedDetectionFoot
        if (selectedSensorName) options.sensor_name = selectedSensorName
      }
      // The parquet bytes describe the session only while nothing rewrote it in
      // the browser; after IMU post-processing the columns are the truth.
      const parquetBytes = imuApplied ? null : parquetBytesRef.current
      const url = `${CALCULATOR_API}/calculate/${calculatorId}`
      const resp = parquetBytes
        ? await fetch(`${url}?${calculatorQuery(options)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/vnd.apache.parquet', 'accept': 'application/json' },
          body: parquetBytes,
        })
        : await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'accept': 'application/json' },
          body: JSON.stringify({ columns: columnsForCalculator(calculatorId, parquetData), ...options }),
        })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(parseApiError(errData, resp.status))
      }

      const data = await resp.json()
      if (dataVersion !== calculatorDataVersionRef.current) return

      setCalculatorResults(prev => ({ ...prev, [calculatorId]: data }))
      setActiveCalculators(prev => prev.includes(calculatorId) ? prev : [...prev, calculatorId])
      if (calculatorId === 'grf-split' && data.data_points?.length) {
        setSelectedCols(prev => prev.includes(GRF_PRED_COL) ? prev : [...prev, GRF_PRED_COL])
      }
      setSelectedCalculatorContact(prev => prev?.calculatorId === calculatorId ? null : prev)

      const left = data.summary?.left?.contact_count || 0
      const right = data.summary?.right?.contact_count || 0
      const cadence = data.summary?.cadence_spm
      const resultText = PROTOCOL_DETECTOR_BY_ID[calculatorId]
        ? protocolDetectorSummary(data)
        : calculatorId === 'jump-events'
          ? `${data.summary?.total_jump_count || 0} прыж. · высота ${formatMetric(data.summary?.mean_jump_height_cm, 1, ' см')}`
          : calculatorId === 'grf-split'
            ? `${data.summary?.jump_count || 0} прыж. · пик ${formatMetric(data.summary?.peak_force?.percent_bw, 0, ' %BW')}`
              + ` · импульс ${formatMetric(data.summary?.mean_contact_impulse_bw_s, 2, ' BW·с')}`
              + ` · ${data.data_points?.length || 0} точек кривой`
          : calculatorId === PLATE_FLIGHT_ID
            ? `${data.summary?.total_jump_count || 0} полётов по плитам`
              + ` (только падение ${data.summary?.jumps_free_fall_only || 0})`
              + ` · маска ${(data.summary?.masked_no_gate || 0) + (data.summary?.masked_insole_loaded || 0)}`
              + ` · через край плиты ${(data.summary?.hops_onto_plate || 0) + (data.summary?.hops_off_plate || 0)}`
              + ` · досинхр. плит ${formatMetric(data.summary?.resync_ms, 0, ' мс')}`
              + ` · нуль L ${formatMetric(data.summary?.plate_zero_n?.left, 0, '')} / R ${formatMetric(data.summary?.plate_zero_n?.right, 0, ' Н')}`
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
  }, [activeCalculators, calculatorResults, parquetData, calculatorLoading, weightKg, jumpEventProtocol, turnDetectionFeet, insoleSensorNames, columns, imuApplied])

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
    xMaxRef.current = xMax
    activitySnapToleranceRef.current = (xMax - xMin) * ACTIVITY_SNAP_FRACTION

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
      // The GRF panel has no raw data at all, so its range comes from the
      // prediction alone.
      if (col === GRF_PRED_COL) {
        const grf = calculatorResults['grf-split']
        if (grf?.data_points?.length) {
          vals = [...vals, ...grf.data_points.map(point => safeNum(point.total))
            .filter(value => value !== null)]
        }
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

      // Total GRF overlay on its own virtual panel, in %BW where 100 is quiet
      // standing. The model's time axis is the shared grid built from both feet,
      // so it carries no S1/S2 shift - it is drawn unshifted on purpose, and
      // 100 %BW gets a reference line to read the curve against. The curve is in
      // the model's own definition (plate low-passed at target_lowpass_hz), so the
      // legend says so; lay Plate_Fz_total_lp20_pctBW over it for a fair check.
      if (col === GRF_PRED_COL) {
        const grf = calculatorResults['grf-split']
        const points = grf?.data_points || []
        if (points.length) {
          const grfTimeScale = timeUnitRef.current === 'ms' ? 1 : 0.001
          const xs = points.map(point => point.time * grfTimeScale)
          const lpHz = grf?.summary?.target_lowpass_hz
          traces.push({
            x: xs,
            y: points.map(point => point.total),
            name: lpHz ? `GRF total · ${lpHz} Гц` : 'GRF total',
            type: 'scatter', mode: 'lines',
            xaxis: xAxis, yaxis: yAxis,
            line: { color: '#111827', width: 2 },
            connectgaps: false,
            hovertemplate: TRACE_HOVER_TEMPLATE,
          })
          traces.push({
            x: [xs[0], xs[xs.length - 1]],
            y: [100, 100],
            name: 'стойка 100 %BW',
            type: 'scatter', mode: 'lines',
            xaxis: xAxis, yaxis: yAxis,
            line: { color: '#9ca3af', width: 1, dash: 'dot' },
            hoverinfo: 'skip',
            showlegend: false,
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
        const candidates = []

        activeCalculatorsRef.current.forEach(calculatorId => {
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
            // GCT windows can be a few dozen ms wide - only a handful of
            // pixels at normal zoom - so pad the hit box with the same
            // on-screen tolerance used for activity-span snapping, or a
            // precise contact becomes unclickable.
            const padding = currentSnapTolerance() / 2
            const x0 = Math.min(start, end) - padding
            const x1 = Math.max(start, end) + padding
            if (x >= x0 && x <= x1) {
              candidates.push({ calculatorId, index, contact })
            }
          })
        })

        // The narrowest event under the cursor wins, so a GCT window inside a
        // longer phase stays clickable.
        candidates.sort((a, b) => Number(a.contact.duration_ms || 0) - Number(b.contact.duration_ms || 0))
        return candidates[0] || null
      }

      const selectCalculatorContactAtX = (x) => {
        const selectedContact = findCalculatorContact(Number(x))
        if (!selectedContact) return false
        setSelectedCalculatorContact(selectedContact)
        return true
      }

      // Zooming in should tighten the snap, so the tolerance is taken from the
      // range on screen at click time rather than the whole recording.
      const currentSnapTolerance = () => {
        const range = chartDivRef.current?._fullLayout?.xaxis?.range
        if (Array.isArray(range) && range.length >= 2) {
          const width = Math.abs(Number(range[1]) - Number(range[0]))
          if (Number.isFinite(width) && width > 0) return width * ACTIVITY_SNAP_FRACTION
        }
        return activitySnapToleranceRef.current
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
        } else if (activityModeRef.current) {
          const pending = pendingActivityFromRef.current
          const snapped = snapActivityEdge(
            activitySpansRef.current, t, currentSnapTolerance())
          if (pending === null) {
            // A click inside an existing span selects it instead of starting a
            // new one, so a mis-click can be corrected rather than stacking.
            const hit = activitySpanAt(activitySpansRef.current, t)
            if (hit >= 0) {
              setSelectedActivityIdx(prev => (prev === hit ? null : hit))
            } else {
              setSelectedActivityIdx(null)
              setPendingActivityFrom(snapped)
            }
          } else {
            const span = closeActivitySpan(pending, snapped, currentActivityRef.current)
            if (span.to > span.from) {
              setActivitySpans(prev => insertActivitySpan(prev, span))
            }
            setPendingActivityFrom(null)
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
        if (relabelStepRef.current || labelingRef.current || activityModeRef.current) return

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
    // calculatorResults is a dependency because the GRF panel draws its curve from
    // the calculator's result, not from chartData - without it a fresh run would
    // only appear on the next unrelated redraw.
  }, [chartData, selectedCols, timeCol, sensorGroups, hasSpeedTracker, updateOverlayShapes, showSensor1, showSensor2, speedPredict, showSpeedPredict, showDistancePredict, calculatorResults])

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

  // ── Force-plate ground truth («Плиты») ────────────────────────────────────
  // Only a session that carries the plate force can be labelled; a production
  // session from GCS never does, so the toggle is simply off for it.
  const hasPlateColumns = useMemo(
    () => columns.some(col => PLATE_FORCE_COLUMNS.includes(col)),
    [columns],
  )
  const plateFlightActive = activeCalculators.includes(PLATE_FLIGHT_ID)
  const plateFlightLoading = calculatorLoading === PLATE_FLIGHT_ID
  const plateFlightResult = calculatorResults[PLATE_FLIGHT_ID]

  /** Hover copy for the toggle: why it is off, or what the labeler found. */
  const plateFlightTitle = useMemo(() => {
    if (!parquetData) return 'Загрузите сессию с силовыми платформами'
    if (!hasPlateColumns) {
      return `В данных нет силы плит (${PLATE_FORCE_COLUMNS.join(' / ')}) — ground truth по плитам недоступен`
    }
    const head = plateFlightActive
      ? 'Скрыть полёты по плитам'
      : 'Показать ground truth прыжков по силовым платформам — разметка v7: обе плиты < 20 Н, '
        + 'гейт «удар в 60 мс ИЛИ свободное падение ≥ 7 м/с²»; обе ноги в воздухе от края до края при «нагруженной» стельке — '
        + 'прыжок (стелька врёт, плиты главнее); прыжки с пола на плиту / с плиты на пол — маска '
        + '(виден только один плитный край); та же разметка, на которой обучается Jump events'
    const s = plateFlightResult?.summary
    if (!s) return head
    return [
      head,
      `${s.total_jump_count} прыжков (удар+падение ${s.jumps_impact_and_free_fall} · только удар ${s.jumps_impact_only} · только падение ${s.jumps_free_fall_only}`
        + ` · стелька врёт ${s.jumps_insole_lies ?? 0})`,
      `маска: ${s.masked_no_gate} без гейта · ${s.masked_insole_loaded} стелька на полу`
        + ` · через край плиты: ${s.hops_onto_plate ?? 0} на плиту / ${s.hops_off_plate ?? 0} с плиты`,
      `досинхронизация плит ${s.resync_ms} мс (r ${s.resync_r})`,
      `нуль плит L ${s.plate_zero_n?.left} / R ${s.plate_zero_n?.right} Н`,
    ].join(' · ')
  }, [parquetData, hasPlateColumns, plateFlightActive, plateFlightResult])

  /**
   * Turn the accepted flights into markup: one S1 and one S2 interval per jump,
   * Target=1 on both feet for the airborne rows - the bilateral convention the
   * detector is trained and scored against. Replaces the current markup, after
   * asking, because merging would stack duplicates on a second press.
   */
  const handlePlateFlightToMarkup = useCallback(() => {
    const jumps = (plateFlightResult?.contacts || []).filter(c => c.kind === 'plate_flight')
    if (!jumps.length) {
      setStatus({ text: 'Полётов по плитам нет — разметка не изменена', type: 'error' })
      return
    }
    const hasMarkup = leftContactsRef.current.length > 0 || rightContactsRef.current.length > 0
    if (hasMarkup && !window.confirm(
      `Заменить текущую разметку S1 и S2 на ${jumps.length} полётов по плитам?`)) return
    const scale = timeUnit === 'ms' ? 1000 : 1
    const pairs = (shift) => jumps.flatMap(c => [c.start_time_s * scale + shift, c.end_time_s * scale + shift])
    setLeftContacts(pairs(offsetS1))
    setRightContacts(pairs(offsetS2))
    setSelectedMarkup(null)
    setStatus({
      text: `✓ ${jumps.length} полётов по плитам записаны в разметку S1 и S2 (Target=1 — полёт)`,
      type: 'ok',
    })
  }, [plateFlightResult, timeUnit, offsetS1, offsetS2])

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

  // ── Video transport ───────────────────────────────────────────────────────
  const seekTo = useCallback((timeS) => {
    const video = videoRef.current
    if (!video || !Number.isFinite(video.duration)) return
    video.currentTime = Math.max(0, Math.min(video.duration, timeS))
  }, [])

  // Frame steps pause first: stepping through a playing video is meaningless,
  // and the operator who steps wants to stay on that frame.
  const seekBy = useCallback((deltaS, { pause = false } = {}) => {
    const video = videoRef.current
    if (!video || !Number.isFinite(video.duration)) return
    if (pause && !video.paused) video.pause()
    video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + deltaS))
  }, [])

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) void video.play().catch(() => {})
    else video.pause()
  }, [])

  const changePlaybackRate = useCallback((rate) => {
    const next = PLAYBACK_RATES.includes(rate) ? rate : 1
    setPlaybackRate(next)
    if (videoRef.current) videoRef.current.playbackRate = next
  }, [])

  const stepPlaybackRate = useCallback((direction) => {
    const index = PLAYBACK_RATES.indexOf(playbackRate)
    const nextIndex = Math.max(0, Math.min(PLAYBACK_RATES.length - 1, (index < 0 ? 2 : index) + direction))
    changePlaybackRate(PLAYBACK_RATES[nextIndex])
  }, [playbackRate, changePlaybackRate])

  // Keyboard transport, active whenever a video is loaded: Space play / pause,
  // arrows 0.1 s (Shift: 1 s), comma / period one frame, Home / End, < > speed.
  // Registered in the capture phase so it runs before the <video>'s own key
  // handling (5 s arrow jumps, its own Space) and replaces it. Typing, focused
  // buttons / links and the keyboard-resizable separators keep their keys; so
  // do the arrows while an activity span is selected for nudging.
  useEffect(() => {
    if (!videoUrl) return undefined
    const onKey = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const target = event.target
      const tag = target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target?.isContentEditable) return
      if (tag === 'BUTTON' || tag === 'A') return
      if (target && target !== document.body && tag !== 'VIDEO' && target.tabIndex >= 0) return
      if (activityModeRef.current && selectedActivityIdxRef.current !== null
        && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) return

      switch (event.key) {
        case ' ': togglePlay(); break
        case 'ArrowLeft': seekBy(event.shiftKey ? -COARSE_SEEK_S : -FINE_SEEK_S); break
        case 'ArrowRight': seekBy(event.shiftKey ? COARSE_SEEK_S : FINE_SEEK_S); break
        case ',': seekBy(-FRAME_STEP_S, { pause: true }); break
        case '.': seekBy(FRAME_STEP_S, { pause: true }); break
        case '<': stepPlaybackRate(-1); break
        case '>': stepPlaybackRate(1); break
        case 'Home': seekTo(0); break
        case 'End': seekTo(videoRef.current?.duration ?? 0); break
        default: return
      }
      event.preventDefault()
      event.stopPropagation()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [videoUrl, togglePlay, seekBy, seekTo, stepPlaybackRate])

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

  // The time under the pointer, written straight into the DOM: a state update
  // per mousemove would re-render the whole app.
  const handleTimelineHover = useCallback((e) => {
    const rect = timelineRef.current?.getBoundingClientRect()
    const hover = tlHoverRef.current
    if (!rect || !hover || videoDuration <= 0) return
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width))
    hover.style.left = `${(x / rect.width) * 100}%`
    hover.textContent = formatTime((x / rect.width) * videoDuration)
    hover.hidden = false
  }, [videoDuration])

  const handleTimelineLeave = useCallback(() => {
    if (tlHoverRef.current) tlHoverRef.current.hidden = true
  }, [])

  // Wheel over the timeline scrubs: 0.1 s per notch, 1 s with Shift. Native
  // listener because React's onWheel is passive and cannot stop the page scroll.
  useEffect(() => {
    const el = timelineRef.current
    if (!el) return undefined
    const onWheel = (e) => {
      e.preventDefault()
      const direction = e.deltaY > 0 || e.deltaX > 0 ? 1 : -1
      seekBy(direction * (e.shiftKey ? COARSE_SEEK_S : FINE_SEEK_S))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [seekBy, videoDuration])

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

  // What the exported Target column is made of, so the operator can see the
  // effect of a deletion without opening the CSV.
  const targetStepStats = useMemo(() => {
    let detected = 0
    let removed = 0
    activeCalculators.forEach(id => {
      (calculatorResults[id]?.contacts || []).forEach(contact => {
        if (!isStepContact(contact)) return
        if (deletedStepKeys.has(stepKey(id, contact))) removed += 1
        else detected += 1
      })
    })
    const manual = Math.floor(leftContacts.length / 2) + Math.floor(rightContacts.length / 2)
    return { detected, removed, manual, total: detected + manual }
  }, [calculatorResults, activeCalculators, deletedStepKeys, leftContacts, rightContacts])

  /** The selected contact, when it is a step Target can hold. */
  const selectedStepEntry = selectedCalculatorContact
    && isStepContact(selectedCalculatorContact.contact)
    ? selectedCalculatorContact
    : null
  const selectedStepDeleted = selectedStepEntry
    ? deletedStepKeys.has(stepKey(selectedStepEntry.calculatorId, selectedStepEntry.contact))
    : false

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
  const activeModelCardCount = activeCalculators.filter(id => MODEL_SECTION_CALCULATOR_IDS.has(id)).length
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
          {hasSessionMetaValue(sessionTimeOffset) && (
            <FileBadge type="offset" title="sessions.time_offset">
              offset {formatTimeOffset(sessionTimeOffset)}
            </FileBadge>
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
        <span className="mobile-session-chip">
          <span className="mobile-session-key">Offset</span>
          <span className="mobile-session-val">
            {hasSessionMetaValue(sessionTimeOffset) ? formatTimeOffset(sessionTimeOffset) : '—'}
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
                    timeOffset={sessionTimeOffset}
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
                          className={`calculator-expand${modelCardsOpen ? ' open' : ''}`}
                          onClick={() => setModelCardsOpen(open => !open)}
                          aria-expanded={modelCardsOpen}
                          aria-controls="model-cards"
                        >
                          <span>
                            Детекторы и модели
                            {activeModelCardCount > 0 && (
                              <span className="calculator-active-count">{activeModelCardCount}</span>
                            )}
                          </span>
                          <span className="calculator-expand-chevron">⌄</span>
                        </button>

                        {modelCardsOpen && (
                          <div id="model-cards" className="calculator-options">
                            <div className="calculator-advanced-settings">
                              <div className="calculator-weight-row">
                                <label htmlFor="calculator-weight">Вес, кг</label>
                                <input
                                  id="calculator-weight"
                                  type="number"
                                  min="1"
                                  max="300"
                                  step="0.1"
                                  value={weightKg}
                                  onChange={event => setWeightKg(event.target.value)}
                                  title="Вес — входной канал модели силы, а не только множитель для ньютонов"
                                />
                                <span>для Total GRF</span>
                              </div>
                              <div className="calculator-weight-row">
                                <label htmlFor="calculator-jump-protocol">Движение</label>
                                <select
                                  id="calculator-jump-protocol"
                                  value={jumpEventProtocol}
                                  onChange={event => setJumpEventProtocol(event.target.value)}
                                  title="Тип прыжка подаётся модели входным каналом — от него зависят найденные события"
                                >
                                  {JUMP_EVENT_PROTOCOL_OPTIONS.map(option => (
                                    <option key={option.value} value={option.value}>{option.label}</option>
                                  ))}
                                </select>
                                <span>для Jump events и Total GRF</span>
                              </div>
                              <div className="calculator-model-note">
                                Модели: <b>gct_best.pt</b>, <b>new_jump_model_byAdil.pt</b>, <b>jump_grf_total.pt</b>
                              </div>
                            </div>
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
                            {EXTRA_CALCULATORS.map(calculator => {
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
                                    {loading ? 'Детектирую…' : active ? `Убрать ${calculator.label}` : calculator.label}
                                  </button>
                                  <span className="calculator-description">{calculator.description}</span>
                                  {result?.model && (
                                    <span className="calculator-model">
                                      Модель: {result.model}{result.model_file ? ` · ${result.model_file}` : ''}
                                    </span>
                                  )}
                                  {result && (
                                    <span className="calculator-summary" style={{ '--calculator-color': calculator.color }}>
                                      {calculator.id === 'jump-events'
                                        ? <>
                                            <span>
                                              {summary?.total_jump_count || 0} прыж. · высота {formatMetric(summary?.mean_jump_height_cm, 1, ' см')}
                                              {' · макс '}{formatMetric(summary?.max_jump_height_cm, 1, ' см')}
                                            </span>
                                            <br />
                                            <span>
                                              flight {formatMetric(summary?.mean_flight_time_ms, 0, ' мс')}
                                              {' · контакт '}{formatMetric(summary?.mean_contact_time_ms, 0, ' мс')}
                                              {summary?.mean_rsi != null && ` · RSI ${summary.mean_rsi.toFixed(2)}`}
                                            </span>
                                          </>
                                        : calculator.id === 'grf-split'
                                        ? <>
                                            <span>
                                              {summary?.jump_count || 0} прыж. · пик {formatMetric(summary?.peak_force?.percent_bw, 0, ' %BW')}
                                              {' ('}{formatMetric(summary?.peak_force?.n, 0, ' Н')}{')'}
                                            </span>
                                            <br />
                                            <span>
                                              отталкивание {formatMetric(summary?.avg_pushoff_force?.percent_bw, 0, ' %BW')}
                                              {' · приземление '}{formatMetric(summary?.avg_landing_force?.percent_bw, 0, ' %BW')}
                                              {' · импульс '}{formatMetric(summary?.mean_contact_impulse_bw_s, 2, ' BW·с')}
                                            </span>
                                            <br />
                                            <span title="Цель модели — сила с плиты после low-pass фильтра; сравнивайте с колонкой Plate_Fz_total_lp20_pctBW, а не с сырой плитой">
                                              {summary?.target_lowpass_hz
                                                ? `определение: плита low-pass ${summary.target_lowpass_hz} Гц`
                                                : 'определение: сырая плита'}
                                              {summary?.event_source === 'caller'
                                                ? ' · полёты из Jump events'
                                                : summary?.event_source === 'unavailable'
                                                  ? ' · детектор прыжков недоступен'
                                                  : ' · полёты: детектор прыжков'}
                                            </span>
                                            <br />
                                            {/* The model's own card, so a landing peak is never read as measured */}
                                            <span title="Измерено по силовым платформам на атлетах, которых модель не видела">
                                              на невиданных атлетах: contact RMSE {formatMetric(summary?.held_out_contact_rmse_pctbw, 1, ' %BW')}
                                              {' · пик приземления MAE '}{formatMetric(summary?.held_out_landing_peak_mae_pctbw, 0, ' %BW')}
                                              {' · bias '}{formatMetric(summary?.held_out_landing_peak_bias_pctbw, 0, ' %BW')}
                                              {summary?.held_out_landing_peak_raw_bias_pctbw != null
                                                && ` (vs сырая плита ${formatMetric(summary.held_out_landing_peak_raw_bias_pctbw, 0, ' %BW')})`}
                                              {summary?.weight_source === 'cohort default' && ' · вес по умолчанию!'}
                                            </span>
                                          </>
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
                  onLoadedMetadata={e => { setVideoDuration(e.target.duration); e.target.playbackRate = playbackRate }}
                  onTimeUpdate={handleTimeUpdate}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
              </div>
            ) : (
              <div className="drop-hint">
                <span>📹</span>
                <p>Перетащите видео или загрузите в панели слева</p>
              </div>
            )}

            {videoUrl && (
              <div className="zoom-overlay" onMouseDown={swallowOverlayMouseDown}>
                <button type="button" tabIndex={-1} className="zoom-btn" onClick={() => changeZoom(1.25)} title="Приблизить (колесо над видео)" aria-label="Приблизить"><UiIcon name="plus" /></button>
                <span className="zoom-label">{zoom.toFixed(1)}×</span>
                <button type="button" tabIndex={-1} className="zoom-btn" onClick={() => changeZoom(1 / 1.25)} title="Отдалить" aria-label="Отдалить"><UiIcon name="minus" /></button>
                {zoom > 1 && (
                  <button type="button" tabIndex={-1} className="zoom-btn zoom-reset" onClick={resetZoom} title="Сбросить масштаб" aria-label="Сбросить масштаб"><UiIcon name="maximize" /></button>
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

          {videoUrl && (
            <div className="video-transport" role="toolbar" aria-label="Перемотка видео">
              <button type="button" className="video-step-btn" onClick={() => seekTo(0)} title="В начало · Home" aria-label="В начало">⏮</button>
              <button type="button" className="video-step-btn" onClick={() => seekBy(-COARSE_SEEK_S)} title="Назад 1 с · Shift+←" aria-label="Назад 1 секунда">−1с</button>
              <button type="button" className="video-step-btn" onClick={() => seekBy(-FINE_SEEK_S)} title="Назад 0.1 с · ←" aria-label="Назад 0.1 секунды">−0.1</button>
              <button type="button" className="video-step-btn" onClick={() => seekBy(-FRAME_STEP_S, { pause: true })} title="Кадр назад (1/30 с) · ," aria-label="Кадр назад">◂▏</button>
              <button
                type="button"
                className={`video-step-btn video-play-btn${isPlaying ? ' playing' : ''}`}
                onClick={togglePlay}
                title={isPlaying ? 'Пауза · Space' : 'Играть · Space'}
                aria-label={isPlaying ? 'Пауза' : 'Играть'}
                aria-pressed={isPlaying}
              >{isPlaying ? '❚❚' : '▶'}</button>
              <button type="button" className="video-step-btn" onClick={() => seekBy(FRAME_STEP_S, { pause: true })} title="Кадр вперёд (1/30 с) · ." aria-label="Кадр вперёд">▕▸</button>
              <button type="button" className="video-step-btn" onClick={() => seekBy(FINE_SEEK_S)} title="Вперёд 0.1 с · →" aria-label="Вперёд 0.1 секунды">+0.1</button>
              <button type="button" className="video-step-btn" onClick={() => seekBy(COARSE_SEEK_S)} title="Вперёд 1 с · Shift+→" aria-label="Вперёд 1 секунда">+1с</button>
              <button type="button" className="video-step-btn" onClick={() => seekTo(videoDuration)} title="В конец · End" aria-label="В конец">⏭</button>
              <select
                className="video-rate"
                value={playbackRate}
                onChange={e => changePlaybackRate(Number(e.target.value))}
                title="Скорость воспроизведения · < >"
                aria-label="Скорость воспроизведения"
              >
                {PLAYBACK_RATES.map(rate => <option key={rate} value={rate}>{rate}×</option>)}
              </select>
              <span className="video-transport-hint" title="Клавиши работают, когда фокус не в поле ввода">
                Space · ←→ 0.1 с · Shift 1 с · , . кадр · колесо по шкале
              </span>
            </div>
          )}

          {videoDuration > 0 && (
            <div
              className="timeline"
              ref={timelineRef}
              onMouseDown={e => { isDragging.current = true; seekFromX(e.clientX) }}
              onMouseMove={handleTimelineHover}
              onMouseLeave={handleTimelineLeave}
            >
              <div className="tl-played" style={{ width: `${cursorPct}%` }} />
              <span ref={tlHoverRef} className="tl-hover" hidden />
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
                  onClick={() => { setLabelingMode(m => !m); setActivityMode(false) }}
                  aria-pressed={labelingMode}
                  title={labelingMode ? 'Выключить режим разметки' : 'Включить режим разметки'}
                >
                  <UiIcon name="pencil" /> {labelingMode ? 'Разметка вкл' : 'Разметка'}
                </button>

                <button
                  type="button"
                  className={`btn-toggle lab-mode-btn act-mode-btn${activityMode ? ' active' : ''}`}
                  onClick={() => { setActivityMode(m => !m); setLabelingMode(false) }}
                  aria-pressed={activityMode}
                  disabled={!chartReady}
                  title={activityMode
                    ? 'Выключить разметку активностей'
                    : 'Разметить отрезки: стоит / идёт / бежит / прыгает / разворот'}
                >
                  <UiIcon name="tag" /> {activityMode ? 'Активности вкл' : 'Активности'}
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
                    {/* Force-plate ground truth. The span carries the tooltip so
                        the reason stays readable while the button is disabled. */}
                    <span className="plate-flight-ctl" title={plateFlightTitle}>
                      <button
                        type="button"
                        className={`btn-secondary btn-unwrap${plateFlightActive ? ' active' : ''}`}
                        disabled={!hasPlateColumns || plateFlightLoading}
                        onClick={() => toggleAdditionalCalculator(PLATE_FLIGHT_ID)}
                        aria-pressed={plateFlightActive}
                        aria-label={plateFlightTitle}
                      >
                        <UiIcon name={plateFlightLoading ? 'loader' : 'plate'} /> Плиты
                      </button>
                      {plateFlightActive && plateFlightResult?.contacts?.length > 0 && (
                        <button
                          type="button"
                          className="btn-secondary btn-unwrap"
                          onClick={handlePlateFlightToMarkup}
                          title="Записать полёты по плитам в разметку S1 и S2 как интервалы Target=1 (текущая разметка заменяется)"
                        >
                          <UiIcon name="pencil" /> в разметку
                        </button>
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
                {targetStepStats.detected + targetStepStats.removed > 0 && (
                  <span
                    className="lab-stat lab-stat-target"
                    title={'Из чего собирается колонка Target: шаги моделей минус удалённые вами '
                      + 'плюс интервалы, размеченные вручную. Клик по шагу на графике → «Удалить из Target» или Del.'}
                  >
                    Target: <b>{targetStepStats.total}</b>
                    <span className="lab-stat-sep">·</span>
                    модель {targetStepStats.detected}
                    {targetStepStats.removed > 0 && <> <span className="lab-stat-del">−{targetStepStats.removed}</span></>}
                    {targetStepStats.manual > 0 && <> <span className="lab-stat-add">+{targetStepStats.manual} вручную</span></>}
                  </span>
                )}
                {targetStepStats.removed > 0 && (
                  <button
                    type="button"
                    className="btn-secondary lab-btn"
                    onClick={restoreDeletedSteps}
                    title={`Вернуть в Target все ${targetStepStats.removed} удалённых шагов`}
                  >
                    <UiIcon name="undo" /> Вернуть удалённые
                  </button>
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
                    ? 'Скачать CSV данных с колонкой Target: шаги моделей за вычетом удалённых плюс ручная разметка (0, если разметки нет)'
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

            {activityMode && (
              <div className="label-toolbar-row label-toolbar-row-secondary act-toolbar">
                <div className="act-kind-group" role="group" aria-label="Вид активности">
                  {ACTIVITY_KINDS.map(kind => (
                    <button
                      key={kind.id}
                      type="button"
                      className={`act-chip${currentActivity === kind.id ? ' active' : ''}`}
                      style={{ '--act-color': kind.color }}
                      onClick={() => {
                        setCurrentActivity(kind.id)
                        if (selectedActivityIdx !== null) relabelSelectedActivity(kind.id)
                      }}
                      aria-pressed={currentActivity === kind.id}
                      title={selectedActivityIdx !== null
                        ? `Сменить метку выбранного отрезка на «${kind.label}»`
                        : `Ставить отрезки «${kind.label}» (клавиша ${kind.key})`}
                    >
                      <span className="act-swatch" /> {kind.label}
                    </button>
                  ))}
                </div>

                <div className="act-actions">
                  <button
                    type="button"
                    className="btn-secondary lab-btn"
                    onClick={undoActivitySpan}
                    disabled={!activitySpans.length && pendingActivityFrom === null}
                    title="Отменить незакрытый клик или убрать последний отрезок"
                  >
                    <UiIcon name="undo" /> Отмена
                  </button>
                  <button
                    type="button"
                    className="btn-secondary lab-btn"
                    onClick={deleteSelectedActivity}
                    disabled={selectedActivityIdx === null}
                    title="Удалить выбранный отрезок — он сольётся с предыдущим"
                  >
                    Удалить отрезок
                  </button>
                  <button
                    type="button"
                    className="btn-secondary lab-btn"
                    onClick={clearActivitySpans}
                    disabled={!activitySpans.length && pendingActivityFrom === null}
                    title="Убрать всю разметку активностей"
                  >
                    Очистить
                  </button>
                  <button
                    type="button"
                    className="btn-secondary lab-btn"
                    onClick={saveActivitiesToDb}
                    disabled={!activitySpans.length || isSaving}
                    title="Сохранить активности в сессию"
                  >
                    <UiIcon name="database" /> В БД
                  </button>
                  <button
                    type="button"
                    className="btn-secondary lab-btn"
                    onClick={exportActivities}
                    disabled={!activitySpans.length}
                    title="Скачать CSV: все каналы плюс колонка Activity"
                  >
                    <UiIcon name="download" /> CSV
                  </button>
                </div>
              </div>
            )}

            {activityMode && selectedActivityIdx !== null && activitySpans[selectedActivityIdx] && (() => {
              const span = activitySpans[selectedActivityIdx]
              const kind = ACTIVITY_BY_ID[span.activity]
              const STEP_MS = 100
              const step = timeUnit === 'ms' ? STEP_MS : STEP_MS / 1000
              return (
                <div className="label-toolbar-row label-toolbar-row-secondary act-edit-row">
                  <span className="act-edit-title" style={{ '--act-color': kind?.color }}>
                    <span className="act-swatch" /> {kind?.label}
                  </span>

                  {['from', 'to'].map(edge => (
                    <span className="act-edit-group" key={edge}>
                      <span className="act-edit-lbl">{edge === 'from' ? 'Начало' : 'Конец'}</span>
                      <button
                        type="button"
                        className="act-edit-nudge"
                        onClick={() => nudgeActivityEdge(edge, -STEP_MS)}
                        title={`Сдвинуть влево на ${step} ${timeUnit}`}
                      >←</button>
                      <OffsetInput
                        value={Number(span[edge].toFixed(timeUnit === 'ms' ? 0 : 3))}
                        step={step}
                        title={edge === 'from' ? 'Время начала' : 'Время конца'}
                        onChange={(v) => setActivityEdge(edge, v)}
                      />
                      <button
                        type="button"
                        className="act-edit-nudge"
                        onClick={() => nudgeActivityEdge(edge, STEP_MS)}
                        title={`Сдвинуть вправо на ${step} ${timeUnit}`}
                      >→</button>
                      <button
                        type="button"
                        className="act-edit-nudge act-edit-snap"
                        onClick={() => snapActivityToNeighbour(edge)}
                        title="Встык к соседнему отрезку — без щели"
                      >⇥</button>
                    </span>
                  ))}

                  {span.activity === 'turn' && (() => {
                    const measured = spanAngles[selectedActivityIdx]
                    const manual = span.angleDeg !== undefined
                    return (
                      <span className="act-edit-group act-edit-angle">
                        <span className="act-edit-lbl">Угол</span>
                        {measured === null && !manual ? (
                          <span className="act-angle-none" title="Нет курса XData на этом отрезке, либо ноги расходятся слишком сильно">
                            —
                          </span>
                        ) : (
                          <OffsetInput
                            value={Number((manual ? span.angleDeg : measured).toFixed(1))}
                            step={5}
                            title={manual
                              ? 'Угол задан вручную'
                              : 'Измерено по курсу обеих ног — можно перебить'}
                            onChange={setActivityAngle}
                          />
                        )}
                        <span className="act-angle-unit">°</span>
                        {manual && (
                          <button
                            type="button"
                            className="act-edit-nudge"
                            onClick={clearActivityAngle}
                            title="Вернуть измеренное значение"
                          >
                            <UiIcon name="undo" />
                          </button>
                        )}
                      </span>
                    )
                  })()}

                  <span className="act-edit-dur">
                    {((span.to - span.from) / (timeUnit === 'ms' ? 1000 : 1)).toFixed(2)}с
                  </span>
                </div>
              )
            })()}

            {activityMode && (activitySpans.length > 0 || pendingActivityFrom !== null) && (
              <div className="zone-dur-block act-seg-block">
                {pendingActivityFrom !== null && (
                  <span className="act-pending">
                    <span className="act-swatch" style={{ '--act-color': ACTIVITY_BY_ID[currentActivity]?.color }} />
                    Кликните конец отрезка «{ACTIVITY_BY_ID[currentActivity]?.label}»
                  </span>
                )}
                {activitySpans.map((span, i) => {
                  const kind = ACTIVITY_BY_ID[span.activity]
                  if (!kind) return null
                  const fmt = (t) => timeUnit === 'ms' ? `${(t / 1000).toFixed(2)}с` : `${t.toFixed(2)}с`
                  const dur = timeUnit === 'ms' ? (span.to - span.from) / 1000 : span.to - span.from
                  return (
                    <button
                      key={`${span.from}-${span.to}-${i}`}
                      type="button"
                      className={`act-seg${selectedActivityIdx === i ? ' active' : ''}`}
                      style={{ '--act-color': kind.color }}
                      onClick={() => setSelectedActivityIdx(prev => (prev === i ? null : i))}
                      title={`${kind.label}: ${fmt(span.from)} → ${fmt(span.to)}`}
                    >
                      <span className="act-swatch" />
                      <b>{kind.label}</b>
                      <span className="act-seg-time">{fmt(span.from)}</span>
                      <span className="act-seg-dur">{dur.toFixed(1)}с</span>
                      {spanAngles[i] !== null && spanAngles[i] !== undefined && (
                        <span className="act-seg-angle">
                          {spanAngles[i] > 0 ? '+' : ''}{spanAngles[i].toFixed(0)}°
                        </span>
                      )}
                    </button>
                  )
                })}
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
            {selectedCalculatorContact && (() => {
              const detail = selectedCalculatorContact.contact
              const calculator = CALCULATOR_BY_ID[selectedCalculatorContact.calculatorId]
              const isStep = isStepContact(detail)
              const detailStyle = isStep && selectedStepDeleted
                ? DELETED_STEP_STYLE
                : calculatorEventStyle(calculator, detail)
              const foot = detail.foot === 'left' ? 'L' : detail.foot === 'right' ? 'R' : 'ALL'
              const durationLabel = {
                flight: 'Flight',
                plate_flight: 'Flight · плиты',
                plate_mask: 'Маска',
                contact: 'GCT',
                step: 'GCT',
                turn: 'Поворот',
                run: 'Беговая фаза',
              }[detail.kind] || 'Событие'
              return (
                <div
                  className="calculator-contact-detail calculator-contact-overlay"
                  style={{ '--calculator-color': detailStyle.color }}
                >
                  <div className="calculator-contact-head">
                    <span>
                      {calculator?.label || 'ML-контакт'} · {foot} · #{selectedCalculatorContact.index + 1}
                      {isStep && selectedStepDeleted && <span className="calculator-contact-tag"> вне Target</span>}
                    </span>
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
                    {detail.status && <span><b>{detail.status}</b></span>}
                    {detail.plate_edge_time_s != null && (
                      <span title="единственный плитный край этого прыжка: приземление для прыжка на плиту, отрыв для прыжка с плиты">
                        плитный край {formatMetric(detail.plate_edge_time_s, 3, ' s')}
                      </span>
                    )}
                    {detail.impact_ratio != null && (
                      <span title="макс. TKEO |a| в 60 мс после приземления, доля от p99 сессии; гейт ≥ 0.30">
                        удар {formatMetric(detail.impact_ratio, 2, '·p99')}
                      </span>
                    )}
                    {detail.free_fall_ms2 != null && (
                      <span title="медиана min(|a| L, |a| R) в середине сегмента; полёт ≈ 9.8, гейт ≥ 7">
                        падение {formatMetric(detail.free_fall_ms2, 1, ' м/с²')}
                      </span>
                    )}
                    {detail.loaded_frac != null && (
                      <span title="доля сегмента, где стелька нагружена (> 0.30 норм. суммы); > 0.20 — стопа на полу мимо плиты">
                        стелька нагружена {formatMetric(Number(detail.loaded_frac) * 100, 0, '%')}
                      </span>
                    )}
                    {detail.direction && <span>направление {detail.direction}</span>}
                    {detail.angle_deg != null && <span>угол {formatMetric(detail.angle_deg, 0, '°')}</span>}
                    {detail.jump_height_cm != null && (
                      <span><b>высота</b> {formatMetric(detail.jump_height_cm, 1, ' см')}</span>
                    )}
                    {detail.contact_time_ms != null && (
                      <span title="от этого приземления до следующего отрыва">контакт {formatMetric(detail.contact_time_ms, 0, ' ms')}</span>
                    )}
                    {detail.rsi != null && <span>RSI {formatMetric(detail.rsi, 2, '')}</span>}
                    {/* Total GRF: peaks read off the predicted curve around this jump */}
                    {detail.pushoff_pct_bw != null && (
                      <span title="пик суммарной силы в окне перед отрывом, %BW (100 = спокойная стойка)">отталкивание {formatMetric(detail.pushoff_pct_bw, 0, ' %BW')}</span>
                    )}
                    {detail.landing_pct_bw != null && (
                      <span title="пик суммарной силы в окне после приземления, плита low-pass 20 Гц">приземление {formatMetric(detail.landing_pct_bw, 0, ' %BW')}</span>
                    )}
                    {detail.contact_impulse_bw_s != null && (
                      <span title="∫F dt от приземления до следующего отрыва, BW·с">импульс {formatMetric(detail.contact_impulse_bw_s, 2, ' BW·с')}</span>
                    )}
                  </div>
                  {isStep && (
                    <div className="calculator-contact-actions">
                      <button
                        type="button"
                        className={`calculator-contact-action${selectedStepDeleted ? ' restore' : ''}`}
                        onClick={() => toggleDeletedStep(selectedCalculatorContact)}
                        title={selectedStepDeleted
                          ? 'Вернуть шаг в колонку Target (Del)'
                          : 'Убрать шаг из колонки Target — в этом интервале станет 0 (Del)'}
                      >
                        {selectedStepDeleted ? '↩ Вернуть в Target' : '✕ Удалить из Target'}
                      </button>
                      <kbd className="calculator-contact-kbd">Del</kbd>
                    </div>
                  )}
                </div>
              )
            })()}
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
