import { useState, useRef, useEffect } from 'react'
import { formatTimeOffset, hasSessionMetaValue } from '../lib/format.js'

export function SidebarSection({ title, open, onToggle, children }) {
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

export function UploadBtn({ accept, onFile, children, className = 'btn-upload btn-secondary', disabled = false, title }) {
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

export function UiIcon({ name, className = '' }) {
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
    case 'plate':
      // Two force plates side by side with a foot-off arrow above them.
      artwork = <><rect x="2.5" y="15" width="8" height="5" rx="1" /><rect x="13.5" y="15" width="8" height="5" rx="1" /><path d="M12 12V4" /><path d="m8.5 7.5 3.5-3.5 3.5 3.5" /></>
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
export function SessionTitleBadge({ title, expanded, onToggle, onSave }) {
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

export function FileBadge({ type, title, children }) {
  return <span className={`file-badge badge-${type}`} title={title}>{children}</span>
}

export function SessionInfoCard({ protocolName, deviceId, timeOffset, gapCount, gapsKnown }) {
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
        <span className="session-info-key">Time offset</span>
        <span className="session-info-val">{formatTimeOffset(timeOffset) || '—'}</span>
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

export function OffsetInput({ value, step, title, onChange }) {
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
