// Activity segmentation of a session (stand / walk / run / ... spans).

// ── Activity segmentation ──────────────────────────────────────────────────
// Free-mode recordings are labelled with independent [from, to] spans, marked
// the way contacts are: click the start, click the end. Unmarked stretches stay
// unlabelled rather than being guessed at, and export them with an empty
// Activity so the training set can drop or keep them deliberately.
export const ACTIVITY_KINDS = [
  { id: 'stand', label: 'Стоит',    color: '#64748b', key: '1' },
  { id: 'walk',  label: 'Идёт',     color: '#2ca02c', key: '2' },
  { id: 'run',   label: 'Бежит',    color: '#ff7f0e', key: '3' },
  { id: 'jump',  label: 'Прыгает',  color: '#9467bd', key: '4' },
  { id: 'turn',  label: 'Разворот', color: '#17becf', key: '5' },
  // Putting the insoles on, walking back to the phone: real signal, but not an
  // exercise. Kept as a label rather than a gap so the ribbon stays unbroken
  // and the export can drop these rows deliberately.
  { id: 'junk',  label: 'Мусор',    color: '#d62728', key: '6' },
]

export const ACTIVITY_BY_ID = Object.fromEntries(ACTIVITY_KINDS.map(a => [a.id, a]))

export const DEFAULT_ACTIVITY = 'stand'

export const ACTIVITY_FILE_TYPE = 'activity_segments'

/** markup_files holds both markups; the contact UI must not see the ribbon. */
export function contactMarkupFiles(files) {
  return (files || []).filter(f => f.type !== ACTIVITY_FILE_TYPE)
}

/** Load stored spans back into plot time by re-adding the saved offset. */
export function activitySpansFromFiles(files, offset) {
  const file = (files || []).find(f => f.type === ACTIVITY_FILE_TYPE)
  if (!file || !Array.isArray(file.activitySpans)) return []
  return file.activitySpans
    .map(sp => ({
      from: Number(sp.from) + offset,
      to: Number(sp.to) + offset,
      activity: sp.activity,
      ...(Number.isFinite(Number(sp.angleDeg)) ? { angleDeg: Number(sp.angleDeg) } : {}),
    }))
    .filter(sp => Number.isFinite(sp.from) && Number.isFinite(sp.to)
      && sp.to > sp.from && ACTIVITY_BY_ID[sp.activity])
    .sort((a, b) => a.from - b.from)
}

/** Close the pending span at t, ordering the ends so a backwards drag works. */
export function closeActivitySpan(pendingFrom, t, activity) {
  return {
    from: Math.min(pendingFrom, t),
    to: Math.max(pendingFrom, t),
    activity,
  }
}

/** Index of the span containing t, or -1. Later spans win where they overlap. */
export function activitySpanAt(spans, t) {
  for (let i = spans.length - 1; i >= 0; i--) {
    if (t >= spans[i].from && t <= spans[i].to) return i
  }
  return -1
}

// Hitting an exact boundary by eye is not realistic, so a new edge lands on a
// neighbour's edge when it is within this fraction of the visible span. It is
// relative to the zoom, not absolute, so it stays usable at every scale.
export const ACTIVITY_SNAP_FRACTION = 0.012

/** Pull t onto the nearest existing edge within tolerance, else leave it be. */
export function snapActivityEdge(spans, t, tolerance, skipIndex = -1) {
  if (!(tolerance > 0)) return t
  let best = t
  let bestGap = tolerance
  spans.forEach((span, i) => {
    if (i === skipIndex) return
    for (const edge of [span.from, span.to]) {
      const gap = Math.abs(edge - t)
      if (gap < bestGap) { bestGap = gap; best = edge }
    }
  })
  return best
}

/**
 * Insert a span, carving it out of whatever it overlaps: an activity happens at
 * one time or another, never both at once. Older spans are trimmed, split when
 * the newcomer lands inside them, and dropped when fully covered.
 */
export function insertActivitySpan(spans, incoming) {
  const out = []
  spans.forEach(span => {
    if (span.to <= incoming.from || span.from >= incoming.to) { out.push(span); return }
    // Covered entirely — the old span disappears.
    if (span.from >= incoming.from && span.to <= incoming.to) return
    // Split in two: the newcomer sits strictly inside.
    if (span.from < incoming.from && span.to > incoming.to) {
      out.push({ ...span, to: incoming.from })
      out.push({ ...span, from: incoming.to })
      return
    }
    out.push(span.from < incoming.from
      ? { ...span, to: incoming.from }
      : { ...span, from: incoming.to })
  })
  out.push(incoming)
  return out.sort((a, b) => a.from - b.from)
}

/** Move one edge of a span, refusing to invert it or cross a neighbour. */
export function resizeActivitySpan(spans, index, edge, t) {
  const span = spans[index]
  if (!span) return spans
  const others = spans.filter((_, i) => i !== index)
  const MIN = 1e-6
  let next
  if (edge === 'from') {
    const limit = others.reduce(
      (acc, o) => (o.to <= span.to && o.to > acc ? o.to : acc), -Infinity)
    next = { ...span, from: Math.min(Math.max(t, limit), span.to - MIN) }
  } else {
    const limit = others.reduce(
      (acc, o) => (o.from >= span.from && o.from < acc ? o.from : acc), Infinity)
    next = { ...span, to: Math.max(Math.min(t, limit), span.from + MIN) }
  }
  const copy = [...spans]
  copy[index] = next
  return copy
}
