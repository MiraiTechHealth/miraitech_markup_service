"""Force-plate ground truth for the jump model, as the markup tool sees it.

This is the labeler the plate-trained jump detector (``jump-events``) was taught
from, lifted out of the research tree so the markup tool can draw the same truth
it draws the prediction against. It is a port of ``plate_flight_v6`` from
``jump_model/jump_model_plate_v6.ipynb`` together with the session loader it sits
on (``train_plate_v2.py`` features and plate zero, ``train_plate_v3.py`` residual
resync, ``train_plate_v4.py`` merge/floor/impact gate). Constants are copied, not
imported: the research modules pull in torch and a hard-coded corpus root, and
the point here is a dependency-free companion to ``calculator_api.py``.

The recipe, per session, on a uniform 2 ms grid spanning both feet:

1. **Plate zero.** Each plate's own unloaded cluster (median of the samples within
   ``BASE_BAND_N`` of its 0.5th percentile) is subtracted, because an export
   offset of +19 N would otherwise cut straight through the 20 N flight rule.
2. **Residual resync.** One lag for both plates, found by correlating the impact
   envelope of each foot's accelerometer against its plate (coarse 4 ms then
   1 ms, within ±``RESYNC_HALF_MS``). Plate-synced parquets are already close;
   this removes what the hand sync left.
3. **Flight candidates.** Both plates below ``FZ_THR`` newtons. Gaps of at most
   ``GT_MERGE_MS`` are stitched (threshold jitter), runs shorter than
   ``GT_FLOOR_MS`` are dropped (60 ms is 4 mm of height - not a jump).
4. **Insole on the floor.** If more than ``CONTAM_MAX`` of a candidate has an
   insole loaded above ``CONTAM_THR`` (normalised pressure sum), the athlete is
   standing beside the plates: the loaded part is an honest negative, the
   unloaded part is unknown and masked.
5. **Impact OR free fall.** A candidate is a jump when the landing edge is
   followed within ``IMPACT_WIN_MS`` by an accelerometer TKEO burst of at least
   ``IMPACT_MIN`` of the session's 99th percentile (v4), **or** when the median of
   ``min(|a| left, |a| right)`` over the middle half of the segment is at least
   ``FF_MIN`` m/s² - both feet in free fall (v6, recovers toe-first landings
   whose heel strike comes 70-150 ms late). Neither: unknown, masked with a
   ``INVALID_MARGIN_MS`` margin on each side.
6. **The insole lies (v7).** An "insole on the floor" candidate in which both
   feet are in free fall right after the take-off edge, right before the landing
   edge AND through the middle (each ≥ ``FF_MIN``), no longer than
   ``LIES_MAX_MS``, is a plate-to-plate jump with both edges exact: the plates
   are authoritative, the loaded insole is the toes pressing in the air, a slow
   unload or a drifted baseline. Accepted as a jump. v6 threw 42 such jumps away
   (27 sessions, 9 athletes) and charged the model false positives for finding them.
7. **Jumps across the plate edge (v7).** Otherwise, inside an "insole on the
   floor" candidate the plates cannot see the athlete at all, so a jump that
   starts beside the plates and lands on them shows only its landing edge, and a
   jump off the plates only its take-off. Both feet in free fall for
   ``HOP_WIN_MS`` right before the landing edge (or right after the take-off
   edge) marks such a hop; ``HOP_MASK_MS`` of the candidate next to that edge
   becomes unknown instead of ground. Without this the v6 labels taught the
   model that a hop onto the plate is *not* a jump whenever the insole did not
   unload below the threshold - a common baseline drift at the start and end of
   a recording.

Input is the column-wise session frame the markup tool uploads: ``Name`` already
canonicalised to ``ESP32_Sensor_1`` (left) / ``ESP32_Sensor_2`` (right), ``Time``
in milliseconds, the IMU and insole channels, and the plate force of that foot in
newtons in one of ``Plate_Fz_N`` (``export_markup_parquet.py``) or ``1:Fz`` /
``2:Fz`` (the synced research corpus, where each foot's rows carry one of the
two). Output is plain dicts, ready for JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# ── constants, copied from jump_model/train_plate_v2.py … v4.py and the v6 notebook ──
FS, DT_MS = 500.0, 2.0
FZ_THR = 20.0             # N: both plates below this = airborne candidate
BASE_BAND_N = 50.0        # N: width of the unloaded-plate cluster used for the zero
GT_MERGE_MS = 10.0        # stitch only plate blips this short (jitter at the 20 N edge)
GT_FLOOR_MS = 60.0        # physical floor: shorter is not flight
IMPACT_WIN_MS = 60.0      # window after the landing edge to look for the impact
IMPACT_MIN = 0.30         # fraction of the session's p99 impact energy
CONTAM_THR = 0.30         # normalised insole pressure sum above which the foot is loaded
CONTAM_MAX = 0.20         # fraction of a candidate that may be loaded before it is "beside the plate"
INVALID_MARGIN_MS = 100.0 # loss-mask margin around an unknown segment
FF_MIN = 7.0              # m/s²: median min(|a|L,|a|R) mid-segment; flight ≈ 9.8, standing beside ≈ 1
HOP_WIN_MS = 200.0        # v7: window at the plate edge where both-feet free fall marks a hop onto / off the plate
HOP_MASK_MS = 700.0       # v7: how much of the candidate is masked back from that edge (p99 plate flight 566 ms + margin)
LIES_MAX_MS = 800.0       # v7: "insole lies" accepted only up to this length - nobody is airborne longer
RESYNC_HALF_MS = 300.0    # search half-window for the residual plate lag
TKEO_WIN = 15             # samples: the impact-energy channel is tkeo_acc_15
MIN_SPAN_MS = 3000.0      # both feet must overlap at least this long
GRAVITY = 9.80665

PRESS_COLS = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4"]
ACC_COLS = ["AcX", "AcY", "AcZ"]
SENSOR_TO_SIDE = {"ESP32_Sensor_1": "L", "ESP32_Sensor_2": "R"}
PLATE_FORCE_COLS = ("Plate_Fz_N", "1:Fz", "2:Fz")

STATUS_BOTH = "прыжок: удар и падение"
STATUS_IMPACT = "прыжок: только удар"
STATUS_FREEFALL = "прыжок: только падение"
STATUS_NOGATE = "маска: ни удара, ни падения"
STATUS_LOADED = "маска: стелька на полу мимо плиты"
STATUS_HOP_ONTO = "маска: прыжок с пола на плиту — отрыв плитой не виден"
STATUS_HOP_OFF = "маска: прыжок с плиты на пол — приземление плитой не видно"
STATUS_LIES = "прыжок: свободное падение при нагруженной стельке (стелька врёт, плиты главнее)"

LABEL_VERSION = "v7"

PARAMS = dict(
    label_version=LABEL_VERSION,
    dt_ms=DT_MS, fz_thr_n=FZ_THR, base_band_n=BASE_BAND_N, merge_ms=GT_MERGE_MS,
    floor_ms=GT_FLOOR_MS, impact_win_ms=IMPACT_WIN_MS, impact_min=IMPACT_MIN,
    contam_thr=CONTAM_THR, contam_max=CONTAM_MAX, invalid_margin_ms=INVALID_MARGIN_MS,
    ff_min_ms2=FF_MIN, hop_win_ms=HOP_WIN_MS, hop_mask_ms=HOP_MASK_MS, lies_max_ms=LIES_MAX_MS,
    resync_half_ms=RESYNC_HALF_MS, tkeo_win=TKEO_WIN,
)


class PlateFlightError(ValueError):
    """The frame cannot be labelled; the message says why, in the operator's words."""


# ───────────────────────────── small helpers (train_plate_v2) ─────────────────────────────
def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of the True runs of a boolean vector."""
    dd = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.where(dd == 1)[0], np.where(dd == -1)[0]))


def plate_baseline(fz: np.ndarray) -> float:
    """Plate zero from its own record: median of the UNLOADED cluster.

    The band is physical (``BASE_BAND_N`` above the 0.5th percentile), not a
    fraction of the range - a landing impact pushes p99 to 1300 N and a "25 % of
    range" cut used to swallow quiet standing (~320 N) into the zero. If the
    plate is never unloaded the minimum is returned: a shifted edge beats lost
    jumps.
    """
    f = np.asarray(fz, float)
    f = f[np.isfinite(f)]
    if not len(f):
        return 0.0
    lo = float(np.percentile(f, 0.5))
    low = f[f < lo + BASE_BAND_N]
    return float(np.median(low)) if len(low) else lo


def tkeo(x: np.ndarray, win: int) -> np.ndarray:
    """Teager-Kaiser energy, centred rolling mean over ``win`` samples, clipped at 0."""
    x = np.asarray(x, dtype=np.float64)
    psi = np.zeros_like(x)
    if len(x) > 2:
        psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    psi = pd.Series(psi).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return np.clip(psi, 0.0, None)


def norm_p05_p95(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), -0.2, 1.2).astype(np.float32)


# ───────────────────────────── residual resync (sync_lib + train_plate_v3) ─────────────────────────────
def _tkeo_abs(x: np.ndarray) -> np.ndarray:
    e = np.zeros_like(x)
    e[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return np.abs(e)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def _env(t: np.ndarray, y: np.ndarray, step: float = 4.0, smooth_ms: float = 60.0,
         hp_ms: float = 400.0) -> Tuple[np.ndarray, np.ndarray]:
    """Impact-energy envelope on a uniform grid, z-scored."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    tg = np.arange(float(t.min()), float(t.max()), step)
    yg = np.interp(tg, t, y)
    yg = yg - _smooth(yg, max(3, int(hp_ms / step)) | 1)
    e = _smooth(_tkeo_abs(yg), max(3, int(smooth_ms / step)) | 1)
    med = np.median(e[e > 0]) if np.any(e > 0) else 1.0
    e = np.log1p(e / med)
    return tg, (e - e.mean()) / (e.std() + 1e-9)


def _corr_at(ti, ei, tp, ep, lag: float, step: float = 1.0) -> float:
    """Pearson on the overlap at the given plate lag."""
    t0, t1 = max(ti[0], tp[0] + lag), min(ti[-1], tp[-1] + lag)
    if t1 - t0 < 1000:
        return -1e9
    g = np.arange(t0, t1, step)
    a = np.interp(g, ti, ei)
    b = np.interp(g, tp + lag, ep)
    return float(np.dot(a - a.mean(), b - b.mean()) / (len(g) * (a.std() + 1e-9) * (b.std() + 1e-9)))


def joint_resync(feet: Dict[str, Tuple[pd.DataFrame, str]]) -> Tuple[float, float]:
    """One plate lag for both feet: per-foot envelopes, correlation summed over feet."""
    envs = []
    for g, col in feet.values():
        t = g["Time"].to_numpy(float)
        acc = np.sqrt(sum(pd.to_numeric(g[c], errors="coerce").to_numpy(float) ** 2 for c in ACC_COLS))
        fz = np.nan_to_num(pd.to_numeric(g[col], errors="coerce").to_numpy(float))
        ok = np.isfinite(t) & np.isfinite(acc)
        ti, ei = _env(t[ok], acc[ok])
        tp, ep = _env(t, fz)
        envs.append((ti, ei, tp, ep))

    def score(lag: float) -> float:
        return sum(_corr_at(ti, ei, tp, ep, lag) for ti, ei, tp, ep in envs)

    lags = np.arange(-RESYNC_HALF_MS, RESYNC_HALF_MS + 4.0, 4.0)
    best = lags[int(np.argmax([score(l) for l in lags]))]
    lags2 = np.arange(best - 4.0, best + 5.0, 1.0)
    sc2 = [score(l) for l in lags2]
    k = int(np.argmax(sc2))
    return float(lags2[k]), float(sc2[k] / len(envs))


# ───────────────────────────── per-foot channels on the common grid ─────────────────────────────
def foot_channels(g: pd.DataFrame, t: np.ndarray) -> Dict[str, np.ndarray]:
    """The three channels the labeler reads, interpolated onto the grid, as float32
    like the training features (so thresholds fall on the same side)."""
    tt = g["Time"].to_numpy(float)
    acc = {c: np.interp(t, tt, pd.to_numeric(g[c], errors="coerce").to_numpy(float)) for c in ACC_COLS}
    acc_mag = np.sqrt(acc["AcX"] ** 2 + acc["AcY"] ** 2 + acc["AcZ"] ** 2)
    press = np.stack([np.interp(t, tt, pd.to_numeric(g[c], errors="coerce").to_numpy(float))
                      for c in PRESS_COLS], 1)
    ssr = norm_p05_p95(press.sum(1))
    f32 = lambda v: np.nan_to_num(np.asarray(v, np.float32))
    return {"acc_mag": f32(acc_mag), "ssr": f32(ssr), "tkeo_acc": f32(tkeo(acc_mag, TKEO_WIN))}


# ───────────────────────────── the v6 label ─────────────────────────────
def plate_flight_v7(t, fzL, fzR, ssrL, ssrR, tk, amL, amR) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """v4 labelling with the v6 "impact OR free fall" gate and the v7 hop mask.
    → (segments, stats).

    Each segment is one run of "both plates unloaded" that survived the floor,
    with ``cls`` in {"jump", "mask"} and the numbers the gate looked at. For an
    "insole on the floor" run only its unknown parts are emitted (as masks): the
    hop windows next to a plate edge the athlete jumped across, and the unloaded
    sub-runs outside them. The loaded remainder is an honest negative and is not
    drawn. Jumps are identical to the v6 labels; only the masks differ.
    """
    air = (fzL < FZ_THR) & (fzR < FZ_THR)
    m = air.copy()
    g = int(round(GT_MERGE_MS / DT_MS))
    for a, b in runs(~m):
        if b - a <= g and a > 0 and b < len(m):
            m[a:b] = True
    loaded = (ssrL > CONTAM_THR) | (ssrR > CONTAM_THR)
    hi = float(np.percentile(tk, 99)) + 1e-9
    am = np.minimum(amL, amR)
    margin = int(round(INVALID_MARGIN_MS / DT_MS))
    floor = GT_FLOOR_MS / DT_MS
    win = int(round(IMPACT_WIN_MS / DT_MS))
    hw, hm = int(round(HOP_WIN_MS / DT_MS)), int(round(HOP_MASK_MS / DT_MS))
    st = dict(n_air=0, n_short=0, n_contam=0, n_noimpact=0, n_jumps=0,
              n_ff_only=0, n_imp_only=0, n_both=0, n_lies=0, n_hop_onto=0, n_hop_off=0,
              hop_neg_ms=0.0, durs=[], tk_p99=hi)
    segs: List[Dict[str, Any]] = []
    n = len(t)

    def t_end(i: int) -> float:  # inclusive end of [.., i) as a timestamp
        return float(t[min(i, n) - 1])

    for a, b in runs(m):
        st["n_air"] += 1
        if b - a < floor:
            st["n_short"] += 1
            continue
        lf = float(loaded[a:b].mean())
        imp = float(tk[b:min(len(tk), b + win)].max() / hi) if b < len(tk) else 0.0
        m0, m1 = a + (b - a) // 4, b - (b - a) // 4
        ff = float(np.median(am[m0:m1]))
        base = dict(t0=float(t[a]), t1=t_end(b), dur_ms=(b - a) * DT_MS, loaded_frac=lf, imp=imp, ff=ff)
        ff_tail = float(np.median(am[max(a, b - hw):b]))
        ff_head = float(np.median(am[a:min(b, a + hw)]))
        # Both plates give both edges and both feet are airborne from edge to
        # edge: a plate-to-plate jump whatever the insole says (toes pressing in
        # the air, slow unload, drifted baseline). The plates are authoritative.
        insole_lies = (lf > CONTAM_MAX and ff_head >= FF_MIN and ff_tail >= FF_MIN
                       and ff >= FF_MIN and (b - a) * DT_MS <= LIES_MAX_MS)
        if lf > CONTAM_MAX and not insole_lies:
            # Athlete beside the plates for most of the run - or jumping across
            # their edge. Both feet in free fall right before the landing edge:
            # a hop ONTO the plate, the plates saw only its landing. Right after
            # the take-off edge: a hop OFF the plate, only its take-off. Either
            # way there is no second plate edge to label, so the flight-like part
            # is unknown, not ground (v6 sent its loaded samples to the loss as
            # negatives - "a jump from the floor is not a jump").
            st["n_contam"] += 1
            unk = ~loaded[a:b]
            hop = np.zeros(b - a, bool)
            if ff_tail >= FF_MIN:
                st["n_hop_onto"] += 1
                lo, hi_ = max(a, b - hm), min(n, b + margin)
                hop[lo - a:] = True
                segs.append(dict(base, cls="mask", status=STATUS_HOP_ONTO, hop="onto",
                                 t0=float(t[lo]), t1=float(t[hi_ - 1]), dur_ms=(hi_ - lo) * DT_MS,
                                 ff=ff_tail, edge_t=t_end(b + 1)))
            if ff_head >= FF_MIN:
                st["n_hop_off"] += 1
                lo, hi_ = max(0, a - margin), min(b, a + hm)
                hop[:hi_ - a] = True
                segs.append(dict(base, cls="mask", status=STATUS_HOP_OFF, hop="off",
                                 t0=float(t[lo]), t1=float(t[hi_ - 1]), dur_ms=(hi_ - lo) * DT_MS,
                                 ff=ff_head, edge_t=float(t[a])))
            # Flight-like samples (free-fall window at the edge) the v6 labels sent
            # to the loss as ground. The mask itself is wider.
            ffwin = np.zeros(b - a, bool)
            if ff_tail >= FF_MIN:
                ffwin[max(0, b - a - hw):] = True
            if ff_head >= FF_MIN:
                ffwin[:min(b - a, hw)] = True
            st["hop_neg_ms"] += float((ffwin & ~unk).sum()) * DT_MS
            rest = unk & ~hop
            for c, d in runs(rest):
                segs.append(dict(base, cls="mask", status=STATUS_LOADED,
                                 t0=float(t[a + c]), t1=float(t[a + d - 1]), dur_ms=(d - c) * DT_MS))
            continue
        ok_imp, ok_ff = imp >= IMPACT_MIN, ff >= FF_MIN
        if not (ok_imp or ok_ff):
            st["n_noimpact"] += 1
            lo, hi_ = max(0, a - margin), min(n, b + margin)
            segs.append(dict(base, cls="mask", status=STATUS_NOGATE,
                             t0=float(t[lo]), t1=float(t[hi_ - 1]), dur_ms=(hi_ - lo) * DT_MS))
            continue
        st["n_jumps"] += 1
        st["durs"].append((b - a) * DT_MS)
        if insole_lies:
            st["n_lies"] += 1
            status = STATUS_LIES
        elif ok_imp and ok_ff:
            st["n_both"] += 1
            status = STATUS_BOTH
        elif ok_imp:
            st["n_imp_only"] += 1
            status = STATUS_IMPACT
        else:
            st["n_ff_only"] += 1
            status = STATUS_FREEFALL
        segs.append(dict(base, cls="jump", status=status))
    return segs, st


# ───────────────────────────── session frame → label ─────────────────────────────
def _plate_column(g: pd.DataFrame) -> str | None:
    """The column holding this foot's plate force in newtons, or None."""
    for col in PLATE_FORCE_COLS:
        if col in g.columns and pd.to_numeric(g[col], errors="coerce").notna().any():
            return col
    return None


def has_plate_columns(columns: Sequence[str]) -> bool:
    return any(c in columns for c in PLATE_FORCE_COLS)


def label_frame(df: pd.DataFrame) -> Dict[str, Any]:
    """Run the v6 labeler on a canonicalised session frame (Time in ms).

    Raises ``PlateFlightError`` when the frame lacks what the label needs.
    """
    if "Name" not in df.columns or "Time" not in df.columns:
        raise PlateFlightError("В данных нет колонок Name и Time")
    missing = [c for c in ACC_COLS + PRESS_COLS if c not in df.columns]
    if missing:
        raise PlateFlightError(f"Нет каналов для гейта: {', '.join(missing)}")

    feet: Dict[str, Tuple[pd.DataFrame, str]] = {}
    for name, g in df.groupby("Name"):
        side = SENSOR_TO_SIDE.get(str(name))
        if side is None:
            continue
        g = g.copy()
        g["Time"] = pd.to_numeric(g["Time"], errors="coerce")
        g = g[g["Time"].notna()].sort_values("Time")
        if not len(g):
            continue
        col = _plate_column(g)
        if col is None:
            raise PlateFlightError(
                f"У {name} нет силы плиты: нужна колонка {' / '.join(PLATE_FORCE_COLS)} в ньютонах")
        feet[side] = (g, col)
    if len(feet) < 2:
        raise PlateFlightError("Нужны обе стопы (ESP32_Sensor_1 и ESP32_Sensor_2) с силой плиты")

    t0 = max(g["Time"].min() for g, _ in feet.values())
    t1 = min(g["Time"].max() for g, _ in feet.values())
    if t1 - t0 < MIN_SPAN_MS:
        raise PlateFlightError(f"Перекрытие стоп {max(0.0, (t1 - t0)) / 1000:.1f} с — меньше {MIN_SPAN_MS / 1000:.0f} с")
    t = np.arange(np.ceil(t0 / DT_MS) * DT_MS, t1, DT_MS)

    resync_ms, resync_r = joint_resync(feet)
    chans, fz, base = {}, {}, {}
    for side, (g, col) in feet.items():
        chans[side] = foot_channels(g, t)
        v = np.interp(t, g["Time"].to_numpy(float) + resync_ms,
                      pd.to_numeric(g[col], errors="coerce").to_numpy(float))
        base[side] = plate_baseline(v)
        fz[side] = (v - base[side]).astype(np.float32)

    tk = chans["L"]["tkeo_acc"] + chans["R"]["tkeo_acc"]
    segs, st = plate_flight_v7(t, fz["L"], fz["R"], chans["L"]["ssr"], chans["R"]["ssr"], tk,
                               chans["L"]["acc_mag"], chans["R"]["acc_mag"])
    return dict(
        segments=segs, stats=st, t0=float(t[0]), t1=float(t[-1]), n_samples=int(len(t)),
        resync_ms=resync_ms, resync_r=resync_r,
        plate_base_n={"left": base["L"], "right": base["R"]},
        plate_cols={"left": feet["L"][1], "right": feet["R"][1]},
    )


def _r(v: Any, digits: int = 2) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def plate_flight_result(df: pd.DataFrame, label: str, model: str) -> Dict[str, Any]:
    """The companion-API response: bilateral contacts + a summary, in the shape the
    markup tool already draws for ``jump-events``."""
    r = label_frame(df)
    contacts = []
    for seg in r["segments"]:
        flight_s = seg["dur_ms"] / 1000.0
        is_jump = seg["cls"] == "jump"
        contacts.append({
            "foot": None,
            "start_time_s": seg["t0"] / 1000.0,
            "end_time_s": seg["t1"] / 1000.0,
            "peak_time_s": seg["t0"] / 1000.0,
            "duration_ms": _r(seg["dur_ms"], 1),
            "kind": "plate_flight" if is_jump else "plate_mask",
            "status": seg["status"],
            "hop": seg.get("hop"),
            # For a hop the one plate edge that IS exact: landing for "onto", take-off for "off".
            "plate_edge_time_s": seg["edge_t"] / 1000.0 if "edge_t" in seg else None,
            "impact_ratio": _r(seg["imp"], 3),
            "free_fall_ms2": _r(seg["ff"], 2),
            "loaded_frac": _r(seg["loaded_frac"], 3),
            "jump_height_cm": _r(GRAVITY * flight_s ** 2 / 8.0 * 100.0, 1) if is_jump else None,
            "confidence": None,
        })
    st = r["stats"]
    durs = st["durs"]
    return {
        "calculator": "plate-flight",
        "label": label,
        "model": model,
        "model_file": None,
        "contacts": contacts,
        "summary": {
            "total_jump_count": st["n_jumps"],
            "flight_count": st["n_jumps"],
            "event_count": len(contacts),
            "jumps_impact_and_free_fall": st["n_both"],
            "jumps_impact_only": st["n_imp_only"],
            "jumps_free_fall_only": st["n_ff_only"],
            # plate-to-plate jumps accepted although an insole read "loaded" (v6 threw these away)
            "jumps_insole_lies": st["n_lies"],
            "masked_no_gate": st["n_noimpact"],
            "masked_insole_loaded": st["n_contam"],
            "hops_onto_plate": st["n_hop_onto"],
            "hops_off_plate": st["n_hop_off"],
            # flight-like samples inside hop windows that the v6 labels sent to the loss as ground
            "hop_negatives_in_v6_ms": _r(st["hop_neg_ms"], 0),
            "label_version": LABEL_VERSION,
            "air_runs": st["n_air"],
            "air_runs_below_floor": st["n_short"],
            "mean_flight_time_ms": _r(np.mean(durs), 1) if durs else None,
            "max_flight_time_ms": _r(np.max(durs), 1) if durs else None,
            "mean_jump_height_cm": _r(np.mean([GRAVITY * (d / 1000.0) ** 2 / 8.0 * 100.0 for d in durs]), 1) if durs else None,
            "resync_ms": _r(r["resync_ms"], 0),
            "resync_r": _r(r["resync_r"], 3),
            "plate_zero_n": {k: _r(v, 1) for k, v in r["plate_base_n"].items()},
            "plate_columns": r["plate_cols"],
            "impact_p99": _r(st["tk_p99"], 3),
            "span_s": _r((r["t1"] - r["t0"]) / 1000.0, 1),
            "is_valid": st["n_jumps"] > 0,
        },
        "params": PARAMS,
    }
