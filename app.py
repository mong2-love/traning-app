import streamlit as st
import sqlite3
import os
import tempfile
import math
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from io import BytesIO
import calendar

ZONE_COLORS = ["#B5D4F4", "#9FE1CB", "#FAC775", "#F0997B", "#E24B4A"]
ZONE_NAMES  = ["Z1 회복", "Z2 유산소", "Z3 템포", "Z4 역치", "Z5 최대"]

DB_PATH     = "training_log.db"
REPORTS_DIR = "reports"

# ── DB 초기화 ──────────────────────────────────────────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT,
                filename        TEXT UNIQUE,
                sport           TEXT,
                indoor          INTEGER,
                course_name     TEXT,
                distance_km     REAL,
                duration_sec    INTEGER,
                moving_time_sec INTEGER,
                stop_time_sec   INTEGER,
                avg_speed_kph   REAL,
                avg_hr          REAL,
                max_hr          REAL,
                drift           REAL,
                drift_grade     TEXT,
                avg_power       REAL,
                max_power       REAL,
                avg_cadence     REAL,
                w_per_bpm       REAL,
                spike_count     INTEGER,
                spike_corrected INTEGER,
                z1_pct          REAL,
                z2_pct          REAL,
                z3_pct          REAL,
                z4_pct          REAL,
                z5_pct          REAL,
                avg_pace        REAL,
                best_pace       REAL,
                pace_drift_sec  REAL,
                calories        REAL,
                indoor_temp     REAL,
                cad_lt70        REAL,
                cad_70_80       REAL,
                cad_80_90       REAL,
                cad_90_100      REAL,
                cad_100p        REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_raw (
                filename  TEXT PRIMARY KEY,
                raw_json  TEXT,
                laps_json TEXT
            )
        """)
        # 기존 DB 마이그레이션
        for col, tbl, typ in [
            ("laps_json",       "training_raw", "TEXT"),
            ("course_name",     "training_log", "TEXT"),
            ("moving_time_sec", "training_log", "INTEGER"),
            ("stop_time_sec",   "training_log", "INTEGER"),
            ("avg_speed_kph",   "training_log", "REAL"),
            ("drift_grade",     "training_log", "TEXT"),
            ("spike_count",     "training_log", "INTEGER"),
            ("spike_corrected", "training_log", "INTEGER"),
            ("best_pace",       "training_log", "REAL"),
            ("pace_drift_sec",  "training_log", "REAL"),
            ("indoor_temp",     "training_log", "REAL"),
            ("cad_lt70",        "training_log", "REAL"),
            ("cad_70_80",       "training_log", "REAL"),
            ("cad_80_90",       "training_log", "REAL"),
            ("cad_90_100",      "training_log", "REAL"),
            ("cad_100p",        "training_log", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
            except Exception:
                pass


# ── 심박 존 계산 ────────────────────────────────────────────────────────────────
def hr_zones(max_hr):
    return [
        (0,           int(max_hr * 0.60)),
        (int(max_hr * 0.60), int(max_hr * 0.70)),
        (int(max_hr * 0.70), int(max_hr * 0.80)),
        (int(max_hr * 0.80), int(max_hr * 0.90)),
        (int(max_hr * 0.90), int(max_hr * 1.00)),
    ]


def assign_zone(bpm, zones):
    for i, (lo, hi) in enumerate(zones):
        if lo <= bpm <= hi:
            return i + 1
    return 5


def _drift_grade(drift):
    if drift is None or (isinstance(drift, float) and math.isnan(drift)):
        return None
    if abs(drift) <= 4: return "정상"
    if abs(drift) <= 8: return "주의"
    return "피로"


def _cad_zones(cad_series: pd.Series) -> dict:
    cad = cad_series[cad_series > 0].dropna()
    total = len(cad)
    if total == 0:
        return dict(cad_lt70=0.0, cad_70_80=0.0, cad_80_90=0.0, cad_90_100=0.0, cad_100p=0.0)
    def pct(mask): return round(float(mask.sum()) / total * 100, 1)
    return dict(
        cad_lt70   = pct(cad < 70),
        cad_70_80  = pct((cad >= 70) & (cad < 80)),
        cad_80_90  = pct((cad >= 80) & (cad < 90)),
        cad_90_100 = pct((cad >= 90) & (cad < 100)),
        cad_100p   = pct(cad >= 100),
    )


def fmt_duration(seconds):
    if pd.isna(seconds) or seconds is None:
        return "-"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_pace(sec_per_km):
    if pd.isna(sec_per_km) or sec_per_km is None or sec_per_km <= 0:
        return "-"
    m, s = divmod(int(sec_per_km), 60)
    return f"{m}'{s:02d}\""


# ── FIT 파싱 ────────────────────────────────────────────────────────────────────
# def_num → 내부 컬럼명  (fitparse가 scale/offset 자동 적용)
# 253=timestamp, 3=heart_rate, 4=cadence, 5=distance(m),
# 6=speed(m/s), 7=power(W), 2=altitude(m)
_FIT_DEF_MAP  = {253: "timestamp", 3: "hr", 4: "cad",
                 5: "distance",    6: "speed", 7: "watts", 2: "altitude", 13: "temp"}
_FIT_NAME_MAP = {"timestamp": "timestamp", "heart_rate": "hr", "cadence": "cad",
                 "distance": "distance",   "speed": "speed",
                 "power": "watts",         "altitude": "altitude",
                 "temperature": "temp"}


def parse_fit(path: str, max_hr: int, ftp: int):
    try:
        from fitparse import FitFile
    except ImportError:
        st.error("fitparse 패키지가 설치되지 않았습니다.")
        return None

    try:
        ff = FitFile(path)
    except Exception as e:
        st.error(f"FIT 파일 읽기 실패: {e}")
        return None

    records      = []
    session_data = {}

    for msg in ff.get_messages():
        if msg.name == "record":
            row = {}
            for f in msg.fields:
                col = _FIT_DEF_MAP.get(f.def_num) or _FIT_NAME_MAP.get(f.name)
                if col and col not in row:
                    row[col] = f.value
            records.append(row)
        elif msg.name == "session":
            session_data = {f.name: f.value for f in msg.fields}

    if not records:
        return None

    df = pd.DataFrame(records)

    # 단위 변환: speed m/s → kph,  distance m → km
    if "speed" in df.columns:
        df["kph"] = pd.to_numeric(df["speed"], errors="coerce") * 3.6
    if "distance" in df.columns:
        df["km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0

    # 타임스탬프 → secs(경과초) + 날짜
    date_str     = str(date.today())
    moving_time  = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        valid_ts = ts.dropna()
        if len(valid_ts) > 0:
            date_str    = valid_ts.iloc[0].to_pydatetime().astimezone().strftime("%Y-%m-%d")
            secs        = (ts - valid_ts.iloc[0]).dt.total_seconds()
            df["secs"]  = secs
            secs_diff   = secs.diff().fillna(1.0)
            moving_time = float(secs_diff[secs_diff < 30].sum())

    # 파워 스파이크 보정 — 전후 5초 중앙값의 3배 & 400W 초과 → 중앙값으로 대체
    spike_count = 0
    if "watts" in df.columns:
        df["watts"]  = pd.to_numeric(df["watts"], errors="coerce")
        roll_med     = df["watts"].rolling(window=11, center=True, min_periods=1).median()
        spike_mask   = (df["watts"] > roll_med * 3) & (df["watts"] > 400)
        spike_count  = int(spike_mask.sum())
        df.loc[spike_mask, "watts"] = roll_med[spike_mask]

    # 실내 온도
    indoor_temp = None
    if "temp" in df.columns:
        t_valid = pd.to_numeric(df["temp"], errors="coerce").dropna()
        if len(t_valid) > 0:
            indoor_temp = round(float(t_valid.mean()), 1)

    # sport / indoor
    sport = str(session_data.get("sport", "unknown")).lower()
    indoor_flag = int(
        str(session_data.get("sub_sport", "")).lower()
        in ["indoor_cycling", "treadmill", "virtual_activity"]
    )

    # 거리
    distance_km = None
    if "km" in df.columns:
        km_valid = df["km"].dropna()
        distance_km = float(km_valid.max()) if len(km_valid) > 0 else None
    if not distance_km:
        raw_dist = session_data.get("total_distance")
        if raw_dist:
            distance_km = raw_dist / 1000.0 if raw_dist > 1000 else raw_dist

    # duration / 이동·정지 시간
    total_elapsed = session_data.get("total_elapsed_time")
    duration_sec  = moving_time or total_elapsed
    moving_time_sec = int(moving_time) if moving_time else None
    stop_time_sec   = int(total_elapsed - moving_time) if (total_elapsed and moving_time and total_elapsed > moving_time) else None

    # 코스명
    course_name = str(session_data.get("sport_profile_name", "") or "").strip() or None

    # 심박
    hr_valid   = df["hr"].dropna()  if "hr"    in df.columns else pd.Series(dtype=float)
    avg_hr     = float(hr_valid.mean()) if len(hr_valid) > 0 else None
    max_hr_val = float(hr_valid.max())  if len(hr_valid) > 0 else None

    # 파워
    w_valid   = df["watts"].dropna() if "watts" in df.columns else pd.Series(dtype=float)
    avg_power = float(w_valid.mean()) if len(w_valid) > 0 else None
    max_power = float(w_valid.max())  if len(w_valid) > 0 else None

    # 케이던스 (0 제외)
    avg_cad = None
    if "cad" in df.columns:
        cad_valid = df[df["cad"] > 0]["cad"].dropna()
        avg_cad   = float(cad_valid.mean()) if len(cad_valid) > 0 else None

    # W/bpm 효율 — 파워 ≥10W, 심박 ≥80bpm 구간만
    w_per_bpm = None
    if "watts" in df.columns and "hr" in df.columns:
        eff_mask = (df["watts"] >= 10) & (df["hr"] >= 80)
        if eff_mask.sum() > 0:
            ep = df.loc[eff_mask, "watts"].mean()
            eh = df.loc[eff_mask, "hr"].mean()
            w_per_bpm = round(ep / eh, 3) if eh > 0 else None

    # 심박 드리프트
    drift = None
    if len(hr_valid) > 10:
        mid = len(df) // 2
        h1  = df["hr"].iloc[:mid].mean()
        h2  = df["hr"].iloc[mid:].mean()
        drift = round((h2 - h1) / h1 * 100, 1) if h1 else None

    # 심박 존 분포
    zones  = hr_zones(max_hr)
    z_pcts = [0.0] * 5
    total  = len(hr_valid)
    if total > 0:
        for i, (lo, hi) in enumerate(zones):
            z_pcts[i] = round(((hr_valid >= lo) & (hr_valid <= hi)).sum() / total * 100, 1)

    # 평균 페이스
    avg_pace = None
    if distance_km and duration_sec and distance_km > 0:
        avg_pace = round(duration_sec / distance_km, 1)

    # 평균 속도
    avg_speed_kph = round(distance_km / (moving_time / 3600), 1) if (distance_km and moving_time and moving_time > 0) else None

    calories = session_data.get("total_calories")

    # 케이던스 구간 분포
    cad_z = _cad_zones(df["cad"]) if "cad" in df.columns else {}

    # 러닝 전용 — 최고 페이스 및 페이스 드리프트
    best_pace_val  = None
    pace_drift_val = None
    is_running_fit = "run" in sport
    if is_running_fit and "kph" in df.columns:
        kph_s   = pd.to_numeric(df["kph"], errors="coerce").dropna()
        kph_pos = kph_s[kph_s > 0]
        if len(kph_pos) > 20:
            best_kph = float(kph_pos.quantile(0.95))
            best_pace_val = round(3600 / best_kph, 1) if best_kph > 0 else None
            pace_s = (3600 / kph_pos).replace([float("inf")], None).dropna()
            if len(pace_s) > 20:
                mid = len(pace_s) // 2
                p1, p2 = pace_s.iloc[:mid].mean(), pace_s.iloc[mid:].mean()
                if not (pd.isna(p1) or pd.isna(p2)):
                    pace_drift_val = round(p2 - p1, 1)

    # 원시 데이터 저장 (차트용)
    fname     = os.path.basename(path)
    save_cols = [c for c in ["secs", "hr", "watts", "cad", "kph"] if c in df.columns]
    if save_cols:
        save_raw(fname, df[save_cols])

    return dict(
        date=date_str,
        filename=fname,
        sport=sport,
        indoor=indoor_flag,
        course_name=course_name,
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(duration_sec)    if duration_sec else None,
        moving_time_sec=moving_time_sec,
        stop_time_sec=stop_time_sec,
        avg_speed_kph=avg_speed_kph,
        avg_hr=round(avg_hr, 1)           if avg_hr       else None,
        max_hr=round(max_hr_val, 1)       if max_hr_val   else None,
        drift=drift,
        drift_grade=_drift_grade(drift),
        avg_power=round(avg_power, 1)     if avg_power    else None,
        max_power=round(max_power, 1)     if max_power    else None,
        avg_cadence=round(avg_cad, 1)     if avg_cad      else None,
        w_per_bpm=w_per_bpm,
        spike_count=spike_count,
        spike_corrected=1 if spike_count > 0 else 0,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=avg_pace,
        best_pace=best_pace_val,
        pace_drift_sec=pace_drift_val,
        calories=float(calories) if calories else None,
        indoor_temp=indoor_temp,
        **cad_z,
    )


# ── GPX 파싱 (러닝 전용) ────────────────────────────────────────────────────────
def parse_gpx(path: str, max_hr: int, ftp: int):
    try:
        import gpxpy as gpx_lib
    except ImportError:
        st.error("gpxpy 패키지가 설치되지 않았습니다.")
        return None

    try:
        with open(path, encoding="utf-8") as f:
            gpx = gpx_lib.parse(f)
    except Exception as e:
        st.error(f"GPX 파일 읽기 실패: {e}")
        return None

    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                row = {"time": pt.time, "lat": pt.latitude, "lon": pt.longitude, "ele": pt.elevation}
                if pt.extensions:
                    for ext in pt.extensions:
                        for child in list(ext):
                            tag = child.tag.split("}")[-1].lower() if "}" in child.tag else child.tag.lower()
                            if tag in ("hr", "heartrate", "heart_rate"):
                                try:
                                    row["hr"] = float(child.text)
                                except Exception:
                                    pass
                points.append(row)

    if not points:
        return None

    df = pd.DataFrame(points)

    # 타임스탬프 → secs + 날짜
    date_str    = str(date.today())
    moving_time = None
    if "time" in df.columns and df["time"].notna().any():
        ts       = pd.to_datetime(df["time"], utc=True, errors="coerce")
        valid_ts = ts.dropna()
        if len(valid_ts) > 0:
            date_str   = valid_ts.iloc[0].to_pydatetime().astimezone().strftime("%Y-%m-%d")
            secs       = (ts - valid_ts.iloc[0]).dt.total_seconds()
            df["secs"] = secs
            secs_diff  = secs.diff().fillna(1.0)
            moving_time = float(secs_diff[secs_diff < 30].sum())

    # Haversine 누적 거리
    dist_m = [0.0]
    for i in range(1, len(df)):
        r0, r1 = df.iloc[i - 1], df.iloc[i]
        if all(pd.notna([r0["lat"], r0["lon"], r1["lat"], r1["lon"]])):
            dist_m.append(dist_m[-1] + _haversine_m(r0["lat"], r0["lon"], r1["lat"], r1["lon"]))
        else:
            dist_m.append(dist_m[-1])
    df["dist_m"] = dist_m
    df["km"]     = df["dist_m"] / 1000.0
    distance_km  = df["dist_m"].iloc[-1] / 1000.0

    # 구간 페이스 (sec/km) — 이상치 제거 후 롤링 스무딩
    if "secs" in df.columns:
        dt = df["secs"].diff().fillna(0)
        dd = df["dist_m"].diff().fillna(0)
        raw_pace = (dt / dd * 1000).where((dd > 0.3) & (dt < 30))  # sec/km
        df["pace"] = raw_pace.rolling(10, center=True, min_periods=1).median()

    # 평균/최고 페이스
    pace_valid = df["pace"].dropna() if "pace" in df.columns else pd.Series(dtype=float)
    avg_pace   = float(pace_valid.mean())         if len(pace_valid) > 0  else None
    best_pace  = float(pace_valid.quantile(0.05)) if len(pace_valid) > 20 else None

    # 페이스 드리프트 (전반 vs 후반)
    pace_drift_sec = None
    if len(pace_valid) > 20:
        mid = len(pace_valid) // 2
        p1, p2 = pace_valid.iloc[:mid].mean(), pace_valid.iloc[mid:].mean()
        if not (pd.isna(p1) or pd.isna(p2)):
            pace_drift_sec = round(float(p2 - p1), 1)

    # 평균 속도
    avg_speed_kph = round(distance_km / (moving_time / 3600), 1) if (distance_km and moving_time and moving_time > 0) else None

    # 심박
    hr_valid   = df["hr"].dropna() if "hr" in df.columns else pd.Series(dtype=float)
    avg_hr     = float(hr_valid.mean()) if len(hr_valid) > 0 else None
    max_hr_val = float(hr_valid.max())  if len(hr_valid) > 0 else None

    # 심박 드리프트
    drift = None
    if len(hr_valid) > 10:
        mid = len(df) // 2
        h1  = df["hr"].iloc[:mid].mean()
        h2  = df["hr"].iloc[mid:].mean()
        drift = round((h2 - h1) / h1 * 100, 1) if h1 else None

    # 심박 존 분포
    zones  = hr_zones(max_hr)
    z_pcts = [0.0] * 5
    total  = len(hr_valid)
    if total > 0:
        for i, (lo, hi) in enumerate(zones):
            z_pcts[i] = round(((hr_valid >= lo) & (hr_valid <= hi)).sum() / total * 100, 1)

    # km 랩 분석
    fname = os.path.basename(path)
    laps  = []
    if "secs" in df.columns and distance_km >= 1:
        df_r     = df.reset_index(drop=True)
        prev_pos = 0
        for km_n in range(1, int(distance_km) + 1):
            cross = df_r[df_r["km"] >= km_n]
            if len(cross) == 0:
                break
            curr_pos  = cross.index[0]
            lap_time  = df_r.loc[curr_pos, "secs"] - df_r.loc[prev_pos, "secs"]
            lap_slice = df_r.iloc[prev_pos: curr_pos + 1]
            hr_mean   = lap_slice["hr"].dropna().mean() if "hr" in lap_slice.columns else float("nan")
            laps.append({
                "km":      km_n,
                "시간":    fmt_duration(lap_time),
                "페이스":  fmt_pace(lap_time),
                "평균 심박": int(round(hr_mean)) if not pd.isna(hr_mean) else "-",
            })
            prev_pos = curr_pos

    # 원시 데이터 + 랩 저장
    save_cols = [c for c in ["secs", "hr", "pace", "km"] if c in df.columns]
    save_raw(fname, df[save_cols])
    if laps:
        save_laps(fname, laps)

    return dict(
        date=date_str,
        filename=fname,
        sport="running",
        indoor=0,
        course_name=None,
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(moving_time)     if moving_time  else None,
        moving_time_sec=int(moving_time)  if moving_time  else None,
        stop_time_sec=None,
        avg_speed_kph=avg_speed_kph,
        avg_hr=round(avg_hr, 1)           if avg_hr       else None,
        max_hr=round(max_hr_val, 1)       if max_hr_val   else None,
        drift=drift,
        drift_grade=_drift_grade(drift),
        avg_power=None, max_power=None,
        avg_cadence=None, w_per_bpm=None,
        spike_count=0, spike_corrected=0,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=round(avg_pace, 1)       if avg_pace     else None,
        best_pace=round(best_pace, 1)     if best_pace    else None,
        pace_drift_sec=pace_drift_sec,
        calories=None,
        indoor_temp=None,
        cad_lt70=0.0, cad_70_80=0.0, cad_80_90=0.0, cad_90_100=0.0, cad_100p=0.0,
    )


# ── 원시 데이터 저장/로드 ──────────────────────────────────────────────────────
def save_raw(filename: str, df: pd.DataFrame):
    raw_json = df.to_json(orient="records")
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO training_raw (filename, raw_json) VALUES (?, ?)",
            (filename, raw_json),
        )


def load_raw(filename: str) -> pd.DataFrame:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT raw_json FROM training_raw WHERE filename = ?", (filename,)
        ).fetchone()
    if row and row[0]:
        return pd.DataFrame(json.loads(row[0]))
    return pd.DataFrame()


def save_laps(filename: str, laps: list):
    with get_conn() as conn:
        conn.execute(
            "UPDATE training_raw SET laps_json = ? WHERE filename = ?",
            (json.dumps(laps), filename),
        )


def load_laps(filename: str) -> list:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT laps_json FROM training_raw WHERE filename = ?", (filename,)
        ).fetchone()
    if row and row[0]:
        return json.loads(row[0])
    return []


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── CSV 파싱 (GoldenCheetah 형식) ──────────────────────────────────────────────
def parse_csv(path: str, max_hr: int, ftp: int):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"CSV 파일 읽기 실패: {e}")
        return None

    required = {"secs", "hr", "watts"}
    if not required.issubset(set(df.columns)):
        st.error(f"GoldenCheetah CSV 형식이 아닙니다. 필요 컬럼: {required}")
        return None

    df = df.copy()

    # 파워 스파이크 보정 — 전후 5초 중앙값의 3배 초과 & 400W 초과 시 중앙값으로 대체
    rolling_med = df["watts"].rolling(window=11, center=True, min_periods=1).median()
    spike_mask  = (df["watts"] > rolling_med * 3) & (df["watts"] > 400)
    spike_count = int(spike_mask.sum())
    df.loc[spike_mask, "watts"] = rolling_med[spike_mask]

    # 이동 시간 — 30초 이상 gap은 정지로 판단
    secs_diff   = df["secs"].diff().fillna(1.0)
    total_secs  = float(df["secs"].max() - df["secs"].min()) if len(df) > 1 else 0.0
    moving_mask = secs_diff < 30
    moving_time = float(secs_diff[moving_mask].sum())
    stop_time_sec = int(total_secs - moving_time) if total_secs > moving_time else None

    # 거리
    distance_km = None
    if "km" in df.columns:
        km_valid = df["km"].dropna()
        if len(km_valid) > 0:
            distance_km = float(km_valid.max() - km_valid.min())
            if distance_km <= 0:
                distance_km = float(km_valid.max())

    # 심박
    hr_valid   = df["hr"].dropna()
    avg_hr     = float(hr_valid.mean()) if len(hr_valid) > 0 else None
    max_hr_val = float(hr_valid.max())  if len(hr_valid) > 0 else None

    # 파워
    watts_valid = df["watts"].dropna()
    avg_power   = float(watts_valid.mean()) if len(watts_valid) > 0 else None
    max_power   = float(watts_valid.max())  if len(watts_valid) > 0 else None

    # 케이던스 (0 제외)
    avg_cad = None
    if "cad" in df.columns:
        cad_valid = df[df["cad"] > 0]["cad"].dropna()
        avg_cad = float(cad_valid.mean()) if len(cad_valid) > 0 else None

    # W/bpm 효율 — 파워 ≥10W, 심박 ≥80bpm 구간만
    w_per_bpm = None
    eff_mask = (df["watts"] >= 10) & (df["hr"] >= 80)
    if eff_mask.sum() > 0:
        eff_pwr = df.loc[eff_mask, "watts"].mean()
        eff_hr  = df.loc[eff_mask, "hr"].mean()
        if eff_hr > 0:
            w_per_bpm = round(eff_pwr / eff_hr, 3)

    # 심박 드리프트 — 전반 평균 vs 후반 평균
    drift = None
    if len(hr_valid) > 10:
        mid = len(df) // 2
        h1  = df["hr"].iloc[:mid].mean()
        h2  = df["hr"].iloc[mid:].mean()
        drift = round((h2 - h1) / h1 * 100, 1) if h1 else None

    # 심박 존 분포
    zones  = hr_zones(max_hr)
    z_pcts = [0.0] * 5
    total  = len(hr_valid)
    if total > 0:
        for i, (lo, hi) in enumerate(zones):
            z_pcts[i] = round(((hr_valid >= lo) & (hr_valid <= hi)).sum() / total * 100, 1)

    # 평균 페이스
    avg_pace = None
    if distance_km and moving_time and distance_km > 0:
        avg_pace = round(moving_time / distance_km, 1)

    # 평균 속도
    avg_speed_kph = round(distance_km / (moving_time / 3600), 1) if (distance_km and moving_time and moving_time > 0) else None

    # 케이던스 구간 분포
    cad_z = _cad_zones(df["cad"]) if "cad" in df.columns else {}

    # 날짜 — 파일 수정 시각 기준
    try:
        date_str = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except Exception:
        date_str = str(date.today())

    # 원시 데이터 저장 (차트용)
    fname     = os.path.basename(path)
    save_cols = [c for c in ["secs", "hr", "watts", "cad", "kph"] if c in df.columns]
    save_raw(fname, df[save_cols])

    return dict(
        date=date_str,
        filename=fname,
        sport="cycling",
        indoor=1,
        course_name=None,
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(moving_time)      if moving_time  else None,
        moving_time_sec=int(moving_time)   if moving_time  else None,
        stop_time_sec=stop_time_sec,
        avg_speed_kph=avg_speed_kph,
        avg_hr=round(avg_hr, 1)            if avg_hr       else None,
        max_hr=round(max_hr_val, 1)        if max_hr_val   else None,
        drift=drift,
        drift_grade=_drift_grade(drift),
        avg_power=round(avg_power, 1)      if avg_power    else None,
        max_power=round(max_power, 1)      if max_power    else None,
        avg_cadence=round(avg_cad, 1)      if avg_cad      else None,
        w_per_bpm=w_per_bpm,
        spike_count=spike_count,
        spike_corrected=1 if spike_count > 0 else 0,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=avg_pace,
        best_pace=None,
        pace_drift_sec=None,
        calories=None,
        indoor_temp=None,
        **cad_z,
    )


def parse_file(path, max_hr, ftp):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fit":
        return parse_fit(path, max_hr, ftp)
    elif ext == ".gpx":
        return parse_gpx(path, max_hr, ftp)
    elif ext == ".csv":
        return parse_csv(path, max_hr, ftp)
    return None


def save_record(data: dict):
    cols = [c for c in data.keys()]
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO training_log ({col_str}) VALUES ({placeholders})"
    with get_conn() as conn:
        conn.execute(sql, list(data.values()))


def load_all() -> pd.DataFrame:
    with get_conn() as conn:
        try:
            df = pd.read_sql("SELECT * FROM training_log ORDER BY date DESC", conn)
        except Exception:
            df = pd.DataFrame()
    return df


# ── 사이드바 ────────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.title("⚙️ 설정")
    max_hr = st.sidebar.number_input("최대 심박수 (bpm)", min_value=100, max_value=220, value=185, step=1)
    ftp = st.sidebar.number_input("FTP (watts)", min_value=50, max_value=500, value=170, step=5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("파일 업로드")
    uploaded = st.sidebar.file_uploader(
        "FIT / GPX / CSV 파일 선택",
        type=["fit", "gpx", "csv"],
        accept_multiple_files=True,
    )
    if uploaded:
        for uf in uploaded:
            tmp = os.path.join(tempfile.gettempdir(), uf.name)
            with open(tmp, "wb") as f:
                f.write(uf.read())
            data = parse_file(tmp, max_hr, ftp)
            if data:
                save_record(data)
                st.sidebar.success(f"✅ {uf.name} 저장됨")
            else:
                st.sidebar.warning(f"⚠️ {uf.name} 파싱 실패")

    return max_hr, ftp


# ── 코칭 피드백 ─────────────────────────────────────────────────────────────────
def coaching_feedback(row: dict) -> list:
    msgs       = []
    sport      = str(row.get("sport", "")).lower()
    is_running = "running" in sport or "run" in sport
    drift      = row.get("drift")
    z1  = row.get("z1_pct") or 0
    z2  = row.get("z2_pct") or 0
    z3  = row.get("z3_pct") or 0
    z4  = row.get("z4_pct") or 0
    z5  = row.get("z5_pct") or 0
    w_bpm      = row.get("w_per_bpm")
    prev_wbpm  = row.get("_prev_wbpm")   # injected by generate_training_pdf
    avg_cad    = row.get("avg_cadence")
    pace_drift = row.get("pace_drift_sec")

    # ── HR 드리프트 ───────────────────────────────────────────────────────────
    if drift is not None and not pd.isna(drift):
        d = abs(drift)
        if d <= 7:
            msgs.append(("success", f"심박 드리프트 {drift:+.1f}% — 심박이 안정적으로 유지됐습니다. 회복 상태 양호합니다."))
        elif d <= 10:
            msgs.append(("info",    f"심박 드리프트 {drift:+.1f}% — 약간 피로 누적이 보입니다. 수분 섭취와 수면을 체크하세요."))
        elif d <= 15:
            msgs.append(("warning", f"심박 드리프트 {drift:+.1f}% — 피로 상태입니다. 다음 훈련 강도를 5W(또는 5초/km) 낮추고 충분히 회복하세요."))
        else:
            msgs.append(("error",   f"심박 드리프트 {drift:+.1f}% — 과부하 상태입니다. 강도를 크게 줄이고 1~2일 회복 훈련을 권장합니다."))

    # ── Z2 비율 ───────────────────────────────────────────────────────────────
    if is_running:
        if z2 >= 80:
            msgs.append(("success", f"Z2 비율 {z2:.0f}% — Z2 러닝 목표를 달성했습니다. 유산소 기반이 탄탄합니다."))
        elif z2 >= 60:
            msgs.append(("info",    f"Z2 비율 {z2:.0f}% — 부분 성공입니다. 다음 세션은 페이스를 5~8초/km 낮춰 Z2 타겟을 재조정해보세요."))
        else:
            msgs.append(("warning", f"Z2 비율 {z2:.0f}% — 강도가 높습니다. 페이스를 5~8초/km 낮추고 Z2 구간 유지에 집중하세요."))
        # 페이스 드리프트
        if pace_drift is not None:
            if abs(pace_drift) <= 10:
                msgs.append(("success", f"페이스 드리프트 {pace_drift:+.0f}초/km — 매우 안정적인 페이스를 유지했습니다."))
            elif abs(pace_drift) <= 20:
                msgs.append(("info",    f"페이스 드리프트 {pace_drift:+.0f}초/km — 후반에 약간 느려졌습니다. 페이스 배분을 조절해보세요."))
            else:
                msgs.append(("warning", f"페이스 드리프트 {pace_drift:+.0f}초/km — 초반 페이스가 너무 빨랐습니다. 출발 페이스를 낮춰보세요."))
        if z4 + z5 > 30:
            msgs.append(("warning", f"고강도 구간 {z4+z5:.0f}% — 충분한 회복 후 다음 훈련에 임하세요."))
    else:
        # 사이클링 Z2
        if z2 >= 80:
            msgs.append(("success", f"Z2 비율 {z2:.0f}% — Z2 목표를 달성했습니다. 유산소 기반 훈련 효과가 우수합니다."))
        elif z2 >= 60:
            msgs.append(("info",    f"Z2 비율 {z2:.0f}% — 부분 성공입니다. 다음 세션은 출력을 5W 낮춰 Z2 구간에 집중하세요."))
        else:
            msgs.append(("warning", f"Z2 비율 {z2:.0f}% — 강도가 높았습니다. 출력을 5~8W 낮추고 Z2 유지에 집중하세요."))
        if z4 + z5 > 40:
            msgs.append(("warning", f"고강도 구간 {z4+z5:.0f}% — 인터벌 효과가 높았습니다. 다음 세션 전 충분히 회복하세요."))
        elif z3 > 30:
            msgs.append(("info",    f"템포 구간 {z3:.0f}% — 젖산 역치 개선에 효과적인 훈련이었습니다."))

    # ── W/bpm 효율 (사이클링) ──────────────────────────────────────────────────
    if not is_running and w_bpm and not pd.isna(w_bpm):
        if prev_wbpm and not pd.isna(prev_wbpm):
            diff = w_bpm - prev_wbpm
            if diff > 0:
                msgs.append(("success", f"W/bpm {w_bpm:.3f} (전 세션 대비 +{diff:.3f}) — 심폐 효율이 향상되고 있습니다."))
            elif drift is not None and not pd.isna(drift) and abs(drift) > 7:
                msgs.append(("info",    f"W/bpm {w_bpm:.3f} (전 세션 대비 {diff:.3f}) — 피로로 인한 일시적 하락으로 보입니다. 회복 후 재확인하세요."))
            else:
                msgs.append(("warning", f"W/bpm {w_bpm:.3f} (전 세션 대비 {diff:.3f}) — 효율 저하가 감지됩니다. 훈련 패턴과 수면을 점검하세요."))
        else:
            if w_bpm > 2.0:
                msgs.append(("success", f"W/bpm 효율 {w_bpm:.3f} — 심박 대비 출력이 우수합니다."))
            elif w_bpm < 1.2:
                msgs.append(("info",    f"W/bpm 효율 {w_bpm:.3f} — 효율 향상을 위해 저강도 지구력 훈련을 늘려보세요."))
            else:
                msgs.append(("info",    f"W/bpm 효율 {w_bpm:.3f} — 양호한 범위입니다. 꾸준한 훈련으로 향상시켜보세요."))

    # ── 케이던스 ───────────────────────────────────────────────────────────────
    if avg_cad and not pd.isna(avg_cad) and avg_cad > 0:
        unit = "spm" if is_running else "rpm"
        if avg_cad >= 88:
            msgs.append(("success", f"평균 케이던스 {avg_cad:.0f} {unit} — 이상적인 케이던스 범위입니다."))
        elif avg_cad >= 83:
            msgs.append(("info",    f"평균 케이던스 {avg_cad:.0f} {unit} — 정상 범위입니다. 88 {unit} 이상을 목표로 해보세요."))
        else:
            msgs.append(("warning", f"평균 케이던스 {avg_cad:.0f} {unit} — 낮은 편입니다. 케이던스를 조금씩 높여보세요."))

    if not msgs:
        msgs.append(("info", "훈련 데이터가 충분히 쌓이면 더 자세한 피드백을 제공합니다."))
    return msgs


# ── 탭 1: 최근 훈련 ──────────────────────────────────────────────────────────
def tab_recent(df: pd.DataFrame, max_hr: int, ftp: int):
    st.header("최근 훈련")

    if df.empty:
        st.info("훈련 데이터가 없습니다. 사이드바에서 파일을 업로드하세요.")
        return

    df_s        = df.sort_values("date", ascending=False)
    latest_date = str(df_s.iloc[0]["date"])
    latest_df   = df_s[df_s["date"] == latest_date]

    st.caption(f"가장 최근 훈련일: **{latest_date}**")
    st.markdown("---")

    for _, row in latest_df.iterrows():
        sport  = str(row.get("sport", "")).lower()
        icon   = "🏃" if "run" in sport else "🚴"
        fname  = os.path.basename(str(row.get("filename", "")))
        with st.expander(f"{icon} {fname} — {row['sport']}", expanded=True):
            show_training_detail(row, df, max_hr, ftp)


# ── 탭 2: 캘린더 ───────────────────────────────────────────────────────────────
def tab_calendar(df: pd.DataFrame):
    st.header("훈련 캘린더")

    if df.empty:
        st.info("데이터가 없습니다.")
        return

    today = date.today()

    # ── 월 네비게이션 ─────────────────────────────────────────────────────────
    if "cal_year"  not in st.session_state: st.session_state.cal_year  = today.year
    if "cal_month" not in st.session_state: st.session_state.cal_month = today.month

    year  = st.session_state.cal_year
    month = st.session_state.cal_month

    nc1, nc2, nc3 = st.columns([1, 4, 1])
    with nc1:
        if st.button("◀ 이전달", key="cal_prev"):
            if month == 1: st.session_state.cal_year -= 1; st.session_state.cal_month = 12
            else:           st.session_state.cal_month -= 1
            st.rerun()
    nc2.markdown(
        f"<h3 style='text-align:center;margin:4px 0'>{year}년 {month}월</h3>",
        unsafe_allow_html=True,
    )
    with nc3:
        if st.button("다음달 ▶", key="cal_next"):
            if month == 12: st.session_state.cal_year += 1; st.session_state.cal_month = 1
            else:            st.session_state.cal_month += 1
            st.rerun()

    # ── 데이터 준비 ───────────────────────────────────────────────────────────
    df_c = df.copy()
    df_c["date"] = pd.to_datetime(df_c["date"]).dt.date
    mdf = df_c[
        (df_c["date"].apply(lambda d: d.year)  == year) &
        (df_c["date"].apply(lambda d: d.month) == month)
    ]

    date_recs: dict = {}
    for _, row in mdf.iterrows():
        date_recs.setdefault(row["date"], []).append(row.to_dict())

    def zone_bg(recs):
        if not recs: return ""
        n  = len(recs)
        z2 = sum((r.get("z2_pct") or 0) for r in recs) / n
        z3 = sum((r.get("z3_pct") or 0) for r in recs) / n
        z4 = sum((r.get("z4_pct") or 0) for r in recs) / n
        z5 = sum((r.get("z5_pct") or 0) for r in recs) / n
        if z4 + z5 >= 50: return "rgba(240,153,123,0.35)"
        if z3 + z4 > z2:  return "rgba(250,199,117,0.35)"
        if z2 >= 70:       return "rgba(159,225,203,0.35)"
        return "rgba(181,212,244,0.25)"

    def _v(val):
        return val if (val is not None and not (isinstance(val, float) and math.isnan(val))) else None

    # ── 메인 레이아웃 (달력 + 사이드 패널) ───────────────────────────────────
    main_col, side_col = st.columns([3, 1])

    with main_col:
        cal     = calendar.monthcalendar(year, month)
        widths  = [1, 1, 1, 1, 1, 1, 1, 1.5]
        headers = ["월", "화", "수", "목", "금", "토", "일", "주 합계"]

        hcols = st.columns(widths)
        for i, h in enumerate(headers):
            hcols[i].markdown(
                f"<div style='text-align:center;font-weight:700;padding:4px;"
                f"border-bottom:2px solid #ddd'>{h}</div>",
                unsafe_allow_html=True,
            )

        for wi, week in enumerate(cal):
            cols      = st.columns(widths)
            week_recs = []

            for i, day in enumerate(week):
                if day == 0:
                    cols[i].markdown("<div style='min-height:90px'></div>", unsafe_allow_html=True)
                    continue

                d    = date(year, month, day)
                recs = date_recs.get(d, [])

                if recs:
                    week_recs.extend(recs)
                    bg     = zone_bg(recs)
                    border = "2px solid #4A90D9" if d == today else "1px solid #ccc"
                    lines  = [f"<b>{day}</b>"]
                    for r in recs:
                        sport  = str(r.get("sport", "")).lower()
                        is_run = "run" in sport
                        icon   = "🏃" if is_run else "🚴"
                        dist   = f"{r['distance_km']:.1f}km" if _v(r.get("distance_km")) else ""
                        hr_s   = f"{r['avg_hr']:.0f}bpm"    if _v(r.get("avg_hr"))       else ""
                        if is_run:
                            pace_s = fmt_pace(r.get("avg_pace"))
                            lines.append(f"{icon} {dist}<br><small>{pace_s} {hr_s}</small>")
                        else:
                            pwr_s = f"{r['avg_power']:.0f}W" if _v(r.get("avg_power")) else ""
                            lines.append(f"{icon} {dist}<br><small>{hr_s} {pwr_s}</small>")

                    cols[i].markdown(
                        f"<div style='background:{bg};border:{border};border-radius:6px;"
                        f"padding:5px;min-height:90px;font-size:0.75em;line-height:1.5'>"
                        + "".join(lines) + "</div>",
                        unsafe_allow_html=True,
                    )
                    if cols[i].button("📋", key=f"cal_{d}", help="훈련 기록으로 이동"):
                        st.session_state.nav_tab = "훈련 기록"
                        st.session_state.rec_filter_date = str(d)
                        st.session_state.rec_detail_file = None
                        st.rerun()
                else:
                    style = (
                        "font-weight:700;color:#4A90D9;border:2px solid #4A90D9;"
                        if d == today else "color:#bbb;"
                    )
                    cols[i].markdown(
                        f"<div style='text-align:center;padding:5px;min-height:90px;{style}'>{day}</div>",
                        unsafe_allow_html=True,
                    )

            if week_recs:
                wd   = sum((r.get("distance_km") or 0) for r in week_recs)
                wt   = sum((r.get("duration_sec") or 0) for r in week_recs)
                rc   = sum(1 for r in week_recs if "run" in str(r.get("sport","")).lower())
                bc   = len(week_recs) - rc
                parts = []
                if bc: parts.append(f"🚴×{bc}")
                if rc: parts.append(f"🏃×{rc}")
                parts += [f"{wd:.1f}km", fmt_duration(wt)]
                cols[7].markdown(
                    "<div style='background:#f0f4f8;border-radius:6px;padding:6px;"
                    "font-size:0.73em;min-height:60px;line-height:1.9'>"
                    + "<br>".join(parts) + "</div>",
                    unsafe_allow_html=True,
                )
                # 주간 리포트 이동 버튼
                first_day = next((date(year, month, d) for d in week if d != 0), None)
                if first_day:
                    iso_y, iso_w, _ = first_day.isocalendar()
                    wlabel = f"{iso_y}-W{iso_w:02d}"
                    if cols[7].button("📊", key=f"cal_wrep_{wi}", help=f"{wlabel} 주간 리포트"):
                        st.session_state.nav_tab = "리포트"
                        st.session_state.report_preset_type  = "주간"
                        st.session_state.report_preset_label = wlabel
                        st.rerun()

    # ── 사이드 패널 ───────────────────────────────────────────────────────────
    with side_col:
        days_in_month = calendar.monthrange(year, month)[1]
        trained_days  = len(date_recs)
        passed_days   = today.day if (today.year == year and today.month == month) else days_in_month

        st.subheader("📆 이번 달")
        st.metric("훈련일", f"{trained_days}일")
        st.progress(min(trained_days / max(passed_days, 1), 1.0),
                    text=f"{passed_days}일 경과 중")

        st.markdown("---")
        st.subheader("🏆 베스트")
        if not mdf.empty:
            wbpm_s  = mdf["w_per_bpm"].dropna()
            dist_s  = mdf["distance_km"].dropna()
            drift_s = mdf["drift"].dropna()
            if len(wbpm_s):  st.metric("최고 W/bpm",    f"{wbpm_s.max():.3f}")
            if len(dist_s):  st.metric("최장 거리",      f"{dist_s.max():.1f} km")
            if len(drift_s):
                best_i = drift_s.abs().idxmin()
                st.metric("최저 드리프트", f"{drift_s[best_i]:+.1f}%")

        st.markdown("---")
        st.subheader("📊 주간 볼륨")
        if not mdf.empty:
            wvol = []
            for wi, week in enumerate(calendar.monthcalendar(year, month)):
                ds    = {d for d in week if d != 0}
                wdist = sum((r["distance_km"] or 0) for _, r in mdf.iterrows() if r["date"].day in ds)
                wvol.append({"주": f"W{wi+1}", "km": round(wdist, 1)})
            fig_v = px.bar(pd.DataFrame(wvol), x="주", y="km",
                           color_discrete_sequence=["#4A90D9"])
            fig_v.update_layout(height=160, margin=dict(t=5, b=20, l=20, r=5),
                                xaxis_title=None, yaxis_title="km")
            st.plotly_chart(fig_v, use_container_width=True)

    # ── 월간 합계 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    if not mdf.empty:
        st.subheader(f"📋 {year}년 {month}월 월간 요약")
        for sport in mdf["sport"].dropna().unique():
            sdf  = mdf[mdf["sport"] == sport]
            icon = "🏃" if "run" in str(sport).lower() else "🚴"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{icon} {sport}", f"{len(sdf)}회")
            c2.metric("총 거리",          f"{sdf['distance_km'].sum():.1f} km")
            c3.metric("총 시간",          fmt_duration(sdf["duration_sec"].sum()))
            wbpm_m = sdf["w_per_bpm"].dropna()
            pace_m = sdf["avg_pace"].dropna()
            if len(wbpm_m):   c4.metric("평균 W/bpm",   f"{wbpm_m.mean():.3f}")
            elif len(pace_m): c4.metric("평균 페이스",   fmt_pace(pace_m.mean()))

        mc1, mc2 = st.columns([4, 1])
        pm, py = (month - 1, year) if month > 1 else (12, year - 1)
        prev_df = df_c[
            (df_c["date"].apply(lambda d: d.year)  == py) &
            (df_c["date"].apply(lambda d: d.month) == pm)
        ]
        cw = mdf["w_per_bpm"].dropna()
        pw = prev_df["w_per_bpm"].dropna() if not prev_df.empty else pd.Series(dtype=float)
        if len(cw) and len(pw):
            chg = (cw.mean() - pw.mean()) / pw.mean() * 100
            mc1.metric("W/bpm 전월 대비", f"{cw.mean():.3f}", delta=f"{chg:+.1f}%")
        mlabel = f"{year}-{month:02d}"
        if mc2.button("📊 월간 리포트", key="cal_mrep"):
            st.session_state.nav_tab = "리포트"
            st.session_state.report_preset_type  = "월간"
            st.session_state.report_preset_label = mlabel
            st.rerun()


# ── 탭 3: 훈련 기록 ────────────────────────────────────────────────────────────
def tab_records(df: pd.DataFrame, max_hr: int, ftp: int):
    # ── 상세 뷰 모드 ──────────────────────────────────────────────────────────
    detail_file = st.session_state.get("rec_detail_file")
    if detail_file:
        bc1, bc2 = st.columns([1, 6])
        if bc1.button("← 목록으로", key="rec_back"):
            st.session_state.rec_detail_file = None
            st.rerun()
        if not df.empty:
            match = df[df["filename"] == detail_file]
            if not match.empty:
                show_training_detail(match.iloc[0], df, max_hr, ftp)
                return
        st.error("해당 기록을 찾을 수 없습니다.")
        return

    st.header("훈련 기록")

    if df.empty:
        st.info("저장된 훈련 기록이 없습니다.")
        return

    # 날짜 필터 (캘린더에서 이동 시 자동 설정)
    filter_date = st.session_state.get("rec_filter_date")
    if filter_date:
        fc1, fc2 = st.columns([4, 1])
        fc1.info(f"📅 {filter_date} 훈련 기록")
        if fc2.button("전체 보기", key="rec_date_clear"):
            st.session_state.rec_filter_date = None
            st.rerun()

    sports       = ["전체"] + sorted(df["sport"].dropna().unique().tolist())
    sport_filter = st.selectbox("종목 필터", sports, key="rec_sport_sel")
    filtered     = df if sport_filter == "전체" else df[df["sport"] == sport_filter]
    if filter_date:
        filtered = filtered[filtered["date"] == filter_date]

    if filtered.empty:
        st.info("해당 기록이 없습니다.")
        return

    hc = st.columns([1.1, 1.6, 0.9, 1.1, 1.0, 0.8, 0.5, 0.5])
    for col, label in zip(hc, ["날짜", "파일명", "종목", "거리(km)", "시간", "평균HR", "📋", "🗑️"]):
        col.markdown(f"**{label}**")
    st.markdown("---")

    for _, row in filtered.iterrows():
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.1, 1.6, 0.9, 1.1, 1.0, 0.8, 0.5, 0.5])
        c1.write(str(row["date"]))
        c2.write(os.path.basename(str(row.get("filename", ""))))
        c3.write(str(row.get("sport", "")))
        c4.write(f"{row['distance_km']:.2f}" if row.get("distance_km") else "-")
        c5.write(fmt_duration(row.get("duration_sec")))
        c6.write(f"{row['avg_hr']:.0f}" if row.get("avg_hr") else "-")
        if c7.button("📋", key=f"rec_det_{row['id']}", help="상세 보기"):
            st.session_state.rec_detail_file = str(row["filename"])
            st.rerun()
        if c8.button("🗑️", key=f"del_{row['id']}", help="삭제"):
            with get_conn() as conn:
                conn.execute("DELETE FROM training_log WHERE id = ?", (int(row["id"]),))
                conn.execute("DELETE FROM training_raw WHERE filename = ?", (str(row["filename"]),))
            st.rerun()


# ── 탭 4: 트렌드 ───────────────────────────────────────────────────────────────
def tab_trends(df: pd.DataFrame, ftp: int):
    st.header("훈련 트렌드")

    if df.empty or len(df) < 2:
        st.info("트렌드 분석을 위해 최소 2개 이상의 훈련 데이터가 필요합니다.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # ── 날짜 범위 슬라이더 ────────────────────────────────────────────────────
    min_d = df["date"].min().date()
    max_d = df["date"].max().date()
    if min_d < max_d:
        date_range = st.slider(
            "날짜 범위 필터",
            min_value=min_d, max_value=max_d,
            value=(min_d, max_d),
            format="YYYY-MM-DD",
        )
        pdf = df[
            (df["date"].dt.date >= date_range[0]) &
            (df["date"].dt.date <= date_range[1])
        ].copy()
    else:
        pdf = df.copy()

    if pdf.empty:
        st.info("선택한 기간에 데이터가 없습니다.")
        return

    # ── 1. W/bpm 효율 라인 차트 (실내=파란, 실외=초록) ───────────────────────
    st.subheader("⚡ W/bpm 효율 추이")
    wbpm_df = pdf.dropna(subset=["w_per_bpm"]).copy()
    if not wbpm_df.empty:
        wbpm_df["환경"] = wbpm_df["indoor"].apply(lambda x: "실내" if x else "실외")
        fig_wbpm = go.Figure()
        for env, color in [("실내", "#4A90D9"), ("실외", "#27AE60")]:
            sub = wbpm_df[wbpm_df["환경"] == env].sort_values("date")
            if not sub.empty:
                fig_wbpm.add_trace(go.Scatter(
                    x=sub["date"], y=sub["w_per_bpm"],
                    mode="lines+markers", name=env,
                    line=dict(color=color, width=2), marker=dict(size=6),
                    hovertemplate="%{x|%Y-%m-%d}<br>W/bpm: %{y:.3f}",
                ))
        fig_wbpm.update_layout(
            height=270, margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.12), yaxis_title="W/bpm",
        )
        st.plotly_chart(fig_wbpm, use_container_width=True)
    else:
        st.info("W/bpm 데이터가 없습니다. (파워+심박 데이터가 있는 사이클링 기록 필요)")

    st.markdown("---")

    # ── 2. HR 드리프트 바 차트 (≤7 초록, 7~10 주황, >10 빨강) ──────────────
    st.subheader("📊 HR 드리프트")
    drift_df = pdf.dropna(subset=["drift"]).copy()
    if not drift_df.empty:
        drift_df["등급"] = drift_df["drift"].apply(
            lambda v: "안정 (≤7%)" if abs(v) <= 7 else ("주의 (7~10%)" if abs(v) <= 10 else "위험 (>10%)")
        )
        color_map = {"안정 (≤7%)": "#27AE60", "주의 (7~10%)": "#F39C12", "위험 (>10%)": "#E24B4A"}
        fig_drift = px.bar(
            drift_df, x="date", y="drift", color="등급",
            color_discrete_map=color_map,
            labels={"date": "날짜", "drift": "드리프트 (%)"},
        )
        fig_drift.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
        fig_drift.update_layout(
            height=250, margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info("드리프트 데이터가 없습니다.")

    st.markdown("---")

    # ── 3. 월별 훈련 볼륨 바 차트 ────────────────────────────────────────────
    st.subheader("📅 월별 훈련 볼륨")
    vol_df = pdf.copy()
    vol_df["month"] = vol_df["date"].dt.to_period("M").dt.start_time
    monthly = vol_df.groupby(["month", "sport"])["distance_km"].sum().reset_index()
    if not monthly.empty:
        fig_vol = px.bar(
            monthly, x="month", y="distance_km", color="sport", barmode="stack",
            labels={"month": "월", "distance_km": "거리 (km)", "sport": "종목"},
            color_discrete_sequence=["#4A90D9", "#E24B4A", "#FAC775"],
        )
        fig_vol.update_layout(
            height=250, margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # ── 4. 러닝 페이스 + 심박 듀얼 차트 (Y축 역방향) ─────────────────────────
    run_df = pdf[pdf["sport"].str.lower().str.contains("run", na=False)].dropna(subset=["avg_pace"])
    if not run_df.empty:
        st.subheader("🏃 러닝 — 페이스 & 심박 추이")
        fig_run = make_subplots(specs=[[{"secondary_y": True}]])
        fig_run.add_trace(go.Scatter(
            x=run_df["date"], y=run_df["avg_pace"],
            name="평균 페이스 (초/km)", mode="lines+markers",
            line=dict(color="#4A90D9", width=2), marker=dict(size=6),
            hovertemplate="%{x|%Y-%m-%d}<br>페이스: %{y:.0f}초/km",
        ), secondary_y=False)
        if run_df["avg_hr"].notna().any():
            fig_run.add_trace(go.Scatter(
                x=run_df["date"], y=run_df["avg_hr"],
                name="평균 심박 (bpm)", mode="lines+markers",
                line=dict(color="#E24B4A", width=2), marker=dict(size=6),
                hovertemplate="%{x|%Y-%m-%d}<br>심박: %{y:.0f} bpm",
            ), secondary_y=True)
        fig_run.update_yaxes(title_text="페이스 (초/km)", secondary_y=False, autorange="reversed")
        fig_run.update_yaxes(title_text="심박 (bpm)", secondary_y=True)
        fig_run.update_layout(
            height=270, margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_run, use_container_width=True)
        st.markdown("---")

    # ── 5. PMC 훈련 부하 (CTL / ATL / TSB) ──────────────────────────────────
    st.subheader("📈 훈련 부하 PMC — 피트니스 / 피로 / 컨디션")

    def calc_trimp(row):
        try:
            hr  = float(row["avg_hr"])
            dur = float(row["duration_sec"])
            mhr = float(row["max_hr"]) if pd.notna(row.get("max_hr")) else 185.0
            if hr <= 0 or dur <= 0: return 0.0
            ratio = max(0.0, (hr - 50) / max(1.0, mhr - 50))
            return (dur / 60) * ratio * math.exp(1.92 * ratio)
        except Exception:
            return 0.0

    today_ts    = pd.Timestamp(date.today())
    df["trimp"] = df.apply(calc_trimp, axis=1)
    all_dates   = pd.date_range(df["date"].min(), today_ts, freq="D")
    daily_base  = pd.DataFrame({"date": all_dates})
    daily_trimp = df.groupby(df["date"].dt.normalize())["trimp"].sum().reset_index()
    daily_trimp.columns = ["date", "trimp"]
    daily_load  = daily_base.merge(daily_trimp, on="date", how="left").fillna({"trimp": 0.0})
    daily_load  = daily_load.sort_values("date").reset_index(drop=True)

    EXP_CTL = math.exp(-1 / 42); EXP_ATL = math.exp(-1 / 7)
    K_CTL = 1 - EXP_CTL;         K_ATL = 1 - EXP_ATL
    ctls, atls, ctl_v, atl_v = [], [], 0.0, 0.0
    for t in daily_load["trimp"]:
        ctl_v = ctl_v * EXP_CTL + t * K_CTL
        atl_v = atl_v * EXP_ATL + t * K_ATL
        ctls.append(round(ctl_v, 2)); atls.append(round(atl_v, 2))

    daily_load["CTL"] = ctls
    daily_load["ATL"] = atls
    daily_load["TSB"] = daily_load["CTL"] - daily_load["ATL"]
    plot_load = daily_load[
        (daily_load["date"] >= pd.Timestamp(date_range[0])) &
        (daily_load["date"] <= pd.Timestamp(date_range[1]))
    ] if min_d < max_d else daily_load

    fig_pmc = go.Figure()
    fig_pmc.add_trace(go.Scatter(x=plot_load["date"], y=plot_load["CTL"],
        name="CTL 피트니스", line=dict(color="#4A90D9", width=2.5)))
    fig_pmc.add_trace(go.Scatter(x=plot_load["date"], y=plot_load["ATL"],
        name="ATL 피로",     line=dict(color="#E24B4A", width=2.5)))
    fig_pmc.add_trace(go.Scatter(x=plot_load["date"], y=plot_load["TSB"],
        name="TSB 컨디션",  line=dict(color="#27AE60", width=2),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.1)"))
    fig_pmc.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig_pmc.update_layout(
        height=320, margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.12), yaxis_title="TRIMP 점수",
    )
    st.plotly_chart(fig_pmc, use_container_width=True)
    st.caption("CTL 42일 누적 피트니스 · ATL 7일 누적 피로 · TSB = CTL−ATL (양수=컨디션 좋음)")

    st.markdown("---")

    # ── 6. 종목 비율 + 개인 기록 ─────────────────────────────────────────────
    pie_col, pr_col = st.columns([1, 2])
    with pie_col:
        st.subheader("🥧 종목 비율")
        sc = df["sport"].value_counts().reset_index()
        sc.columns = ["sport", "count"]
        fig_pie = px.pie(sc, names="sport", values="count", hole=0.4,
                         color_discrete_sequence=["#4A90D9", "#E24B4A", "#FAC775"])
        fig_pie.update_layout(height=240, margin=dict(t=20, b=5, l=5, r=5))
        st.plotly_chart(fig_pie, use_container_width=True)

    with pr_col:
        st.subheader("🏅 개인 기록 (전체 기간)")
        pr_c, pr_r = st.columns(2)
        with pr_c:
            st.markdown("**🚴 사이클링**")
            cdf = df[df["sport"].str.lower().str.contains("cycl|bike|cycling", na=False)]
            if not cdf.empty:
                wbpm_c = cdf["w_per_bpm"].dropna(); pwr_c = cdf["avg_power"].dropna(); dist_c = cdf["distance_km"].dropna()
                if len(wbpm_c): st.metric("최고 W/bpm",    f"{wbpm_c.max():.3f}")
                if len(pwr_c):  st.metric("최고 평균 파워", f"{pwr_c.max():.0f} W")
                if len(dist_c): st.metric("최장 라이드",    f"{dist_c.max():.1f} km")
            else: st.info("기록 없음")
        with pr_r:
            st.markdown("**🏃 러닝**")
            rdf = df[df["sport"].str.lower().str.contains("run", na=False)]
            if not rdf.empty:
                pace_r = rdf["avg_pace"].dropna(); dist_r = rdf["distance_km"].dropna(); hr_r = rdf["avg_hr"].dropna()
                if len(pace_r): st.metric("최고 평균 페이스", fmt_pace(pace_r.min()))
                if len(dist_r): st.metric("최장 런",          f"{dist_r.max():.1f} km")
                if len(hr_r):   st.metric("최저 평균 심박",   f"{hr_r.min():.0f} bpm")
            else: st.info("기록 없음")


# ── 탭 5: 연간 히트맵 ──────────────────────────────────────────────────────────
def tab_heatmap(df: pd.DataFrame):
    st.header("연간 훈련 히트맵")

    if df.empty:
        st.info("데이터가 없습니다.")
        return

    df_c = df.copy()
    df_c["date"] = pd.to_datetime(df_c["date"]).dt.date

    # 연도 선택
    data_years = sorted(df_c["date"].apply(lambda d: d.year).unique(), reverse=True)
    today_year = date.today().year
    if today_year not in data_years:
        data_years = [today_year] + list(data_years)
    year = st.selectbox("연도 선택", data_years, index=0)

    # 일별 집계
    year_df = df_c[df_c["date"].apply(lambda d: d.year) == year]
    if not year_df.empty:
        daily_agg = year_df.groupby("date").agg(
            duration_sec=("duration_sec", "sum"),
            n=("id", "count"),
            sports=("sport", lambda x: "/".join(sorted(x.dropna().unique()))),
            distance_km=("distance_km", "sum"),
        ).reset_index()
        daily_dict = {row["date"]: row.to_dict() for _, row in daily_agg.iterrows()}
    else:
        daily_dict = {}

    # 그리드 생성 (7행 × ~53열)
    start     = date(year, 1, 1)
    end       = date(year, 12, 31)
    n_days    = (end - start).days + 1
    start_dow = start.weekday()               # 0=월
    n_weeks   = (start_dow + n_days + 6) // 7

    DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]
    z      = [[None] * n_weeks for _ in range(7)]
    text   = [[""]   * n_weeks for _ in range(7)]

    for idx in range(n_days):
        d     = start + timedelta(days=idx)
        col   = (idx + start_dow) // 7
        row_i = (idx + start_dow) % 7

        if d in daily_dict:
            rec     = daily_dict[d]
            dur_min = (rec.get("duration_sec") or 0) / 60
            val     = 4 if dur_min >= 90 else 3 if dur_min >= 60 else 2 if dur_min >= 30 else 1 if dur_min > 0 else 0
            z[row_i][col] = val
            text[row_i][col] = (
                f"<b>{d.strftime('%Y년 %m월 %d일')} ({DOW_KR[row_i]})</b><br>"
                f"훈련 {rec['n']}회 · {rec['sports']}<br>"
                f"시간 {fmt_duration(int(rec['duration_sec']))}"
                + (f" · {rec['distance_km']:.1f}km" if rec.get("distance_km") else "")
            )
        else:
            z[row_i][col] = 0
            text[row_i][col] = f"<b>{d.strftime('%Y년 %m월 %d일')} ({DOW_KR[row_i]})</b><br>휴식"

    # 이산 컬러스케일 (5단계)
    eps = 1e-6
    colorscale = [
        [0.0,        "#EEEEEE"], [0.25 - eps, "#EEEEEE"],
        [0.25,       "#C6E48B"], [0.50 - eps, "#C6E48B"],
        [0.50,       "#7BC96F"], [0.75 - eps, "#7BC96F"],
        [0.75,       "#239A3B"], [1.00 - eps, "#239A3B"],
        [1.00,       "#196127"],
    ]

    # 월 레이블 (x축 위쪽)
    month_x, month_lb = [], []
    for m in range(1, 13):
        ms  = date(year, m, 1)
        col = ((ms - start).days + start_dow) // 7
        month_x.append(col)
        month_lb.append(f"{m}월")

    fig_h = go.Figure(go.Heatmap(
        z=z, text=text, hoverinfo="text",
        colorscale=colorscale, showscale=False,
        xgap=3, ygap=3, zmin=0, zmax=4,
    ))
    fig_h.update_layout(
        height=185, margin=dict(t=40, b=10, l=45, r=10),
        xaxis=dict(tickvals=month_x, ticktext=month_lb, showgrid=False, side="top"),
        yaxis=dict(tickvals=list(range(7)), ticktext=DOW_KR, showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # 범례
    st.markdown(
        "<div style='display:flex;gap:14px;font-size:0.82em;align-items:center;margin-top:-10px'>"
        "<span style='color:#666'>훈련 시간:</span>"
        "<span style='background:#EEEEEE;border:1px solid #ccc;padding:2px 10px;border-radius:3px'>휴식</span>"
        "<span style='background:#C6E48B;padding:2px 10px;border-radius:3px'>~30분</span>"
        "<span style='background:#7BC96F;padding:2px 10px;border-radius:3px'>30~60분</span>"
        "<span style='background:#239A3B;color:white;padding:2px 10px;border-radius:3px'>60~90분</span>"
        "<span style='background:#196127;color:white;padding:2px 10px;border-radius:3px'>90분+</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 월별 통계 바 차트 ─────────────────────────────────────────────────────
    st.markdown("---")
    if not year_df.empty:
        yr2 = year_df.copy()
        yr2["month_num"] = yr2["date"].apply(lambda d: d.month)
        mstats = yr2.groupby("month_num").agg(
            sessions=("id", "count"),
            distance=("distance_km", "sum"),
            time_sec=("duration_sec", "sum"),
        ).reset_index()

        fig_ms = make_subplots(rows=1, cols=2,
                               subplot_titles=[f"{year}년 월별 거리 (km)", "월별 훈련 횟수"])
        fig_ms.add_trace(go.Bar(x=mstats["month_num"], y=mstats["distance"],
                                marker_color="#4A90D9", name="거리"), row=1, col=1)
        fig_ms.add_trace(go.Bar(x=mstats["month_num"], y=mstats["sessions"],
                                marker_color="#27AE60", name="횟수"), row=1, col=2)
        fig_ms.update_xaxes(tickvals=list(range(1, 13)),
                            ticktext=[f"{m}월" for m in range(1, 13)])
        fig_ms.update_layout(height=250, margin=dict(t=35, b=10, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_ms, use_container_width=True)

        # 연간 합계
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("연간 훈련 횟수", f"{len(year_df)}회")
        c2.metric("연간 총 거리",   f"{year_df['distance_km'].sum():.0f} km")
        c3.metric("연간 총 시간",   fmt_duration(year_df["duration_sec"].sum()))
        freq = 365 / max(len(daily_dict), 1)
        c4.metric("평균 훈련 간격", f"{freq:.1f}일에 1회")

    # ── 날짜 클릭 → 상세 분석 ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 날짜 상세 분석")

    if daily_dict:
        trained_dates = sorted(daily_dict.keys(), reverse=True)
        sel_date = st.selectbox(
            "훈련 날짜 선택",
            trained_dates,
            format_func=lambda d: (
                f"{d.strftime('%Y-%m-%d')}  "
                f"{daily_dict[d]['sports']}  "
                f"{fmt_duration(int(daily_dict[d]['duration_sec']))}"
            ),
        )
        if sel_date:
            sessions = year_df[year_df["date"] == sel_date]
            for _, row in sessions.iterrows():
                sport  = str(row.get("sport", "")).lower()
                is_run = "run" in sport
                icon   = "🏃" if is_run else "🚴"
                fname  = os.path.basename(str(row.get("filename", "")))
                with st.expander(f"{icon} {fname} — {row.get('sport','')}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("거리",      f"{row['distance_km']:.2f} km" if pd.notna(row.get("distance_km") or float("nan")) else "-")
                    c2.metric("시간",      fmt_duration(row.get("duration_sec")))
                    c3.metric("평균 심박", f"{row['avg_hr']:.0f} bpm"     if pd.notna(row.get("avg_hr") or float("nan")) else "-")
                    if is_run:
                        c4.metric("평균 페이스", fmt_pace(row.get("avg_pace")))
                    else:
                        c4.metric("평균 파워", f"{row['avg_power']:.0f} W" if pd.notna(row.get("avg_power") or float("nan")) else "-")

                    z_vals = [row.get(f"z{i}_pct") or 0 for i in range(1, 6)]
                    if any(v > 0 for v in z_vals):
                        fig_z = go.Figure(go.Bar(
                            x=ZONE_NAMES, y=z_vals, marker_color=ZONE_COLORS,
                            text=[f"{v:.1f}%" for v in z_vals], textposition="outside",
                        ))
                        fig_z.update_layout(
                            yaxis=dict(range=[0, max(z_vals) * 1.3 + 5]),
                            height=190, margin=dict(t=15, b=10, l=10, r=10), showlegend=False,
                        )
                        st.plotly_chart(fig_z, use_container_width=True)

                    raw_df = load_raw(str(row.get("filename", "")))
                    if not raw_df.empty and "secs" in raw_df.columns:
                        raw_df = raw_df.copy()
                        raw_df["min"] = (raw_df["secs"] // 60).astype(int)
                        if is_run and {"hr", "pace"}.issubset(raw_df.columns):
                            mdf2 = raw_df.groupby("min").agg(hr=("hr","mean"), pace=("pace","median")).reset_index()
                            fig_r = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_r.add_trace(go.Scatter(x=mdf2["min"], y=mdf2["hr"],   name="심박",   line=dict(color="#E24B4A", width=1.5)), secondary_y=False)
                            fig_r.add_trace(go.Scatter(x=mdf2["min"], y=mdf2["pace"], name="페이스", line=dict(color="#4A90D9", width=1.5)), secondary_y=True)
                            fig_r.update_yaxes(secondary_y=True, autorange="reversed")
                            fig_r.update_layout(height=200, margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation="h",y=1.15))
                            st.plotly_chart(fig_r, use_container_width=True)
                        elif not is_run and {"hr", "watts"}.issubset(raw_df.columns):
                            mdf2 = raw_df.groupby("min").agg(hr=("hr","mean"), watts=("watts","mean")).reset_index()
                            fig_r = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_r.add_trace(go.Scatter(x=mdf2["min"], y=mdf2["hr"],    name="심박", line=dict(color="#E24B4A", width=1.5)), secondary_y=False)
                            fig_r.add_trace(go.Scatter(x=mdf2["min"], y=mdf2["watts"], name="파워", line=dict(color="#4A90D9", width=1.5)), secondary_y=True)
                            fig_r.update_layout(height=200, margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation="h",y=1.15))
                            st.plotly_chart(fig_r, use_container_width=True)

                    if is_run:
                        laps = load_laps(str(row.get("filename", "")))
                        if laps:
                            st.markdown("**km 랩**")
                            st.dataframe(pd.DataFrame(laps), use_container_width=True, hide_index=True)
    else:
        st.info(f"{year}년 훈련 기록이 없습니다.")




# ── 리포트 헬퍼 ────────────────────────────────────────────────────────────────
def _ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _report_filename(period_type: str, period_start: date) -> str:
    slug = {"주간": "weekly", "월간": "monthly", "분기": "quarterly",
            "반기": "halfyear", "연간": "annual"}.get(period_type, period_type)
    return os.path.join(REPORTS_DIR, f"report_{slug}_{period_start.strftime('%Y-%m-%d')}.pdf")


def _period_range(period_type: str, label: str):
    if period_type == "주간":
        yr, wk = int(label[:4]), int(label[6:])
        s = date.fromisocalendar(yr, wk, 1)
        return s, s + timedelta(days=6)
    if period_type == "월간":
        yr, mo = int(label[:4]), int(label[5:7])
        s = date(yr, mo, 1)
        return s, date(yr, mo, calendar.monthrange(yr, mo)[1])
    if period_type == "분기":
        yr, q = int(label[:4]), int(label[6])
        sm, em = (q - 1) * 3 + 1, q * 3
        return date(yr, sm, 1), date(yr, em, calendar.monthrange(yr, em)[1])
    if period_type == "반기":
        yr, h = int(label[:4]), int(label[6])
        return (date(yr, 1, 1), date(yr, 6, 30)) if h == 1 else (date(yr, 7, 1), date(yr, 12, 31))
    yr = int(label[:4])
    return date(yr, 1, 1), date(yr, 12, 31)


def _period_labels(df: pd.DataFrame, period_type: str) -> list:
    dc = df.copy()
    dc["date"] = pd.to_datetime(dc["date"]).dt.date
    dates = sorted(dc["date"].unique(), reverse=True)
    seen, result = set(), []
    for d in dates:
        if period_type == "주간":
            iso = d.isocalendar()
            lb = f"{iso.year}-W{iso.week:02d}"
        elif period_type == "월간":
            lb = f"{d.year}-{d.month:02d}"
        elif period_type == "분기":
            lb = f"{d.year}-Q{(d.month - 1) // 3 + 1}"
        elif period_type == "반기":
            lb = f"{d.year}-H{1 if d.month <= 6 else 2}"
        else:
            lb = str(d.year)
        if lb not in seen:
            seen.add(lb); result.append(lb)
    return result


def _coaching_summary(pf: pd.DataFrame) -> str:
    if pf.empty:
        return "No training data for this period."
    lines = []
    n = len(pf)
    dist = pf["distance_km"].sum()
    time_h = pf["duration_sec"].sum() / 3600
    lines.append(f"Total {n} sessions: {dist:.1f} km, {time_h:.1f} hours.")

    wbpm = pf["w_per_bpm"].dropna()
    if len(wbpm) > 1:
        mid = len(wbpm) // 2
        a1, a2 = wbpm.iloc[:mid].mean(), wbpm.iloc[mid:].mean()
        if a2 > a1 * 1.03:
            lines.append(f"W/bpm efficiency trending up ({a1:.2f} to {a2:.2f}). Great fitness gains!")
        elif a2 < a1 * 0.97:
            lines.append(f"W/bpm efficiency declining ({a1:.2f} to {a2:.2f}). Consider extra recovery.")
        else:
            lines.append(f"W/bpm efficiency stable at {wbpm.mean():.2f}. Consistent training!")
    elif len(wbpm) == 1:
        lines.append(f"W/bpm efficiency: {wbpm.iloc[0]:.2f}.")

    drift = pf["drift"].dropna()
    if len(drift) > 0:
        avg_d = drift.mean()
        if abs(avg_d) <= 5:
            lines.append("HR drift is low — excellent pacing control.")
        elif abs(avg_d) <= 10:
            lines.append(f"Average HR drift {avg_d:+.1f}%. Monitor hydration and even pacing.")
        else:
            lines.append(f"High HR drift ({avg_d:+.1f}%). Reduce starting intensity or improve recovery.")

    z2_avg  = pf["z2_pct"].mean()
    z45_avg = (pf["z4_pct"] + pf["z5_pct"]).mean()
    if not pd.isna(z2_avg)  and z2_avg  > 60:
        lines.append(f"Z2 aerobic focus {z2_avg:.0f}% — excellent base-building approach.")
    if not pd.isna(z45_avg) and z45_avg > 30:
        lines.append(f"High intensity {z45_avg:.0f}% — ensure adequate recovery between hard sessions.")
    return " ".join(lines)


def generate_pdf_report(period_df: pd.DataFrame, period_type: str,
                        period_start: date, period_end: date) -> bytes:
    import matplotlib
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Korean font for matplotlib
    try:
        import matplotlib.font_manager as fm
        for fn_name in ['Malgun Gothic', 'NanumGothic', 'Apple SD Gothic Neo', 'AppleGothic']:
            if any(f.name == fn_name for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = fn_name
                break
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Korean font for fpdf2
    kr_font = None
    for fp in [
        r"C:\Windows\Fonts\malgunsl.ttf", r"C:\Windows\Fonts\malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]:
        if os.path.exists(fp):
            try:
                pdf.add_font("KR", "", fp, uni=True)
                kr_font = "KR"
                break
            except Exception:
                pass

    def fnt(sz): return ("KR" if kr_font else "Helvetica"), "", sz

    def _add_chart(fig, h=52):
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            buf.seek(0)
            tmp = os.path.join(tempfile.gettempdir(), f"rpt_{id(fig)}.png")
            with open(tmp, 'wb') as fo:
                fo.write(buf.read())
            pdf.image(tmp, x=15, y=pdf.get_y(), w=170, h=h)
            pdf.ln(h + 3)
            try: os.remove(tmp)
            except: pass
        except Exception as e:
            plt.close(fig)
            pdf.set_font(*fnt(8))
            pdf.cell(0, 5, f"[Chart error: {e}]", ln=True)

    pf = period_df.copy()
    pf["date"]   = pd.to_datetime(pf["date"])
    pf["date_d"] = pf["date"].dt.date

    type_en = {"주간": "Weekly", "월간": "Monthly", "분기": "Quarterly",
               "반기": "Semi-Annual", "연간": "Annual"}

    # ── PAGE 1 : Title + Summary + W/bpm + Drift ─────────────────────────────
    pdf.add_page()
    pdf.set_font(*fnt(20)); pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, f"Training Report  —  {type_en.get(period_type, period_type)}", ln=True, align="C")
    pdf.set_font(*fnt(10)); pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, f"{period_start.strftime('%Y-%m-%d')}  ~  {period_end.strftime('%Y-%m-%d')}", ln=True, align="C")
    pdf.cell(0, 4, f"Generated: {date.today()}", ln=True, align="C")
    pdf.ln(6)

    # Summary row
    n_sess = len(pf)
    tot_d  = pf["distance_km"].sum()
    tot_t  = pf["duration_sec"].sum()
    avg_hr = pf["avg_hr"].dropna().mean()
    avg_wb = pf["w_per_bpm"].dropna().mean()
    items  = [
        ("Sessions",  str(n_sess)),
        ("Distance",  f"{tot_d:.1f} km"),
        ("Time",      fmt_duration(int(tot_t))),
        ("Avg HR",    f"{avg_hr:.0f} bpm" if not pd.isna(avg_hr) else "-"),
        ("Avg W/bpm", f"{avg_wb:.3f}"     if not pd.isna(avg_wb) else "-"),
    ]
    pdf.set_fill_color(245, 247, 250); pdf.set_draw_color(210, 215, 220)
    box_y = pdf.get_y(); cw = 36
    pdf.rect(15, box_y, 180, 22, 'FD')
    for i, (lbl, val) in enumerate(items):
        x = 15 + i * cw
        pdf.set_xy(x, box_y + 3); pdf.set_font(*fnt(8)); pdf.set_text_color(120, 120, 120)
        pdf.cell(cw, 5, lbl, align="C")
        pdf.set_xy(x, box_y + 10); pdf.set_font(*fnt(11)); pdf.set_text_color(30, 30, 30)
        pdf.cell(cw, 7, val, align="C")
    pdf.set_y(box_y + 26)

    # W/bpm chart
    pdf.set_font(*fnt(12)); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "W/bpm Efficiency Trend", ln=True)
    wbpm_d = pf.dropna(subset=["w_per_bpm"])
    if not wbpm_d.empty:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        for indoor, color, label in [(1, '#4A90D9', 'Indoor'), (0, '#27AE60', 'Outdoor')]:
            sub = wbpm_d[wbpm_d["indoor"] == indoor]
            if not sub.empty:
                ax.plot(sub["date_d"], sub["w_per_bpm"], 'o-', color=color,
                        label=label, linewidth=1.5, markersize=4)
        ax.set_ylabel("W/bpm", fontsize=8); ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if wbpm_d["indoor"].nunique() > 1: ax.legend(fontsize=7)
        fig.tight_layout(pad=0.5); _add_chart(fig, h=52)
    else:
        pdf.set_font(*fnt(9)); pdf.cell(0, 6, "No W/bpm data.", ln=True)

    # Drift chart
    pdf.set_font(*fnt(12)); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "HR Drift Trend", ln=True)
    drift_d = pf.dropna(subset=["drift"])
    if not drift_d.empty:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        colors = ['#27AE60' if abs(v) <= 7 else '#F39C12' if abs(v) <= 10 else '#E24B4A'
                  for v in drift_d["drift"]]
        ax.bar(drift_d["date_d"], drift_d["drift"], color=colors, width=0.6)
        ax.axhline(0, color='#bbb', linewidth=0.8, linestyle='--')
        ax.set_ylabel("Drift %", fontsize=8); ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        handles = [mpatches.Patch(color='#27AE60', label='Stable ≤7%'),
                   mpatches.Patch(color='#F39C12', label='Caution 7-10%'),
                   mpatches.Patch(color='#E24B4A', label='High >10%')]
        ax.legend(handles=handles, fontsize=7)
        fig.tight_layout(pad=0.5); _add_chart(fig, h=52)
    else:
        pdf.set_font(*fnt(9)); pdf.cell(0, 6, "No drift data.", ln=True)

    # ── PAGE 2 : Zone Distribution + Weekly Volume ────────────────────────────
    pdf.add_page()

    # Zone distribution (bar + pie)
    pdf.set_font(*fnt(12)); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "HR Zone Distribution  (cumulative average)", ln=True)
    zone_cols  = ["z1_pct", "z2_pct", "z3_pct", "z4_pct", "z5_pct"]
    zone_avgs  = [0.0 if pd.isna(pf[c].mean()) else float(pf[c].mean()) for c in zone_cols]
    zone_en    = ["Z1 Recovery", "Z2 Aerobic", "Z3 Tempo", "Z4 Threshold", "Z5 Max"]
    z_colors   = ["#B5D4F4", "#9FE1CB", "#FAC775", "#F0997B", "#E24B4A"]
    if sum(zone_avgs) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
        bars = ax1.bar(["Z1","Z2","Z3","Z4","Z5"], zone_avgs, color=z_colors)
        ax1.set_ylabel("Avg %", fontsize=7); ax1.tick_params(labelsize=7)
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        for bar, val in zip(bars, zone_avgs):
            if val > 0.5:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val:.0f}%', ha='center', va='bottom', fontsize=6)
        nz = [(n, v, c) for n, v, c in zip(zone_en, zone_avgs, z_colors) if v > 0]
        if nz:
            ns, vs, cs = zip(*nz)
            ax2.pie(vs, labels=None, colors=cs, autopct='%1.0f%%',
                    pctdistance=0.7, textprops={'fontsize': 6})
            ax2.legend(ns, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=5.5)
        fig.tight_layout(pad=0.3); _add_chart(fig, h=60)
    else:
        pdf.set_font(*fnt(9)); pdf.cell(0, 6, "No zone data.", ln=True)

    # Weekly volume
    pdf.set_font(*fnt(12)); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Weekly Training Volume", ln=True)
    wv = pf.copy()
    wv["week_start"] = wv["date_d"].apply(lambda d: d - timedelta(days=d.weekday()))
    wgrp = wv.groupby("week_start").agg(
        dist=("distance_km", "sum"),
        n_run=("sport", lambda x: sum(1 for s in x if "run" in str(s).lower())),
        n_cyc=("sport", lambda x: sum(1 for s in x if "run" not in str(s).lower())),
    ).reset_index()
    if not wgrp.empty:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        xlabels = [d.strftime("%m/%d") for d in wgrp["week_start"]]
        ax.bar(xlabels, wgrp["dist"], color='#4A90D9')
        for xi, (_, row) in enumerate(wgrp.iterrows()):
            parts = []
            if row["n_cyc"] > 0: parts.append(f"C{int(row['n_cyc'])}")
            if row["n_run"] > 0: parts.append(f"R{int(row['n_run'])}")
            if parts:
                ax.text(xi, row["dist"] + 0.2, " ".join(parts),
                        ha='center', va='bottom', fontsize=7)
        ax.set_ylabel("km", fontsize=8); ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout(pad=0.5); _add_chart(fig, h=52)
    else:
        pdf.set_font(*fnt(9)); pdf.cell(0, 6, "No data.", ln=True)

    # ── PAGE 3 : Best Records + Coaching Summary ──────────────────────────────
    pdf.add_page()
    pdf.set_font(*fnt(12)); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Best Records", ln=True)

    best_wb   = pf["w_per_bpm"].dropna()
    best_dist = pf["distance_km"].dropna()
    best_dr   = pf["drift"].dropna()
    best_pwr  = pf["avg_power"].dropna()
    best_pace = pf["avg_pace"].dropna()
    bests = [
        ("Best W/bpm",       f"{best_wb.max():.3f}"     if len(best_wb)   else "-"),
        ("Longest Distance", f"{best_dist.max():.1f} km" if len(best_dist) else "-"),
        ("Best HR Drift",    f"{best_dr.abs().min():.1f}%" if len(best_dr) else "-"),
        ("Best Avg Power",   f"{best_pwr.max():.0f} W"  if len(best_pwr)  else "-"),
        ("Best Avg Pace",    fmt_pace(best_pace.min())   if len(best_pace) else "-"),
        ("Total Sessions",   str(n_sess)),
    ]
    pdf.set_fill_color(248, 249, 251); pdf.set_draw_color(215, 220, 225)
    box_y = pdf.get_y(); cw2 = 58; rh = 18
    for i, (lbl, val) in enumerate(bests):
        cx = 15 + (i % 3) * (cw2 + 2)
        cy = box_y + (i // 3) * (rh + 2)
        pdf.rect(cx, cy, cw2, rh, 'FD')
        pdf.set_xy(cx + 2, cy + 2); pdf.set_font(*fnt(8)); pdf.set_text_color(130, 130, 130)
        pdf.cell(cw2 - 4, 5, lbl)
        pdf.set_xy(cx + 2, cy + 8); pdf.set_font(*fnt(11)); pdf.set_text_color(30, 30, 30)
        pdf.cell(cw2 - 4, 7, val)
    pdf.set_y(box_y + ((len(bests) + 2) // 3) * (rh + 2) + 8)

    # Coaching summary
    pdf.set_font(*fnt(12)); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Coaching Summary", ln=True)
    summary = _coaching_summary(pf)
    pdf.set_fill_color(252, 253, 255); pdf.set_draw_color(200, 210, 230)
    pdf.set_text_color(50, 50, 50); pdf.set_font(*fnt(10))
    pdf.multi_cell(180, 6, summary, border=1, fill=True)
    pdf.ln(6)

    # Footer
    pdf.set_font(*fnt(8)); pdf.set_text_color(170, 170, 170)
    pdf.cell(0, 5, f"Training Analysis App  |  Generated {date.today()}", align="C", ln=True)

    return bytes(pdf.output())


# ── 단일 훈련 PDF 리포트 ──────────────────────────────────────────────────────
def generate_training_pdf(row: dict, df: pd.DataFrame, max_hr: int, ftp: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from fpdf import FPDF

    sport      = str(row.get("sport", "")).lower()
    is_running = "run" in sport
    fname      = str(row.get("filename", ""))
    date_str   = str(row.get("date", ""))

    kr_font = None
    for fp in [r"C:\Windows\Fonts\malgunsl.ttf", r"C:\Windows\Fonts\malgun.ttf"]:
        if os.path.exists(fp):
            kr_font = fp
            break

    for fn in ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic"]:
        try:
            fm.findfont(fn, fallback_to_default=False)
            plt.rcParams["font.family"] = fn
            break
        except Exception:
            pass

    df_c = df.copy()
    df_c["date"] = pd.to_datetime(df_c["date"])
    sport_key = "run" if is_running else "cycl"
    hist = df_c[df_c["sport"].str.contains(sport_key, case=False, na=False)].sort_values("date").tail(10)

    # training count and prev W/bpm
    _total_count = len(df_c[df_c["sport"].str.contains(sport_key, case=False, na=False)])
    _prev_wbpm   = None
    if not is_running:
        prev_rows = hist[hist["filename"] != fname]["w_per_bpm"].dropna()
        if len(prev_rows) > 0:
            _prev_wbpm = float(prev_rows.iloc[-1])

    # load raw data early to compute half-splits
    raw_df = load_raw(fname)
    _hr1 = _hr2 = _pwr1 = _pwr2 = None
    if not raw_df.empty and "hr" in raw_df.columns:
        hr_s = raw_df["hr"].dropna()
        if len(hr_s) >= 4:
            mid  = len(hr_s) // 2
            _hr1 = round(float(hr_s.iloc[:mid].mean()), 1)
            _hr2 = round(float(hr_s.iloc[mid:].mean()), 1)
    if not is_running and not raw_df.empty and "watts" in raw_df.columns:
        pw_s = raw_df["watts"].dropna()
        pw_s = pw_s[pw_s > 0]
        if len(pw_s) >= 4:
            mid  = len(pw_s) // 2
            _pwr1 = round(float(pw_s.iloc[:mid].mean()), 1)
            _pwr2 = round(float(pw_s.iloc[mid:].mean()), 1)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    if kr_font:
        pdf.add_font("KR", "", kr_font)
        pdf.add_font("KR", "B", kr_font)

    def fnt(sz, bold=False):
        return ("KR" if kr_font else "Helvetica"), ("B" if bold else ""), sz

    def _add_chart(fig, h=60):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        import tempfile as _tf
        tmp = _tf.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(buf.read()); tmp.close()
        pdf.image(tmp.name, x=None, w=pdf.epw, h=h)
        os.unlink(tmp.name)

    # ── Page 1: 정보 + 존 분포 + 분당 차트 ──────────────────────────────────
    pdf.add_page()
    fam, sty, sz = fnt(14, True)
    pdf.set_font(fam, sty, sz)
    icon_txt = "러닝" if is_running else "사이클링"
    pdf.cell(pdf.epw, 8, f"[{icon_txt}] 훈련 리포트  {date_str}", ln=1, align="C")
    pdf.ln(2)

    _wbpm_change_str = "-"
    if not is_running and row.get("w_per_bpm") and _prev_wbpm:
        diff = row["w_per_bpm"] - _prev_wbpm
        _wbpm_change_str = f"{diff:+.3f} (전 {_prev_wbpm:.3f})"

    fields = [
        ("종목",          row.get("sport", "-")),
        ("파일",          os.path.basename(fname)),
        ("코스",          row.get("course_name") or "-"),
        ("실내/실외",     "실내" if row.get("indoor") else "실외"),
        ("거리",          f"{row['distance_km']:.2f} km"       if row.get("distance_km")   else "-"),
        ("이동 시간",     fmt_duration(row.get("moving_time_sec") or row.get("duration_sec"))),
        ("정지 시간",     fmt_duration(row.get("stop_time_sec")) if row.get("stop_time_sec") else "-"),
        ("평균 속도",     f"{row['avg_speed_kph']:.1f} km/h"   if row.get("avg_speed_kph") else "-"),
        ("칼로리",        f"{row['calories']:.0f} kcal"         if row.get("calories")      else "-"),
        ("실내 온도",     f"{row['indoor_temp']:.1f} °C"        if row.get("indoor_temp")   else "-"),
        ("총 훈련 횟수",  f"{_total_count}회"),
        ("평균 심박",     f"{row['avg_hr']:.0f} bpm"            if row.get("avg_hr")        else "-"),
        ("최대 심박",     f"{row['max_hr']:.0f} bpm"            if row.get("max_hr")        else "-"),
        ("전반부 평균 심박", f"{_hr1:.0f} bpm" if _hr1 is not None else "-"),
        ("후반부 평균 심박", f"{_hr2:.0f} bpm" if _hr2 is not None else "-"),
        ("심박 드리프트", f"{row['drift']:+.1f}%  ({_hr2 - _hr1:+.1f} bpm)" if (row.get("drift") and _hr1 is not None and _hr2 is not None) else
                          (f"{row['drift']:+.1f}%" if row.get("drift") else "-")),
        ("드리프트 평가", row.get("drift_grade") or "-"),
    ]
    if is_running:
        fields += [
            ("평균 페이스",     fmt_pace(row.get("avg_pace"))),
            ("최고 페이스",     fmt_pace(row.get("best_pace"))),
            ("페이스 드리프트", f"{row['pace_drift_sec']:+.0f}초/km" if row.get("pace_drift_sec") else "-"),
            ("평균 케이던스",   f"{row['avg_cadence']:.0f} spm"       if row.get("avg_cadence")   else "-"),
        ]
    else:
        fields += [
            ("전반부 평균 파워", f"{_pwr1:.0f} W" if _pwr1 is not None else "-"),
            ("후반부 평균 파워", f"{_pwr2:.0f} W" if _pwr2 is not None else "-"),
            ("평균 파워",        f"{row['avg_power']:.0f} W"  if row.get("avg_power")  else "-"),
            ("최대 파워",        f"{row['max_power']:.0f} W"  if row.get("max_power")  else "-"),
            ("W/bpm",            f"{row['w_per_bpm']:.3f}"     if row.get("w_per_bpm")  else "-"),
            ("W/bpm 변화",       _wbpm_change_str),
            ("스파이크",         f"{row['spike_count']}개 보정" if row.get("spike_count") else "없음"),
            ("평균 케이던스",    f"{row['avg_cadence']:.0f} rpm" if row.get("avg_cadence") else "-"),
        ]

    lw = 38  # 레이블 너비
    vw = pdf.epw - lw  # 값 너비
    for label, val in fields:
        fam2, _, _ = fnt(8, True);  pdf.set_font(fam2, "B", 8)
        pdf.cell(lw, 5, label + ":", border=0, ln=0)
        fam3, _, _ = fnt(8);        pdf.set_font(fam3, "", 8)
        pdf.cell(vw, 5, str(val)[:60], border=0, ln=1)
    pdf.ln(3)

    # 심박 존 분포 차트
    z_vals = [row.get(f"z{j}_pct") or 0 for j in range(1, 6)]
    if any(v > 0 for v in z_vals):
        fig, ax = plt.subplots(figsize=(7, 2))
        bars = ax.bar(["Z1","Z2","Z3","Z4","Z5"], z_vals,
                      color=["#B5D4F4","#9FE1CB","#FAC775","#F0997B","#E24B4A"])
        for bar, v in zip(bars, z_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title("심박 존 분포", fontsize=9); ax.set_ylabel("%")
        ax.set_ylim(0, max(z_vals) * 1.3 + 5)
        fig.tight_layout(); _add_chart(fig, 48)

    # 케이던스 구간 분포
    cad_vals = [row.get(k) or 0 for k in ["cad_lt70","cad_70_80","cad_80_90","cad_90_100","cad_100p"]]
    if any(v > 0 for v in cad_vals):
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.bar(["<70","70-79","80-89","90-99","100+"], cad_vals, color="#4A90D9")
        for i2, v in enumerate(cad_vals):
            if v > 0:
                ax.text(i2, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.set_title("케이던스 구간 분포 (rpm)", fontsize=9); ax.set_ylabel("%")
        fig.tight_layout(); _add_chart(fig, 48)

    # 분당 심박 + 파워/페이스 차트
    if not raw_df.empty and "secs" in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["min"] = (raw_df["secs"] // 60).astype(int)
        if is_running and {"hr","pace"}.issubset(raw_df.columns):
            mdf2 = raw_df.groupby("min").agg(hr=("hr","mean"), pace=("pace","median")).reset_index()
            fig, ax1 = plt.subplots(figsize=(7, 2.8))
            ax1.plot(mdf2["min"], mdf2["hr"],   color="#E24B4A", linewidth=1.5, label="심박(bpm)")
            ax2 = ax1.twinx()
            ax2.plot(mdf2["min"], mdf2["pace"], color="#4A90D9", linewidth=1.5, label="페이스(초/km)")
            ax2.invert_yaxis()
            ax1.set_xlabel("경과(분)"); ax1.set_ylabel("심박(bpm)"); ax2.set_ylabel("페이스(초/km)")
            h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc="upper right", fontsize=7)
            ax1.set_title("분당 심박 & 페이스", fontsize=9)
            fig.tight_layout(); _add_chart(fig, 60)
        elif not is_running and {"hr","watts"}.issubset(raw_df.columns):
            mdf2 = raw_df.groupby("min").agg(hr=("hr","mean"), watts=("watts","mean")).reset_index()
            fig, ax1 = plt.subplots(figsize=(7, 2.8))
            ax1.plot(mdf2["min"], mdf2["hr"],    color="#E24B4A", linewidth=1.5, label="심박(bpm)")
            ax2 = ax1.twinx()
            ax2.plot(mdf2["min"], mdf2["watts"], color="#4A90D9", linewidth=1.5, label="파워(W)")
            ax1.set_xlabel("경과(분)"); ax1.set_ylabel("심박(bpm)"); ax2.set_ylabel("파워(W)")
            h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc="upper right", fontsize=7)
            ax1.set_title("분당 심박 & 파워", fontsize=9)
            fig.tight_layout(); _add_chart(fig, 60)

    # ── Page 2: 누적 추이 + 코칭 피드백 + 랩 ───────────────────────────────
    pdf.add_page()

    if not is_running and len(hist) >= 2:
        wbpm_h = hist["w_per_bpm"].dropna()
        if len(wbpm_h) >= 2:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            ax.plot(range(len(wbpm_h)), wbpm_h.values, "o-", color="#4A90D9", linewidth=2)
            cur = row.get("w_per_bpm")
            if cur:
                ax.axhline(cur, color="#E24B4A", linestyle="--", linewidth=1, label="이번")
                ax.legend(fontsize=7)
            ax.set_title("최근 W/bpm 효율 추이", fontsize=9)
            ax.set_xlabel("세션(오래된→최근)"); ax.set_ylabel("W/bpm")
            fig.tight_layout(); _add_chart(fig, 55)

    drift_h = hist["drift"].dropna() if len(hist) >= 2 else pd.Series(dtype=float)
    if len(drift_h) >= 2:
        colors_d = ["#9FE1CB" if abs(v) <= 4 else "#FAC775" if abs(v) <= 8 else "#E24B4A" for v in drift_h.values]
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.bar(range(len(drift_h)), drift_h.values, color=colors_d)
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.set_title("최근 HR 드리프트 추이 (초록=정상 노랑=주의 빨강=피로)", fontsize=8)
        ax.set_xlabel("세션(오래된→최근)"); ax.set_ylabel("드리프트(%)")
        fig.tight_layout(); _add_chart(fig, 55)

    fam4, _, _ = fnt(10, True); pdf.set_font(fam4, "B", 10)
    pdf.cell(pdf.epw, 7, "코칭 피드백", ln=1)
    _cf_row = dict(row)
    _cf_row["_prev_wbpm"] = _prev_wbpm
    for level, msg in coaching_feedback(_cf_row):
        icon_m = {"success": "[OK]", "warning": "[!]", "error": "[X]"}.get(level, "[i]")
        fam5, _, _ = fnt(8); pdf.set_font(fam5, "", 8)
        pdf.multi_cell(pdf.epw, 5, f"{icon_m} {msg}", ln=1)

    laps = load_laps(fname)
    if laps and is_running:
        pdf.ln(3)
        fam6, _, _ = fnt(10, True); pdf.set_font(fam6, "B", 10)
        pdf.cell(pdf.epw, 7, "km 랩 분석", ln=1)
        headers = list(laps[0].keys())
        cw2 = pdf.epw / max(len(headers), 1)
        fam7, _, _ = fnt(8, True); pdf.set_font(fam7, "B", 8)
        for h in headers:
            pdf.cell(cw2, 5, str(h), border=1, align="C", ln=0)
        pdf.ln()
        fam8, _, _ = fnt(8); pdf.set_font(fam8, "", 8)
        for lap in laps:
            for h in headers:
                pdf.cell(cw2, 5, str(lap.get(h, "")), border=1, align="C", ln=0)
            pdf.ln()

    # ── Page 3: 분당 시계열 데이터 ──────────────────────────────────────────────
    if not raw_df.empty and "secs" in raw_df.columns:
        pdf.add_page()
        fam_t, sty_t, sz_t = fnt(11, True)
        pdf.set_font(fam_t, sty_t, sz_t)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(pdf.epw, 8, "분당 시계열 데이터", ln=1)
        pdf.ln(1)

        rdf = raw_df.copy()
        rdf["min"] = (rdf["secs"] // 60).astype(int)

        # detect moving minutes (kph > 0.5 km/h, or watts > 0 for indoor cycling)
        if "kph" in rdf.columns:
            rdf["_moving"] = pd.to_numeric(rdf["kph"], errors="coerce").fillna(0) > 0.5
        elif "watts" in rdf.columns:
            rdf["_moving"] = pd.to_numeric(rdf["watts"], errors="coerce").fillna(0) > 0
        else:
            rdf["_moving"] = True

        agg_dict = {}
        if "hr"    in rdf.columns: agg_dict["hr"]    = ("hr",    "mean")
        if "watts" in rdf.columns: agg_dict["watts"] = ("watts", "mean")
        if "pace"  in rdf.columns: agg_dict["pace"]  = ("pace",  "median")
        if "cad"   in rdf.columns: agg_dict["cad"]   = ("cad",   "mean")
        if "kph"   in rdf.columns: agg_dict["spd"]   = ("kph",   "mean")
        agg_dict["moving"] = ("_moving", "mean")

        mdf3 = rdf.groupby("min").agg(**agg_dict).reset_index()

        # build rows list
        tbl_rows = []
        for _, mr in mdf3.iterrows():
            m_val   = int(mr["min"])
            moving  = mr.get("moving", 1.0) > 0.3
            hr_val  = f"{mr['hr']:.0f}" if "hr" in mr.index and not pd.isna(mr.get("hr", float('nan'))) and moving else "-"
            if is_running:
                p_val = fmt_pace(mr.get("pace")) if "pace" in mr.index and not pd.isna(mr.get("pace", float('nan'))) and moving else "-"
            else:
                p_val = f"{mr['watts']:.0f}" if "watts" in mr.index and not pd.isna(mr.get("watts", float('nan'))) and moving else "-"
            cad_val = f"{mr['cad']:.0f}" if "cad"  in mr.index and not pd.isna(mr.get("cad", float('nan'))) and moving else "-"
            spd_val = f"{mr['spd']:.1f}" if "spd"  in mr.index and not pd.isna(mr.get("spd", float('nan'))) and moving else "-"
            tbl_rows.append((str(m_val), hr_val, p_val, cad_val, spd_val))

        p_hdr = "페이스(분:초/km)" if is_running else "파워(W)"
        hdrs  = ["경과(분)", "심박(bpm)", p_hdr, "케이던스(rpm)", "속도(km/h)"]

        use_two_col = len(tbl_rows) > 20
        if use_two_col:
            half   = (len(tbl_rows) + 1) // 2
            col1   = tbl_rows[:half]
            col2   = tbl_rows[half:]
            cw_set = [10, 14, 22, 18, 14]  # widths per sub-col (sum=78)
            gap    = 4                       # gap between the two columns
        else:
            col1   = tbl_rows
            col2   = []
            cw_set = [14, 18, 30, 24, 20]  # single column widths (sum≈106)
            gap    = 0

        def _tbl_header(x_start):
            fam_h, _, _ = fnt(7, True); pdf.set_font(fam_h, "B", 7)
            pdf.set_fill_color(230, 235, 245); pdf.set_text_color(40, 40, 40)
            cur_x = x_start
            for hd, cw in zip(hdrs, cw_set):
                pdf.set_xy(cur_x, pdf.get_y())
                pdf.cell(cw, 5, hd, border=1, align="C", fill=True, ln=0)
                cur_x += cw

        def _tbl_row(x_start, cells, shade):
            fam_r, _, _ = fnt(7); pdf.set_font(fam_r, "", 7)
            if shade:
                pdf.set_fill_color(248, 249, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(50, 50, 50)
            cur_x = x_start
            for cell, cw in zip(cells, cw_set):
                pdf.set_xy(cur_x, pdf.get_y())
                pdf.cell(cw, 4.5, str(cell), border=1, align="C", fill=True, ln=0)
                cur_x += cw

        if use_two_col:
            x1 = pdf.l_margin
            x2 = x1 + sum(cw_set) + gap
            # header row for both columns
            start_y = pdf.get_y()
            _tbl_header(x1); pdf.set_xy(x2, start_y); _tbl_header(x2)
            pdf.ln(5)
            for i, r1 in enumerate(col1):
                r2      = col2[i] if i < len(col2) else None
                row_y   = pdf.get_y()
                _tbl_row(x1, r1, i % 2 == 0)
                if r2:
                    pdf.set_xy(x2, row_y)
                    _tbl_row(x2, r2, i % 2 == 0)
                pdf.ln(4.5)
        else:
            x1 = pdf.l_margin
            _tbl_header(x1); pdf.ln(5)
            for i, r1 in enumerate(col1):
                row_y = pdf.get_y()
                _tbl_row(x1, r1, i % 2 == 0)
                pdf.ln(4.5)

    pdf.set_y(-15)
    fam9, _, _ = fnt(7); pdf.set_font(fam9, "", 7)
    pdf.cell(pdf.epw, 4, f"Generated by 훈련 분석 앱  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  max_hr={max_hr}  FTP={ftp}W", align="C", ln=1)

    return bytes(pdf.output())


# ── 단일 훈련 상세 뷰 (공용) ──────────────────────────────────────────────────
def show_training_detail(row, df: pd.DataFrame, max_hr: int, ftp: int):
    sport      = str(row.get("sport", "")).lower()
    is_running = "run" in sport
    fname      = str(row.get("filename", ""))
    raw_df     = load_raw(fname)

    # DB 저장값 우선, 없으면 원시 데이터에서 재계산
    pace_drift_sec = row.get("pace_drift_sec")
    best_pace_v    = row.get("best_pace")
    if is_running and not raw_df.empty and "pace" in raw_df.columns:
        pv = raw_df["pace"].dropna()
        if len(pv) > 20:
            if best_pace_v is None:
                best_pace_v = float(pv.quantile(0.05))
            if pace_drift_sec is None:
                mid = len(pv) // 2
                p1, p2 = pv.iloc[:mid].mean(), pv.iloc[mid:].mean()
                if not (pd.isna(p1) or pd.isna(p2)):
                    pace_drift_sec = round(p2 - p1, 1)

    icon = "🏃" if is_running else "🚴"

    # ── PDF 버튼 ──────────────────────────────────────────────────────────────
    pdf_key  = f"pdf_cache_{fname}"
    hdr1, hdr2 = st.columns([7, 1])
    hdr1.markdown(
        f"**{row.get('date','')}** &nbsp;|&nbsp; {row.get('sport','')} &nbsp;|&nbsp; "
        f"{'실내' if row.get('indoor') else '실외'}"
        + (f" &nbsp;|&nbsp; {row['course_name']}" if row.get("course_name") else ""),
        unsafe_allow_html=True,
    )
    with hdr2:
        if st.button("📄 PDF 생성", key=f"pdfbtn_{fname}"):
            with st.spinner("PDF 생성 중..."):
                try:
                    st.session_state[pdf_key] = generate_training_pdf(dict(row), df, max_hr, ftp)
                except Exception as e:
                    st.error(f"PDF 생성 실패: {e}")
        if st.session_state.get(pdf_key):
            st.download_button(
                "📥 다운로드",
                data=st.session_state[pdf_key],
                file_name=f"training_{os.path.basename(fname)}.pdf",
                mime="application/pdf",
                key=f"pdfdl_{fname}",
            )

    # ── 메트릭 행 1: 거리 / 시간 / 속도 / 칼로리 ────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("거리",      f"{row['distance_km']:.2f} km"    if row.get("distance_km")   else "-")
    c2.metric("이동 시간", fmt_duration(row.get("moving_time_sec") or row.get("duration_sec")))
    c3.metric("정지 시간", fmt_duration(row.get("stop_time_sec")) if row.get("stop_time_sec") else "-")
    c4.metric("평균 속도", f"{row['avg_speed_kph']:.1f} km/h" if row.get("avg_speed_kph") else "-")

    # ── 메트릭 행 2: 심박 ────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    drift_val   = row.get("drift")
    drift_str   = f"{drift_val:+.1f}%" if drift_val is not None and not pd.isna(drift_val) else "-"
    drift_grade = row.get("drift_grade") or _drift_grade(drift_val) or "-"
    c5.metric("평균 심박",  f"{row['avg_hr']:.0f} bpm"  if row.get("avg_hr")  else "-")
    c6.metric("최대 심박",  f"{row['max_hr']:.0f} bpm"  if row.get("max_hr")  else "-")
    c7.metric("HR 드리프트", drift_str, help=f"평가: {drift_grade}")
    c8.metric("소모 칼로리", f"{row['calories']:.0f} kcal" if row.get("calories") else "-")

    # ── 메트릭 행 3: 종목별 ──────────────────────────────────────────────────
    c9, c10, c11, c12 = st.columns(4)
    if is_running:
        c9.metric("평균 페이스",     fmt_pace(row.get("avg_pace")))
        c10.metric("최고 페이스",    fmt_pace(best_pace_v))
        pd_str = f"{pace_drift_sec:+.0f}초/km" if pace_drift_sec is not None else "-"
        c11.metric("페이스 드리프트", pd_str)
        c12.metric("평균 케이던스",  f"{row['avg_cadence']:.0f} spm" if row.get("avg_cadence") else "-")
    else:
        c9.metric("평균 파워",   f"{row['avg_power']:.0f} W"  if row.get("avg_power")  else "-")
        c10.metric("최대 파워",  f"{row['max_power']:.0f} W"  if row.get("max_power")  else "-")
        c11.metric("W/bpm 효율", f"{row['w_per_bpm']:.3f}"    if row.get("w_per_bpm")  else "-")
        c12.metric("평균 케이던스", f"{row['avg_cadence']:.0f} rpm" if row.get("avg_cadence") else "-")

    st.markdown("---")

    # ── 심박 존 분포 + 케이던스 분포 ─────────────────────────────────────────
    col_z, col_cad = st.columns([3, 2])
    with col_z:
        z_vals = [row.get(f"z{i}_pct") or 0 for i in range(1, 6)]
        if any(v > 0 for v in z_vals):
            fig_zone = go.Figure(go.Bar(
                x=ZONE_NAMES, y=z_vals, marker_color=ZONE_COLORS,
                text=[f"{v:.1f}%" for v in z_vals], textposition="outside",
            ))
            fig_zone.update_layout(
                title="심박 존 분포",
                yaxis=dict(title="%", range=[0, max(z_vals) * 1.3 + 5]),
                height=230, margin=dict(t=35, b=10, l=10, r=10), showlegend=False,
            )
            st.plotly_chart(fig_zone, use_container_width=True)

    with col_cad:
        cad_keys   = ["cad_lt70", "cad_70_80", "cad_80_90", "cad_90_100", "cad_100p"]
        cad_labels = ["<70", "70-79", "80-89", "90-99", "100+"]
        cad_vals   = [row.get(k) or 0 for k in cad_keys]
        avg_cad    = row.get("avg_cadence")
        if any(v > 0 for v in cad_vals) and avg_cad:
            unit = "spm" if is_running else "rpm"
            fig_cad = go.Figure(go.Bar(
                x=cad_labels, y=cad_vals, marker_color="#7BC96F",
                text=[f"{v:.1f}%" for v in cad_vals], textposition="outside",
            ))
            fig_cad.update_layout(
                title=f"케이던스 분포 (평균 {avg_cad:.0f} {unit})",
                yaxis=dict(title="%", range=[0, max(cad_vals) * 1.3 + 5]),
                height=230, margin=dict(t=35, b=10, l=10, r=10), showlegend=False,
            )
            st.plotly_chart(fig_cad, use_container_width=True)

    # ── 분당 추이 차트 ────────────────────────────────────────────────────────
    if not raw_df.empty and "secs" in raw_df.columns:
        rdf = raw_df.copy()
        rdf["min"] = (rdf["secs"] // 60).astype(int)
        if is_running and {"hr", "pace"}.issubset(rdf.columns):
            mdf = rdf.groupby("min").agg(hr=("hr","mean"), pace=("pace","median")).reset_index()
            fig_r = make_subplots(specs=[[{"secondary_y": True}]])
            fig_r.add_trace(go.Scatter(x=mdf["min"], y=mdf["hr"],   name="심박(bpm)", line=dict(color="#E24B4A", width=2)), secondary_y=False)
            fig_r.add_trace(go.Scatter(x=mdf["min"], y=mdf["pace"], name="페이스(초/km)", line=dict(color="#4A90D9", width=2)), secondary_y=True)
            fig_r.update_layout(title="분당 심박 & 페이스", height=270, margin=dict(t=35,b=10,l=10,r=10), legend=dict(orientation="h",y=1.12))
            fig_r.update_xaxes(title_text="경과 시간 (분)")
            fig_r.update_yaxes(title_text="심박 (bpm)", secondary_y=False)
            fig_r.update_yaxes(title_text="페이스 (초/km)", secondary_y=True, autorange="reversed")
            st.plotly_chart(fig_r, use_container_width=True)
        elif not is_running and {"hr", "watts"}.issubset(rdf.columns):
            mdf = rdf.groupby("min").agg(hr=("hr","mean"), watts=("watts","mean")).reset_index()
            fig_r = make_subplots(specs=[[{"secondary_y": True}]])
            fig_r.add_trace(go.Scatter(x=mdf["min"], y=mdf["hr"],    name="심박(bpm)", line=dict(color="#E24B4A", width=2)), secondary_y=False)
            fig_r.add_trace(go.Scatter(x=mdf["min"], y=mdf["watts"], name="파워(W)",   line=dict(color="#4A90D9", width=2)), secondary_y=True)
            fig_r.update_layout(title="분당 심박 & 파워", height=270, margin=dict(t=35,b=10,l=10,r=10), legend=dict(orientation="h",y=1.12))
            fig_r.update_xaxes(title_text="경과 시간 (분)")
            fig_r.update_yaxes(title_text="심박 (bpm)", secondary_y=False)
            fig_r.update_yaxes(title_text="파워 (W)", secondary_y=True)
            st.plotly_chart(fig_r, use_container_width=True)

    # km 랩 (러닝)
    if is_running:
        laps = load_laps(fname)
        if laps:
            st.subheader("📊 km 랩 분석")
            st.dataframe(pd.DataFrame(laps), use_container_width=True, hide_index=True)

    # ── 누적 비교 ─────────────────────────────────────────────────────────────
    if not df.empty and len(df) >= 2:
        df_c = df.copy()
        df_c["date"] = pd.to_datetime(df_c["date"])
        _sport_key = "run" if is_running else "cycl"
        hist2 = df_c[df_c["sport"].str.contains(_sport_key, case=False, na=False)].sort_values("date").tail(10)
        if len(hist2) >= 2:
            col_t1, col_t2 = st.columns(2)
            if not is_running:
                wbpm_h = hist2["w_per_bpm"].dropna()
                if len(wbpm_h) >= 2:
                    with col_t1:
                        fig_t = px.line(x=list(range(len(wbpm_h))), y=wbpm_h.values, markers=True,
                                        title="최근 W/bpm 추이", labels={"x":"세션","y":"W/bpm"})
                        fig_t.update_layout(height=180, margin=dict(t=35,b=20,l=20,r=10))
                        st.plotly_chart(fig_t, use_container_width=True)
            else:
                pace_h = hist2["avg_pace"].dropna()
                if len(pace_h) >= 2:
                    with col_t1:
                        fig_t = px.line(x=list(range(len(pace_h))), y=pace_h.values, markers=True,
                                        title="최근 평균 페이스 추이", labels={"x":"세션","y":"초/km"})
                        fig_t.update_yaxes(autorange="reversed")
                        fig_t.update_layout(height=180, margin=dict(t=35,b=20,l=20,r=10))
                        st.plotly_chart(fig_t, use_container_width=True)
            drift_h = hist2["drift"].dropna()
            if len(drift_h) >= 2:
                with col_t2:
                    colors_d = ["#9FE1CB" if abs(v) <= 4 else "#FAC775" if abs(v) <= 8 else "#E24B4A" for v in drift_h.values]
                    fig_d = go.Figure(go.Bar(x=list(range(len(drift_h))), y=drift_h.values, marker_color=colors_d))
                    fig_d.update_layout(title="최근 드리프트 추이", height=180,
                                        margin=dict(t=35,b=20,l=20,r=10), xaxis_title="세션", yaxis_title="%")
                    st.plotly_chart(fig_d, use_container_width=True)

    # ── 코칭 피드백 ───────────────────────────────────────────────────────────
    row_extra = dict(row)
    if pace_drift_sec is not None:
        row_extra["pace_drift_sec"] = pace_drift_sec
    st.markdown("**💬 코칭 피드백**")
    for level, msg in coaching_feedback(row_extra):
        getattr(st, level if level in ("success", "warning", "error") else "info")(msg)


# ── 탭 6: 리포트 ───────────────────────────────────────────────────────────────
def tab_report(df: pd.DataFrame, max_hr: int, ftp: int):
    st.header("훈련 리포트")

    if df.empty:
        st.info("데이터가 없습니다.")
        return

    _ensure_reports_dir()

    # ── 리포트 생성 ─────────────────────────────────────────────────────────
    st.subheader("📄 리포트 생성")

    # 캘린더에서 이동 시 preset 값 소비
    preset_t = st.session_state.pop("report_preset_type",  None)
    preset_l = st.session_state.pop("report_preset_label", None)
    type_opts = ["주간", "월간", "분기", "반기", "연간"]
    if preset_t and preset_t in type_opts:
        st.session_state["rpt_type_sel"] = preset_t

    r1, r2 = st.columns(2)
    period_type = r1.selectbox("리포트 종류", type_opts, key="rpt_type_sel")

    labels = _period_labels(df, period_type)
    if not labels:
        st.info("해당 종류의 기간이 없습니다.")
        return

    if preset_l and preset_l in labels:
        st.session_state["rpt_label_sel"] = preset_l
    if "rpt_label_sel" not in st.session_state or st.session_state["rpt_label_sel"] not in labels:
        st.session_state["rpt_label_sel"] = labels[0]

    period_label = r2.selectbox("기간 선택", labels, key="rpt_label_sel")
    period_start, period_end = _period_range(period_type, period_label)
    st.caption(f"대상 기간: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}")

    if st.button("📄 PDF 생성 및 저장", type="primary"):
        df_p = df.copy()
        df_p["date"] = pd.to_datetime(df_p["date"])
        period_df = df_p[
            (df_p["date"].dt.date >= period_start) &
            (df_p["date"].dt.date <= period_end)
        ].copy()
        if period_df.empty:
            st.warning("해당 기간에 훈련 데이터가 없습니다.")
        else:
            with st.spinner("PDF 생성 중..."):
                try:
                    pdf_bytes = generate_pdf_report(period_df, period_type, period_start, period_end)
                    fpath     = _report_filename(period_type, period_start)
                    with open(fpath, "wb") as f:
                        f.write(pdf_bytes)
                    st.success(f"저장 완료: `{fpath}`")
                    st.download_button(
                        "📥 바로 다운로드",
                        data=pdf_bytes,
                        file_name=os.path.basename(fpath),
                        mime="application/pdf",
                        key="dl_new",
                    )
                except Exception as e:
                    st.error(f"생성 실패: {e}")

    # ── 과거 리포트 목록 ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📂 저장된 리포트")
    if os.path.isdir(REPORTS_DIR):
        files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")], reverse=True)
        if files:
            hc1, hc2, hc3 = st.columns([4, 2, 1])
            hc1.markdown("**파일명**"); hc2.markdown("**크기 · 생성일**"); hc3.markdown("**다운로드**")
            st.markdown("---")
            for fname in files:
                fpath = os.path.join(REPORTS_DIR, fname)
                sz    = os.path.getsize(fpath) / 1024
                mt    = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
                c1, c2, c3 = st.columns([4, 2, 1])
                c1.write(f"📄 {fname}")
                c2.caption(f"{sz:.0f} KB · {mt}")
                with open(fpath, "rb") as f:
                    c3.download_button("↓", data=f.read(), file_name=fname,
                                       mime="application/pdf", key=f"dl_{fname}")
        else:
            st.info("저장된 리포트가 없습니다. 위에서 생성해주세요.")
    else:
        st.info("저장된 리포트가 없습니다.")


# ── 탭 6: 설정 ─────────────────────────────────────────────────────────────────
def tab_settings(max_hr: int, ftp: int):
    st.header("앱 설정")

    st.subheader("심박 존 기준")
    zones = hr_zones(max_hr)
    zone_names = ["Z1 회복", "Z2 유산소", "Z3 템포", "Z4 역치", "Z5 최대"]
    for name, (lo, hi), color in zip(zone_names, zones, ZONE_COLORS):
        st.markdown(
            f"<span style='background:{color};padding:2px 8px;border-radius:4px;color:white'>{name}</span>  "
            f"**{lo} – {hi} bpm**",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("FTP 기반 파워 존")
    pwr_zones = [
        ("P1 Active Recovery", 0, int(ftp * 0.55)),
        ("P2 Endurance",       int(ftp * 0.55), int(ftp * 0.75)),
        ("P3 Tempo",           int(ftp * 0.75), int(ftp * 0.90)),
        ("P4 Threshold",       int(ftp * 0.90), int(ftp * 1.05)),
        ("P5 VO2max",          int(ftp * 1.05), int(ftp * 1.20)),
        ("P6 Anaerobic",       int(ftp * 1.20), int(ftp * 1.50)),
        ("P7 Neuromuscular",   int(ftp * 1.50), 9999),
    ]
    for name, lo, hi in pwr_zones:
        hi_str = "∞" if hi >= 9999 else str(hi)
        st.markdown(f"- **{name}**: {lo} – {hi_str} W")

    st.markdown("---")
    st.subheader("DB 정보")
    st.caption(f"DB 경로: `{os.path.abspath(DB_PATH)}`")
    with get_conn() as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM training_log").fetchone()[0]
    st.metric("저장된 훈련 수", cnt)

    if st.button("⚠️ 전체 데이터 삭제", type="secondary"):
        confirm = st.checkbox("정말 삭제하시겠습니까?")
        if confirm:
            with get_conn() as conn:
                conn.execute("DELETE FROM training_log")
            st.success("삭제 완료")
            st.rerun()


# ── 메인 ───────────────────────────────────────────────────────────────────────
_NAV = ["최근 훈련", "캘린더", "훈련 기록", "트렌드", "히트맵", "리포트", "설정"]


def main():
    st.set_page_config(
        page_title="훈련 분석",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_db()
    max_hr, ftp = sidebar()
    df = load_all()

    # ── 커스텀 탭 내비게이션 (session_state 기반 → 프로그래밍으로 전환 가능) ──
    if "nav_tab" not in st.session_state:
        st.session_state.nav_tab = _NAV[0]
    if "rec_detail_file"  not in st.session_state: st.session_state.rec_detail_file  = None
    if "rec_filter_date"  not in st.session_state: st.session_state.rec_filter_date  = None

    nav_cols = st.columns(len(_NAV))
    for i, name in enumerate(_NAV):
        active = st.session_state.nav_tab == name
        btn_type = "primary" if active else "secondary"
        if nav_cols[i].button(name, use_container_width=True, type=btn_type, key=f"nav_{name}"):
            st.session_state.nav_tab = name
            # 탭 전환 시 기록 상세 뷰·날짜 필터 초기화
            if name != "훈련 기록":
                st.session_state.rec_detail_file = None
                st.session_state.rec_filter_date = None
            st.rerun()

    st.markdown("---")

    tab = st.session_state.nav_tab
    if   tab == "최근 훈련": tab_recent(df, max_hr, ftp)
    elif tab == "캘린더":    tab_calendar(df)
    elif tab == "훈련 기록": tab_records(df, max_hr, ftp)
    elif tab == "트렌드":    tab_trends(df, ftp)
    elif tab == "히트맵":    tab_heatmap(df)
    elif tab == "리포트":    tab_report(df, max_hr, ftp)
    elif tab == "설정":      tab_settings(max_hr, ftp)


if __name__ == "__main__":
    main()
