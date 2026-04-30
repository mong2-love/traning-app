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
import calendar

ZONE_COLORS = ["#B5D4F4", "#9FE1CB", "#FAC775", "#F0997B", "#E24B4A"]
ZONE_NAMES  = ["Z1 회복", "Z2 유산소", "Z3 템포", "Z4 역치", "Z5 최대"]

DB_PATH = "training_log.db"

# ── DB 초기화 ──────────────────────────────────────────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                date          TEXT,
                filename      TEXT UNIQUE,
                sport         TEXT,
                indoor        INTEGER,
                distance_km   REAL,
                duration_sec  INTEGER,
                avg_hr        REAL,
                max_hr        REAL,
                avg_power     REAL,
                max_power     REAL,
                avg_cadence   REAL,
                w_per_bpm     REAL,
                drift         REAL,
                z1_pct        REAL,
                z2_pct        REAL,
                z3_pct        REAL,
                z4_pct        REAL,
                z5_pct        REAL,
                avg_pace      REAL,
                calories      REAL
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
        try:
            conn.execute("ALTER TABLE training_raw ADD COLUMN laps_json TEXT")
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
                 5: "distance",    6: "speed", 7: "watts", 2: "altitude"}
_FIT_NAME_MAP = {"timestamp": "timestamp", "heart_rate": "hr", "cadence": "cad",
                 "distance": "distance",   "speed": "speed",
                 "power": "watts",         "altitude": "altitude"}


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
            date_str    = valid_ts.iloc[0].strftime("%Y-%m-%d")
            secs        = (ts - valid_ts.iloc[0]).dt.total_seconds()
            df["secs"]  = secs
            secs_diff   = secs.diff().fillna(1.0)
            moving_time = float(secs_diff[secs_diff < 30].sum())

    # 파워 스파이크 보정 — 전후 5초 중앙값의 3배 & 400W 초과 → 중앙값으로 대체
    if "watts" in df.columns:
        df["watts"]  = pd.to_numeric(df["watts"], errors="coerce")
        roll_med     = df["watts"].rolling(window=11, center=True, min_periods=1).median()
        spike_mask   = (df["watts"] > roll_med * 3) & (df["watts"] > 400)
        df.loc[spike_mask, "watts"] = roll_med[spike_mask]

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

    # duration
    duration_sec = moving_time or session_data.get("total_elapsed_time")

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

    calories = session_data.get("total_calories")

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
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(duration_sec)    if duration_sec else None,
        avg_hr=round(avg_hr, 1)           if avg_hr       else None,
        max_hr=round(max_hr_val, 1)       if max_hr_val   else None,
        avg_power=round(avg_power, 1)     if avg_power    else None,
        max_power=round(max_power, 1)     if max_power    else None,
        avg_cadence=round(avg_cad, 1)     if avg_cad      else None,
        w_per_bpm=w_per_bpm,
        drift=drift,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=avg_pace,
        calories=float(calories) if calories else None,
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
            date_str   = valid_ts.iloc[0].strftime("%Y-%m-%d")
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
    avg_pace   = float(pace_valid.mean())              if len(pace_valid) > 0 else None
    best_pace  = float(pace_valid.quantile(0.05))      if len(pace_valid) > 20 else None

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
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(moving_time)     if moving_time  else None,
        avg_hr=round(avg_hr, 1)           if avg_hr       else None,
        max_hr=round(max_hr_val, 1)       if max_hr_val   else None,
        avg_power=None, max_power=None, avg_cadence=None, w_per_bpm=None,
        drift=drift,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=round(avg_pace, 1)       if avg_pace     else None,
        calories=None,
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
    spike_mask = (df["watts"] > rolling_med * 3) & (df["watts"] > 400)
    df.loc[spike_mask, "watts"] = rolling_med[spike_mask]

    # 이동 시간 — 30초 이상 gap은 정지로 판단
    secs_diff = df["secs"].diff().fillna(1.0)
    moving_mask = secs_diff < 30
    moving_time = float(secs_diff[moving_mask].sum())

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
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(moving_time) if moving_time else None,
        avg_hr=round(avg_hr, 1) if avg_hr else None,
        max_hr=round(max_hr_val, 1) if max_hr_val else None,
        avg_power=round(avg_power, 1) if avg_power else None,
        max_power=round(max_power, 1) if max_power else None,
        avg_cadence=round(avg_cad, 1) if avg_cad else None,
        w_per_bpm=w_per_bpm,
        drift=drift,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=avg_pace,
        calories=None,
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
    w_bpm        = row.get("w_per_bpm")
    pace_drift   = row.get("pace_drift_sec")  # 러닝 전용, tab_today에서 주입

    if is_running:
        # 페이스 드리프트
        if pace_drift is not None:
            if abs(pace_drift) <= 10:
                msgs.append(("success", f"페이스 드리프트 {pace_drift:+.0f}초/km — 매우 안정적인 페이스를 유지했습니다. ✅"))
            elif abs(pace_drift) <= 20:
                msgs.append(("info",    f"페이스 드리프트 {pace_drift:+.0f}초/km — 후반에 약간 느려졌습니다. 페이스 배분을 조절해보세요."))
            else:
                msgs.append(("warning", f"페이스 드리프트 {pace_drift:+.0f}초/km — 초반 페이스가 너무 빨랐습니다. 출발 페이스를 낮춰보세요."))
        # Z2 러닝 목표
        if z2 >= 80:
            msgs.append(("success", f"Z2 비율 {z2:.0f}% — Z2 러닝 목표를 달성했습니다. ✅ 유산소 기반이 탄탄합니다."))
        elif z1 + z2 > 65:
            msgs.append(("success", f"유산소 기반 훈련 {z1+z2:.0f}% — 지방 연소 효율 향상에 효과적인 훈련입니다."))
        if z4 + z5 > 30:
            msgs.append(("warning", f"고강도 구간 {z4+z5:.0f}% — 충분한 회복 후 다음 훈련에 임하세요."))
        # HR 드리프트
        if drift is not None and not pd.isna(drift):
            if drift > 8:
                msgs.append(("warning", f"심박 드리프트 {drift:+.1f}% — 탈수 또는 과부하 가능성. 수분 섭취를 확인하세요."))
            elif drift > 4:
                msgs.append(("info", f"심박 드리프트 {drift:+.1f}% — 후반부 심박이 올라갔습니다. 컨디션을 체크하세요."))
    else:
        # 사이클링 / 기타
        if drift is not None and not pd.isna(drift):
            if drift > 8:
                msgs.append(("warning", f"심박 드리프트 {drift:+.1f}% — 후반에 심박이 크게 올랐습니다. 수분 섭취와 페이스 조절을 확인하세요."))
            elif drift > 4:
                msgs.append(("info",    f"심박 드리프트 {drift:+.1f}% — 약간의 피로 누적이 감지됩니다."))
            elif drift < -4:
                msgs.append(("info",    f"심박 드리프트 {drift:+.1f}% — 후반에 강도가 낮아졌습니다."))
            else:
                msgs.append(("success", f"심박 드리프트 {drift:+.1f}% — 안정적인 페이스를 유지했습니다."))

        if z4 + z5 > 40:
            msgs.append(("warning", f"고강도 구간 {z4+z5:.0f}% — 인터벌 효과가 높았습니다. 다음 세션 전 충분히 회복하세요."))
        elif z1 + z2 > 65:
            msgs.append(("success", f"유산소 기반 훈련 {z1+z2:.0f}% — 지방 연소 및 기초 체력 향상에 효과적이었습니다."))
        elif z3 > 30:
            msgs.append(("info",    f"템포 구간 {z3:.0f}% — 젖산 역치 개선에 효과적인 훈련이었습니다."))

        if w_bpm and not pd.isna(w_bpm):
            if w_bpm > 2.0:
                msgs.append(("success", f"W/bpm 효율 {w_bpm:.2f} — 심박 대비 출력이 우수합니다."))
            elif w_bpm < 1.2:
                msgs.append(("info",    f"W/bpm 효율 {w_bpm:.2f} — 효율 향상을 위해 저강도 지구력 훈련을 늘려보세요."))

    if not msgs:
        msgs.append(("info", "훈련 데이터가 충분히 쌓이면 더 자세한 피드백을 제공합니다."))
    return msgs


# ── 탭 1: 오늘의 훈련 ──────────────────────────────────────────────────────────
def tab_today(df: pd.DataFrame, max_hr: int, ftp: int):
    st.header("오늘의 훈련")
    today_str = str(date.today())
    today_df  = df[df["date"] == today_str] if not df.empty else pd.DataFrame()

    if today_df.empty:
        st.info("오늘 등록된 훈련 데이터가 없습니다. 사이드바에서 파일을 업로드하세요.")
        return

    for _, row in today_df.iterrows():
        sport      = str(row.get("sport", "")).lower()
        is_running = "running" in sport or "run" in sport
        raw_df     = load_raw(row["filename"])

        # 러닝 전용: 원시 데이터에서 페이스 드리프트·최고페이스 계산
        pace_drift_sec = None
        best_pace      = None
        if is_running and not raw_df.empty and "pace" in raw_df.columns:
            pv = raw_df["pace"].dropna()
            if len(pv) > 20:
                best_pace = float(pv.quantile(0.05))
                mid = len(pv) // 2
                p1, p2 = pv.iloc[:mid].mean(), pv.iloc[mid:].mean()
                if not (pd.isna(p1) or pd.isna(p2)):
                    pace_drift_sec = round(p2 - p1, 1)

        icon = "🏃" if is_running else "🚴"
        with st.expander(f"{icon} {row['filename']} — {row['sport']}", expanded=True):

            # ── 8개 메트릭 카드 (2×4) ─────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("거리",      f"{row['distance_km']:.2f} km" if row["distance_km"] else "-")
            c2.metric("이동 시간", fmt_duration(row["duration_sec"]))
            c3.metric("평균 심박", f"{row['avg_hr']:.0f} bpm"     if row["avg_hr"]    else "-")
            c4.metric("최대 심박", f"{row['max_hr']:.0f} bpm"     if row["max_hr"]    else "-")

            c5, c6, c7, c8 = st.columns(4)
            drift_val = row["drift"]
            drift_str = f"{drift_val:+.1f}%" if drift_val is not None and not pd.isna(drift_val) else "-"
            if is_running:
                c5.metric("평균 페이스",     fmt_pace(row["avg_pace"]) if row["avg_pace"] else "-")
                c6.metric("최고 페이스",     fmt_pace(best_pace)        if best_pace       else "-")
                pd_str = f"{pace_drift_sec:+.0f}초/km" if pace_drift_sec is not None else "-"
                c7.metric("페이스 드리프트", pd_str)
                c8.metric("HR 드리프트",     drift_str)
            else:
                c5.metric("평균 파워",  f"{row['avg_power']:.0f} W" if row["avg_power"] else "-")
                c6.metric("최대 파워",  f"{row['max_power']:.0f} W" if row["max_power"] else "-")
                c7.metric("W/bpm 효율", f"{row['w_per_bpm']:.3f}"   if row["w_per_bpm"] else "-")
                c8.metric("HR 드리프트", drift_str)

            st.markdown("---")

            # ── 심박 존 분포 바 차트 (공통) ──────────────────────────────────
            z_vals = [row.get(f"z{i}_pct") or 0 for i in range(1, 6)]
            if any(v > 0 for v in z_vals):
                fig_zone = go.Figure(go.Bar(
                    x=ZONE_NAMES, y=z_vals,
                    marker_color=ZONE_COLORS,
                    text=[f"{v:.1f}%" for v in z_vals],
                    textposition="outside",
                ))
                fig_zone.update_layout(
                    title="심박 존 분포",
                    yaxis=dict(title="%", range=[0, max(z_vals) * 1.25 + 5]),
                    height=260, margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
                )
                st.plotly_chart(fig_zone, use_container_width=True)

            if is_running:
                # ── 러닝: HR + 페이스 듀얼 차트 (페이스 Y축 역방향) ──────────
                if not raw_df.empty and "hr" in raw_df.columns and "pace" in raw_df.columns and "secs" in raw_df.columns:
                    raw_df["min"] = (raw_df["secs"] // 60).astype(int)
                    min_df = raw_df.groupby("min").agg(
                        hr=("hr", "mean"), pace=("pace", "median")
                    ).reset_index()

                    fig_run = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_run.add_trace(
                        go.Scatter(x=min_df["min"], y=min_df["hr"],
                                   name="심박 (bpm)", line=dict(color="#E24B4A", width=2)),
                        secondary_y=False,
                    )
                    fig_run.add_trace(
                        go.Scatter(x=min_df["min"], y=min_df["pace"],
                                   name="페이스 (초/km)", line=dict(color="#4A90D9", width=2)),
                        secondary_y=True,
                    )
                    fig_run.update_layout(
                        title="분당 심박 & 페이스",
                        height=280, margin=dict(t=40, b=10, l=10, r=10),
                        legend=dict(orientation="h", y=1.12),
                    )
                    fig_run.update_xaxes(title_text="경과 시간 (분)")
                    fig_run.update_yaxes(title_text="심박 (bpm)", secondary_y=False)
                    fig_run.update_yaxes(title_text="페이스 (초/km)", secondary_y=True, autorange="reversed")
                    st.plotly_chart(fig_run, use_container_width=True)

                # ── km 랩 분석 테이블 ─────────────────────────────────────────
                laps = load_laps(row["filename"])
                if laps:
                    st.subheader("📊 km 랩 분석")
                    st.dataframe(pd.DataFrame(laps), use_container_width=True, hide_index=True)

            else:
                # ── 사이클링: HR + 파워 듀얼 차트 + 케이던스 도넛 ──────────────
                col_charts, col_donut = st.columns([3, 1])
                with col_charts:
                    if not raw_df.empty and "hr" in raw_df.columns and "watts" in raw_df.columns and "secs" in raw_df.columns:
                        raw_df["min"] = (raw_df["secs"] // 60).astype(int)
                        min_df = raw_df.groupby("min").agg(
                            hr=("hr", "mean"), watts=("watts", "mean")
                        ).reset_index()
                        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_dual.add_trace(
                            go.Scatter(x=min_df["min"], y=min_df["hr"],
                                       name="심박 (bpm)", line=dict(color="#E24B4A", width=2)),
                            secondary_y=False,
                        )
                        fig_dual.add_trace(
                            go.Scatter(x=min_df["min"], y=min_df["watts"],
                                       name="파워 (W)", line=dict(color="#4A90D9", width=2)),
                            secondary_y=True,
                        )
                        fig_dual.update_layout(
                            title="분당 심박 & 파워",
                            height=280, margin=dict(t=40, b=10, l=10, r=10),
                            legend=dict(orientation="h", y=1.12),
                        )
                        fig_dual.update_xaxes(title_text="경과 시간 (분)")
                        fig_dual.update_yaxes(title_text="심박 (bpm)", secondary_y=False)
                        fig_dual.update_yaxes(title_text="파워 (W)",   secondary_y=True)
                        st.plotly_chart(fig_dual, use_container_width=True)

                with col_donut:
                    if not raw_df.empty and "cad" in raw_df.columns:
                        cad_s = raw_df[raw_df["cad"] > 0]["cad"].dropna()
                        if len(cad_s) > 0:
                            bins     = [0, 60, 70, 80, 90, 100, 9999]
                            labels   = ["<60", "60-69", "70-79", "80-89", "90-99", "100+"]
                            cad_cut  = pd.cut(cad_s, bins=bins, labels=labels, right=False)
                            cad_cnt  = cad_cut.value_counts().reindex(labels, fill_value=0)
                            cad_cols = ["#B5D4F4", "#9FE1CB", "#FAC775", "#F0997B", "#E24B4A", "#A259FF"]
                            fig_cad  = go.Figure(go.Pie(
                                labels=cad_cnt.index, values=cad_cnt.values,
                                hole=0.5, marker_colors=cad_cols,
                            ))
                            fig_cad.update_layout(
                                title="케이던스 분포", height=280,
                                margin=dict(t=40, b=10, l=5, r=5),
                                legend=dict(orientation="v", x=1.0, font=dict(size=10)),
                            )
                            st.plotly_chart(fig_cad, use_container_width=True)

            # ── 코칭 피드백 ────────────────────────────────────────────────────
            row_extra = dict(row)
            if pace_drift_sec is not None:
                row_extra["pace_drift_sec"] = pace_drift_sec
            st.markdown("**💬 코칭 피드백**")
            for level, msg in coaching_feedback(row_extra):
                if level == "success":
                    st.success(msg)
                elif level == "warning":
                    st.warning(msg)
                elif level == "error":
                    st.error(msg)
                else:
                    st.info(msg)


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
    if "cal_sel"   not in st.session_state: st.session_state.cal_sel   = None

    year  = st.session_state.cal_year
    month = st.session_state.cal_month

    nc1, nc2, nc3 = st.columns([1, 4, 1])
    with nc1:
        if st.button("◀ 이전달", key="cal_prev"):
            if month == 1: st.session_state.cal_year -= 1; st.session_state.cal_month = 12
            else:           st.session_state.cal_month -= 1
            st.session_state.cal_sel = None
            st.rerun()
    nc2.markdown(
        f"<h3 style='text-align:center;margin:4px 0'>{year}년 {month}월</h3>",
        unsafe_allow_html=True,
    )
    with nc3:
        if st.button("다음달 ▶", key="cal_next"):
            if month == 12: st.session_state.cal_year += 1; st.session_state.cal_month = 1
            else:            st.session_state.cal_month += 1
            st.session_state.cal_sel = None
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

        for week in cal:
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
                    if cols[i].button("📋", key=f"cal_{d}", help="상세 분석"):
                        st.session_state.cal_sel = d
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
                    "font-size:0.73em;min-height:90px;line-height:1.9'>"
                    + "<br>".join(parts) + "</div>",
                    unsafe_allow_html=True,
                )

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

        pm, py = (month - 1, year) if month > 1 else (12, year - 1)
        prev_df = df_c[
            (df_c["date"].apply(lambda d: d.year)  == py) &
            (df_c["date"].apply(lambda d: d.month) == pm)
        ]
        cw = mdf["w_per_bpm"].dropna()
        pw = prev_df["w_per_bpm"].dropna() if not prev_df.empty else pd.Series(dtype=float)
        if len(cw) and len(pw):
            chg = (cw.mean() - pw.mean()) / pw.mean() * 100
            st.metric("W/bpm 전월 대비", f"{cw.mean():.3f}", delta=f"{chg:+.1f}%")

    # ── 날짜 상세 분석 ────────────────────────────────────────────────────────
    sel_d = st.session_state.cal_sel
    if sel_d and sel_d in date_recs:
        st.markdown("---")
        dcol, ccol = st.columns([5, 1])
        dcol.subheader(f"📅 {sel_d} 상세 분석")
        if ccol.button("✕ 닫기", key="cal_close"):
            st.session_state.cal_sel = None
            st.rerun()

        for rec in date_recs[sel_d]:
            sport  = str(rec.get("sport", "")).lower()
            is_run = "run" in sport
            icon   = "🏃" if is_run else "🚴"
            fname  = os.path.basename(str(rec.get("filename", "")))
            with st.expander(f"{icon} {fname} — {rec.get('sport','')}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("거리",      f"{rec['distance_km']:.2f} km" if _v(rec.get("distance_km")) else "-")
                c2.metric("시간",      fmt_duration(rec.get("duration_sec")))
                c3.metric("평균 심박", f"{rec['avg_hr']:.0f} bpm"     if _v(rec.get("avg_hr"))       else "-")
                if is_run:
                    c4.metric("평균 페이스", fmt_pace(rec.get("avg_pace")))
                else:
                    c4.metric("평균 파워", f"{rec['avg_power']:.0f} W" if _v(rec.get("avg_power")) else "-")

                z_vals = [rec.get(f"z{i}_pct") or 0 for i in range(1, 6)]
                if any(v > 0 for v in z_vals):
                    fig_z = go.Figure(go.Bar(
                        x=ZONE_NAMES, y=z_vals, marker_color=ZONE_COLORS,
                        text=[f"{v:.1f}%" for v in z_vals], textposition="outside",
                    ))
                    fig_z.update_layout(
                        yaxis=dict(title="%", range=[0, max(z_vals) * 1.3 + 5]),
                        height=200, margin=dict(t=20, b=10, l=10, r=10), showlegend=False,
                    )
                    st.plotly_chart(fig_z, use_container_width=True)

                raw_df = load_raw(str(rec.get("filename", "")))
                if not raw_df.empty and "secs" in raw_df.columns:
                    raw_df = raw_df.copy()
                    raw_df["min"] = (raw_df["secs"] // 60).astype(int)
                    if is_run and {"hr", "pace"}.issubset(raw_df.columns):
                        min_df = raw_df.groupby("min").agg(
                            hr=("hr", "mean"), pace=("pace", "median")
                        ).reset_index()
                        fig_r = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_r.add_trace(go.Scatter(x=min_df["min"], y=min_df["hr"],   name="심박",   line=dict(color="#E24B4A", width=1.5)), secondary_y=False)
                        fig_r.add_trace(go.Scatter(x=min_df["min"], y=min_df["pace"], name="페이스", line=dict(color="#4A90D9", width=1.5)), secondary_y=True)
                        fig_r.update_yaxes(secondary_y=True, autorange="reversed")
                        fig_r.update_layout(height=220, margin=dict(t=10,b=10,l=10,r=10),
                                            legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig_r, use_container_width=True)
                    elif not is_run and {"hr", "watts"}.issubset(raw_df.columns):
                        min_df = raw_df.groupby("min").agg(
                            hr=("hr", "mean"), watts=("watts", "mean")
                        ).reset_index()
                        fig_r = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_r.add_trace(go.Scatter(x=min_df["min"], y=min_df["hr"],    name="심박", line=dict(color="#E24B4A", width=1.5)), secondary_y=False)
                        fig_r.add_trace(go.Scatter(x=min_df["min"], y=min_df["watts"], name="파워", line=dict(color="#4A90D9", width=1.5)), secondary_y=True)
                        fig_r.update_layout(height=220, margin=dict(t=10,b=10,l=10,r=10),
                                            legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig_r, use_container_width=True)

                if is_run:
                    laps = load_laps(str(rec.get("filename", "")))
                    if laps:
                        st.markdown("**km 랩**")
                        st.dataframe(pd.DataFrame(laps), use_container_width=True, hide_index=True)


# ── 탭 3: 훈련 기록 ────────────────────────────────────────────────────────────
def tab_records(df: pd.DataFrame):
    st.header("훈련 기록")

    if df.empty:
        st.info("저장된 훈련 기록이 없습니다.")
        return

    sports = ["전체"] + sorted(df["sport"].dropna().unique().tolist())
    sport_filter = st.selectbox("종목 필터", sports)
    filtered = df if sport_filter == "전체" else df[df["sport"] == sport_filter]

    if filtered.empty:
        st.info("해당 종목의 기록이 없습니다.")
        return

    hc = st.columns([1.2, 1.8, 1, 1.2, 1.2, 1, 0.6])
    for col, label in zip(hc, ["날짜", "파일명", "종목", "거리(km)", "시간", "평균HR", ""]):
        col.markdown(f"**{label}**")
    st.markdown("---")

    for _, row in filtered.iterrows():
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 1.8, 1, 1.2, 1.2, 1, 0.6])
        c1.write(str(row["date"]))
        c2.write(os.path.basename(str(row.get("filename", ""))))
        c3.write(str(row.get("sport", "")))
        c4.write(f"{row['distance_km']:.2f}" if row["distance_km"] else "-")
        c5.write(fmt_duration(row["duration_sec"]))
        c6.write(f"{row['avg_hr']:.0f}" if row["avg_hr"] else "-")
        if c7.button("🗑️", key=f"del_{row['id']}"):
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
    today_ts = pd.Timestamp(date.today())

    # ── 필터 ─────────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([3, 1])
    period = fc1.radio("기간", ["1개월", "3개월", "6개월", "전체"], horizontal=True)
    sports_avail = ["전체"] + sorted(df["sport"].dropna().unique().tolist())
    sport_sel    = fc2.selectbox("종목", sports_avail)

    cutoff = {"1개월": 30, "3개월": 90, "6개월": 180}.get(period)
    pdf = df.copy()
    if cutoff:
        pdf = pdf[pdf["date"] >= today_ts - timedelta(days=cutoff)]
    if sport_sel != "전체":
        pdf = pdf[pdf["sport"] == sport_sel]

    if pdf.empty:
        st.info("해당 조건에 데이터가 없습니다.")
        return

    # ── 훈련 부하 PMC (CTL / ATL / TSB) ──────────────────────────────────────
    st.subheader("📈 훈련 부하 — 피트니스 / 피로 / 컨디션")

    def calc_trimp(row):
        try:
            hr  = float(row["avg_hr"])
            dur = float(row["duration_sec"])
            mhr = float(row["max_hr"]) if pd.notna(row.get("max_hr")) else 185.0
            if hr <= 0 or dur <= 0:
                return 0.0
            ratio = max(0.0, (hr - 50) / max(1.0, mhr - 50))
            return (dur / 60) * ratio * math.exp(1.92 * ratio)
        except Exception:
            return 0.0

    df["trimp"] = df.apply(calc_trimp, axis=1)

    all_dates   = pd.date_range(df["date"].min(), today_ts, freq="D")
    daily_base  = pd.DataFrame({"date": all_dates})
    daily_trimp = df.groupby(df["date"].dt.normalize())["trimp"].sum().reset_index()
    daily_trimp.columns = ["date", "trimp"]
    daily_load  = daily_base.merge(daily_trimp, on="date", how="left").fillna({"trimp": 0.0})
    daily_load  = daily_load.sort_values("date").reset_index(drop=True)

    EXP_CTL = math.exp(-1 / 42)
    EXP_ATL = math.exp(-1 / 7)
    K_CTL   = 1 - EXP_CTL
    K_ATL   = 1 - EXP_ATL
    ctls, atls, ctl_v, atl_v = [], [], 0.0, 0.0
    for t in daily_load["trimp"]:
        ctl_v = ctl_v * EXP_CTL + t * K_CTL
        atl_v = atl_v * EXP_ATL + t * K_ATL
        ctls.append(round(ctl_v, 2))
        atls.append(round(atl_v, 2))

    daily_load["CTL"] = ctls
    daily_load["ATL"] = atls
    daily_load["TSB"] = daily_load["CTL"] - daily_load["ATL"]
    plot_load = daily_load[daily_load["date"] >= today_ts - timedelta(days=cutoff)] if cutoff else daily_load

    fig_pmc = go.Figure()
    fig_pmc.add_trace(go.Scatter(
        x=plot_load["date"], y=plot_load["CTL"],
        name="CTL 피트니스", line=dict(color="#4A90D9", width=2.5),
    ))
    fig_pmc.add_trace(go.Scatter(
        x=plot_load["date"], y=plot_load["ATL"],
        name="ATL 피로", line=dict(color="#E24B4A", width=2.5),
    ))
    fig_pmc.add_trace(go.Scatter(
        x=plot_load["date"], y=plot_load["TSB"],
        name="TSB 컨디션", line=dict(color="#27AE60", width=2),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.1)",
    ))
    fig_pmc.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig_pmc.update_layout(
        height=340, margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.12),
        yaxis_title="TRIMP 부하 점수",
    )
    st.plotly_chart(fig_pmc, use_container_width=True)
    st.caption(
        "CTL(파란): 42일 누적 장기 피트니스  ·  "
        "ATL(빨간): 7일 누적 단기 피로  ·  "
        "TSB(초록): CTL − ATL  →  양수 = 컨디션 좋음 / 음수 = 피로 누적"
    )

    st.markdown("---")

    # ── 지표 트렌드 ──────────────────────────────────────────────────────────
    st.subheader("📉 지표 트렌드")
    metric_opts = {
        "avg_hr":      "평균 심박수 (bpm)",
        "avg_power":   "평균 파워 (W)",
        "distance_km": "거리 (km)",
        "w_per_bpm":   "W/bpm 효율",
        "drift":       "HR 드리프트 (%)",
        "avg_pace":    "평균 페이스 (초/km)",
        "calories":    "칼로리",
    }
    metric = st.selectbox("지표 선택", list(metric_opts.keys()),
                          format_func=lambda x: metric_opts[x])
    plot_m = pdf.dropna(subset=[metric])
    if not plot_m.empty:
        fig_m = px.scatter(
            plot_m, x="date", y=metric, color="sport",
            trendline="lowess",
            labels={"date": "날짜", metric: metric_opts[metric]},
            color_discrete_sequence=["#4A90D9", "#E24B4A", "#FAC775"],
        )
        fig_m.update_traces(marker=dict(size=7))
        fig_m.update_layout(
            height=300, margin=dict(t=20, b=10),
            legend=dict(y=1.1, orientation="h"),
        )
        if metric == "avg_pace":
            fig_m.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("해당 기간·종목에 데이터가 없습니다.")

    st.markdown("---")

    # ── 주간 볼륨 ────────────────────────────────────────────────────────────
    st.subheader("📊 주간 훈련 볼륨")
    weekly = pdf.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    weekly_grp = weekly.groupby(["week", "sport"])["distance_km"].sum().reset_index()
    if not weekly_grp.empty:
        fig_w = px.bar(
            weekly_grp, x="week", y="distance_km", color="sport", barmode="stack",
            labels={"week": "주", "distance_km": "거리 (km)", "sport": "종목"},
            color_discrete_sequence=["#4A90D9", "#E24B4A", "#FAC775"],
        )
        fig_w.update_layout(height=260, margin=dict(t=20, b=10),
                            legend=dict(y=1.1, orientation="h"))
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")

    # ── 심박 존 분포 트렌드 (월별 누적 스택 바) ──────────────────────────────
    st.subheader("🎯 심박 존 분포 트렌드 (월별)")
    zone_df = pdf.copy()
    zone_df["month"] = zone_df["date"].dt.to_period("M").astype(str)
    zone_cols    = ["z1_pct", "z2_pct", "z3_pct", "z4_pct", "z5_pct"]
    zone_monthly = zone_df.groupby("month")[zone_cols].mean().reset_index()
    if not zone_monthly.empty and zone_monthly[zone_cols].sum().sum() > 0:
        zone_long = zone_monthly.melt(
            id_vars="month", value_vars=zone_cols,
            var_name="zone", value_name="pct",
        )
        zone_long["zone"] = zone_long["zone"].map(
            {z: n for z, n in zip(zone_cols, ZONE_NAMES)}
        )
        fig_zt = px.bar(
            zone_long, x="month", y="pct", color="zone", barmode="stack",
            color_discrete_map={n: c for n, c in zip(ZONE_NAMES, ZONE_COLORS)},
            labels={"month": "월", "pct": "비율 (%)", "zone": "존"},
        )
        fig_zt.update_layout(height=260, margin=dict(t=20, b=10),
                             legend=dict(y=1.15, orientation="h"))
        st.plotly_chart(fig_zt, use_container_width=True)

    st.markdown("---")

    # ── 종목 비율 + 개인 기록 ────────────────────────────────────────────────
    pie_col, pr_col = st.columns([1, 2])

    with pie_col:
        st.subheader("🥧 종목 비율")
        sport_cnt = df["sport"].value_counts().reset_index()
        sport_cnt.columns = ["sport", "count"]
        fig_pie = px.pie(sport_cnt, names="sport", values="count", hole=0.4,
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
                wbpm_c = cdf["w_per_bpm"].dropna()
                pwr_c  = cdf["avg_power"].dropna()
                dist_c = cdf["distance_km"].dropna()
                if len(wbpm_c): st.metric("최고 W/bpm",    f"{wbpm_c.max():.3f}")
                if len(pwr_c):  st.metric("최고 평균 파워", f"{pwr_c.max():.0f} W")
                if len(dist_c): st.metric("최장 라이드",    f"{dist_c.max():.1f} km")
            else:
                st.info("기록 없음")
        with pr_r:
            st.markdown("**🏃 러닝**")
            rdf = df[df["sport"].str.lower().str.contains("run", na=False)]
            if not rdf.empty:
                pace_r = rdf["avg_pace"].dropna()
                dist_r = rdf["distance_km"].dropna()
                hr_r   = rdf["avg_hr"].dropna()
                if len(pace_r): st.metric("최고 평균 페이스", fmt_pace(pace_r.min()))
                if len(dist_r): st.metric("최장 런",          f"{dist_r.max():.1f} km")
                if len(hr_r):   st.metric("최저 평균 심박",   f"{hr_r.min():.0f} bpm")
            else:
                st.info("기록 없음")


# ── 탭 5: 리포트 ───────────────────────────────────────────────────────────────
def tab_report(df: pd.DataFrame):
    st.header("훈련 리포트 생성")

    if df.empty:
        st.info("데이터가 없습니다.")
        return

    try:
        from fpdf import FPDF
    except ImportError:
        st.error("fpdf2 패키지가 필요합니다.")
        return

    today = date.today()
    date_str_series = df["date"].astype(str)
    months_available = sorted(
        date_str_series.str[:7].unique().tolist(), reverse=True
    )
    sel_month = st.selectbox("리포트 기간 (월)", months_available)

    if st.button("PDF 리포트 생성"):
        m_df = df[df["date"].astype(str).str.startswith(sel_month)]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Training Report: {sel_month}", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Sessions: {len(m_df)}", ln=True)
        pdf.cell(0, 8, f"Total Distance: {m_df['distance_km'].sum():.1f} km", ln=True)
        pdf.cell(0, 8, f"Total Time: {fmt_duration(m_df['duration_sec'].sum())}", ln=True)
        avg_hr_mean = m_df["avg_hr"].mean()
        if not pd.isna(avg_hr_mean):
            pdf.cell(0, 8, f"Avg Heart Rate: {avg_hr_mean:.0f} bpm", ln=True)
        avg_pwr_mean = m_df["avg_power"].mean()
        if not pd.isna(avg_pwr_mean):
            pdf.cell(0, 8, f"Avg Power: {avg_pwr_mean:.0f} W", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 12)
        headers = ["Date", "Sport", "Dist(km)", "Time", "AvgHR", "AvgW", "Cal"]
        col_w = [28, 28, 25, 25, 22, 22, 22]
        for h, w in zip(headers, col_w):
            pdf.cell(w, 8, h, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        for _, row in m_df.iterrows():
            vals = [
                str(row["date"]),
                str(row["sport"] or ""),
                f"{row['distance_km']:.1f}" if row["distance_km"] else "-",
                fmt_duration(row["duration_sec"]),
                f"{row['avg_hr']:.0f}" if row["avg_hr"] else "-",
                f"{row['avg_power']:.0f}" if row["avg_power"] else "-",
                f"{row['calories']:.0f}" if row["calories"] else "-",
            ]
            for v, w in zip(vals, col_w):
                pdf.cell(w, 7, v, border=1, align="C")
            pdf.ln()

        pdf_bytes = bytes(pdf.output())
        st.download_button(
            label="📥 PDF 다운로드",
            data=pdf_bytes,
            file_name=f"training_report_{sel_month}.pdf",
            mime="application/pdf",
        )
        st.success("리포트가 생성되었습니다!")


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

    tabs = st.tabs(["오늘의 훈련", "캘린더", "훈련 기록", "트렌드", "리포트", "설정"])

    with tabs[0]:
        tab_today(df, max_hr, ftp)
    with tabs[1]:
        tab_calendar(df)
    with tabs[2]:
        tab_records(df)
    with tabs[3]:
        tab_trends(df, ftp)
    with tabs[4]:
        tab_report(df)
    with tabs[5]:
        tab_settings(max_hr, ftp)


if __name__ == "__main__":
    main()
