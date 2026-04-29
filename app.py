import streamlit as st
import sqlite3
import os
import tempfile
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
                filename TEXT PRIMARY KEY,
                raw_json TEXT
            )
        """)


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


# ── GPX 파싱 ────────────────────────────────────────────────────────────────────
def parse_gpx(path, max_hr, ftp):
    try:
        import gpxpy as gpx_lib
    except ImportError:
        st.error("gpxpy 패키지가 설치되지 않았습니다.")
        return None

    try:
        with open(path) as f:
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
                            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                            try:
                                row[tag] = float(child.text)
                            except Exception:
                                pass
                points.append(row)

    if not points:
        return None

    df = pd.DataFrame(points)
    sport = (gpx.tracks[0].type or "unknown").lower() if gpx.tracks else "unknown"

    duration_sec = None
    date_str = str(date.today())
    if "time" in df.columns and df["time"].notna().any():
        times = pd.to_datetime(df["time"].dropna())
        duration_sec = (times.iloc[-1] - times.iloc[0]).total_seconds()
        date_str = times.iloc[0].strftime("%Y-%m-%d")

    distance_km = gpx.length_3d() / 1000.0 if gpx.length_3d() else 0

    hr_col = next((c for c in ["hr", "heartrate", "heart_rate"] if c in df.columns), None)
    avg_hr = float(df[hr_col].mean()) if hr_col else None
    max_hr_val = float(df[hr_col].max()) if hr_col else None

    drift = None
    if hr_col and len(df) > 10:
        mid = len(df) // 2
        h1 = df[hr_col].iloc[:mid].mean()
        h2 = df[hr_col].iloc[mid:].mean()
        drift = round((h2 - h1) / h1 * 100, 1) if h1 else None

    zones = hr_zones(max_hr)
    z_pcts = [0.0] * 5
    if hr_col:
        hr_series = df[hr_col].dropna()
        total = len(hr_series)
        if total > 0:
            for i, (lo, hi) in enumerate(zones):
                z_pcts[i] = round(((hr_series >= lo) & (hr_series <= hi)).sum() / total * 100, 1)

    avg_pace = (duration_sec / distance_km) if distance_km and duration_sec and distance_km > 0 else None

    return dict(
        date=date_str,
        filename=os.path.basename(path),
        sport=sport,
        indoor=0,
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(duration_sec) if duration_sec else None,
        avg_hr=round(avg_hr, 1) if avg_hr else None,
        max_hr=round(max_hr_val, 1) if max_hr_val else None,
        avg_power=None, max_power=None, avg_cadence=None, w_per_bpm=None,
        drift=drift,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=round(avg_pace, 1) if avg_pace else None,
        calories=None,
    )


# ── CSV 원시 데이터 저장/로드 ───────────────────────────────────────────────────
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
        return pd.read_json(row[0], orient="records")
    return pd.DataFrame()


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

    # ── 폴더 선택기 ────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("감시 폴더")

    if "watch_dir" not in st.session_state:
        st.session_state["watch_dir"] = os.path.expanduser("~/Downloads")

    # 바로가기 버튼
    quick = {"🖥️ 바탕화면": "~/Desktop", "📥 다운로드": "~/Downloads", "📄 문서": "~/Documents"}
    q_cols = st.sidebar.columns(3)
    for i, (label, path) in enumerate(quick.items()):
        exp = os.path.expanduser(path)
        if os.path.isdir(exp):
            if q_cols[i].button(label, use_container_width=True):
                st.session_state["watch_dir"] = exp
                st.rerun()

    # 텍스트 직접 입력
    typed = st.sidebar.text_input("경로 직접 입력", value=st.session_state["watch_dir"])
    if typed != st.session_state["watch_dir"]:
        st.session_state["watch_dir"] = typed

    watch_dir = st.session_state["watch_dir"]

    # 하위 폴더 탐색
    if os.path.isdir(watch_dir):
        try:
            subdirs = sorted([
                d for d in os.listdir(watch_dir)
                if os.path.isdir(os.path.join(watch_dir, d)) and not d.startswith(".")
            ])
            if subdirs:
                sel_sub = st.sidebar.selectbox("📁 하위 폴더", ["(현재 폴더)"] + subdirs)
                if sel_sub != "(현재 폴더)":
                    new_path = os.path.join(watch_dir, sel_sub)
                    if st.sidebar.button("이 폴더 선택", use_container_width=True):
                        st.session_state["watch_dir"] = new_path
                        watch_dir = new_path
                        st.rerun()
        except PermissionError:
            pass

        # 훈련 파일 개수 표시
        try:
            found = [f for f in os.listdir(watch_dir) if f.lower().endswith((".fit", ".gpx", ".csv"))]
            st.sidebar.caption(f"훈련 파일 {len(found)}개 감지됨")
        except PermissionError:
            pass
    else:
        st.sidebar.warning("폴더를 찾을 수 없습니다.")

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

    st.sidebar.markdown("---")
    if st.sidebar.button("폴더 스캔"):
        if os.path.isdir(watch_dir):
            count = 0
            for fname in os.listdir(watch_dir):
                if fname.lower().endswith((".fit", ".gpx", ".csv")):
                    fpath = os.path.join(watch_dir, fname)
                    data = parse_file(fpath, max_hr, ftp)
                    if data:
                        save_record(data)
                        count += 1
            st.sidebar.success(f"{count}개 파일 처리 완료")
        else:
            st.sidebar.error("폴더를 찾을 수 없습니다.")

    return max_hr, ftp, watch_dir


# ── 코칭 피드백 ─────────────────────────────────────────────────────────────────
def coaching_feedback(row: dict) -> list:
    msgs = []
    drift = row.get("drift")
    z1  = row.get("z1_pct") or 0
    z2  = row.get("z2_pct") or 0
    z3  = row.get("z3_pct") or 0
    z4  = row.get("z4_pct") or 0
    z5  = row.get("z5_pct") or 0
    w_bpm = row.get("w_per_bpm")

    if drift is not None and not pd.isna(drift):
        if drift > 8:
            msgs.append(("warning", f"심박 드리프트 {drift:+.1f}% — 후반에 심박이 크게 올랐습니다. 수분 섭취와 페이스 조절을 확인하세요."))
        elif drift > 4:
            msgs.append(("info", f"심박 드리프트 {drift:+.1f}% — 약간의 피로 누적이 감지됩니다."))
        elif drift < -4:
            msgs.append(("info", f"심박 드리프트 {drift:+.1f}% — 후반에 강도가 낮아졌습니다."))
        else:
            msgs.append(("success", f"심박 드리프트 {drift:+.1f}% — 안정적인 페이스를 유지했습니다."))

    if z4 + z5 > 40:
        msgs.append(("warning", f"고강도 구간 {z4+z5:.0f}% — 인터벌 효과가 높았습니다. 다음 세션 전 충분히 회복하세요."))
    elif z1 + z2 > 65:
        msgs.append(("success", f"유산소 기반 훈련 {z1+z2:.0f}% — 지방 연소 및 기초 체력 향상에 효과적이었습니다."))
    elif z3 > 30:
        msgs.append(("info", f"템포 구간 {z3:.0f}% — 젖산 역치 개선에 효과적인 훈련이었습니다."))

    if w_bpm and not pd.isna(w_bpm):
        if w_bpm > 2.0:
            msgs.append(("success", f"W/bpm 효율 {w_bpm:.2f} — 심박 대비 출력이 우수합니다."))
        elif w_bpm < 1.2:
            msgs.append(("info", f"W/bpm 효율 {w_bpm:.2f} — 효율 향상을 위해 저강도 지구력 훈련을 늘려보세요."))

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
        with st.expander(f"📋 {row['filename']} — {row['sport']}", expanded=True):

            # ── 8개 메트릭 카드 (2×4) ─────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("거리",      f"{row['distance_km']:.2f} km" if row['distance_km'] else "-")
            c2.metric("이동 시간", fmt_duration(row['duration_sec']))
            c3.metric("평균 심박", f"{row['avg_hr']:.0f} bpm"     if row['avg_hr']    else "-")
            c4.metric("최대 심박", f"{row['max_hr']:.0f} bpm"     if row['max_hr']    else "-")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("평균 파워",  f"{row['avg_power']:.0f} W"   if row['avg_power'] else "-")
            c6.metric("최대 파워",  f"{row['max_power']:.0f} W"   if row['max_power'] else "-")
            c7.metric("W/bpm 효율", f"{row['w_per_bpm']:.3f}"     if row['w_per_bpm'] else "-")
            drift_val = row["drift"]
            c8.metric("HR 드리프트",
                      f"{drift_val:+.1f}%" if drift_val is not None and not pd.isna(drift_val) else "-")

            st.markdown("---")

            # ── 차트 영역 ──────────────────────────────────────────────────────
            raw_df = load_raw(row["filename"])
            col_charts, col_donut = st.columns([3, 1])

            with col_charts:
                # 심박 존 분포 바 차트
                z_vals = [row.get(f"z{i}_pct") or 0 for i in range(1, 6)]
                if any(v > 0 for v in z_vals):
                    fig_zone = go.Figure(go.Bar(
                        x=ZONE_NAMES,
                        y=z_vals,
                        marker_color=ZONE_COLORS,
                        text=[f"{v:.1f}%" for v in z_vals],
                        textposition="outside",
                    ))
                    fig_zone.update_layout(
                        title="심박 존 분포",
                        yaxis=dict(title="%", range=[0, max(z_vals) * 1.25 + 5]),
                        height=280,
                        margin=dict(t=40, b=10, l=10, r=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_zone, use_container_width=True)

                # 분당 심박 + 파워 듀얼 Y축 라인 차트
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
                        height=280,
                        margin=dict(t=40, b=10, l=10, r=10),
                        legend=dict(orientation="h", y=1.12),
                    )
                    fig_dual.update_xaxes(title_text="경과 시간 (분)")
                    fig_dual.update_yaxes(title_text="심박 (bpm)", secondary_y=False)
                    fig_dual.update_yaxes(title_text="파워 (W)",   secondary_y=True)
                    st.plotly_chart(fig_dual, use_container_width=True)

            with col_donut:
                # 케이던스 분포 도넛 차트
                if not raw_df.empty and "cad" in raw_df.columns:
                    cad_s = raw_df[raw_df["cad"] > 0]["cad"].dropna()
                    if len(cad_s) > 0:
                        bins   = [0, 60, 70, 80, 90, 100, 9999]
                        labels = ["<60", "60-69", "70-79", "80-89", "90-99", "100+"]
                        cad_cut = pd.cut(cad_s, bins=bins, labels=labels, right=False)
                        cad_cnt = cad_cut.value_counts().reindex(labels, fill_value=0)
                        cad_colors = ["#B5D4F4", "#9FE1CB", "#FAC775", "#F0997B", "#E24B4A", "#A259FF"]

                        fig_cad = go.Figure(go.Pie(
                            labels=cad_cnt.index,
                            values=cad_cnt.values,
                            hole=0.5,
                            marker_colors=cad_colors,
                        ))
                        fig_cad.update_layout(
                            title="케이던스 분포",
                            height=280,
                            margin=dict(t=40, b=10, l=5, r=5),
                            legend=dict(orientation="v", x=1.0, font=dict(size=10)),
                        )
                        st.plotly_chart(fig_cad, use_container_width=True)

            # ── 코칭 피드백 ────────────────────────────────────────────────────
            st.markdown("**💬 코칭 피드백**")
            for level, msg in coaching_feedback(dict(row)):
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
    col_y, col_m = st.columns(2)
    year = col_y.selectbox("연도", list(range(today.year - 3, today.year + 1))[::-1], index=0)
    month = col_m.selectbox("월", list(range(1, 13)), index=today.month - 1)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    month_df = df[(df["date"].apply(lambda d: d.year) == year) &
                  (df["date"].apply(lambda d: d.month) == month)]

    trained_dates = set(month_df["date"].tolist())

    cal = calendar.monthcalendar(year, month)
    day_names = ["월", "화", "수", "목", "금", "토", "일"]

    header_cols = st.columns(7)
    for i, d in enumerate(day_names):
        header_cols[i].markdown(f"<center><b>{d}</b></center>", unsafe_allow_html=True)

    for week in cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day == 0:
                cols[i].write("")
            else:
                d = date(year, month, day)
                label = f"**{day}**" if d == today else str(day)
                if d in trained_dates:
                    cols[i].markdown(f"<div style='background:#1f77b4;border-radius:4px;text-align:center;color:white'>{label} 🏃</div>", unsafe_allow_html=True)
                else:
                    cols[i].markdown(f"<div style='text-align:center'>{label}</div>", unsafe_allow_html=True)

    st.markdown("---")
    if not month_df.empty:
        st.subheader(f"{year}년 {month}월 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("훈련 횟수", f"{len(month_df)}회")
        total_dist = month_df["distance_km"].sum()
        c2.metric("총 거리", f"{total_dist:.1f} km")
        total_time = month_df["duration_sec"].sum()
        c3.metric("총 시간", fmt_duration(total_time))


# ── 탭 3: 훈련 기록 ────────────────────────────────────────────────────────────
def tab_records(df: pd.DataFrame):
    st.header("훈련 기록")

    if df.empty:
        st.info("저장된 훈련 기록이 없습니다.")
        return

    # 필터
    sports = ["전체"] + sorted(df["sport"].dropna().unique().tolist())
    sport_filter = st.selectbox("종목 필터", sports)
    filtered = df if sport_filter == "전체" else df[df["sport"] == sport_filter]

    display_cols = {
        "date": "날짜", "sport": "종목", "distance_km": "거리(km)",
        "duration_sec": "시간", "avg_hr": "평균HR", "max_hr": "최대HR",
        "avg_power": "평균W", "avg_pace": "평균페이스", "calories": "칼로리",
        "w_per_bpm": "W/bpm", "drift": "드리프트%",
    }

    show = filtered[[c for c in display_cols.keys() if c in filtered.columns]].copy()
    show["duration_sec"] = show["duration_sec"].apply(fmt_duration)
    show["avg_pace"] = show["avg_pace"].apply(fmt_pace)
    show.columns = [display_cols.get(c, c) for c in show.columns]

    st.dataframe(show, use_container_width=True, hide_index=True)

    # 삭제
    st.markdown("---")
    with st.expander("기록 삭제"):
        del_id = st.number_input("삭제할 ID", min_value=1, step=1)
        if st.button("삭제"):
            with get_conn() as conn:
                conn.execute("DELETE FROM training_log WHERE id = ?", (int(del_id),))
            st.success("삭제 완료. 페이지를 새로고침하세요.")


# ── 탭 4: 트렌드 ───────────────────────────────────────────────────────────────
def tab_trends(df: pd.DataFrame, ftp: int):
    st.header("훈련 트렌드")

    if df.empty or len(df) < 2:
        st.info("트렌드 분석을 위해 최소 2개 이상의 훈련 데이터가 필요합니다.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    metric = st.selectbox("지표 선택", ["avg_hr", "avg_power", "distance_km", "w_per_bpm", "drift", "calories"],
                          format_func=lambda x: {
                              "avg_hr": "평균 심박수", "avg_power": "평균 파워",
                              "distance_km": "거리", "w_per_bpm": "W/bpm 효율",
                              "drift": "HR 드리프트", "calories": "칼로리",
                          }.get(x, x))

    period = st.radio("기간", ["1개월", "3개월", "6개월", "전체"], horizontal=True)
    cutoff = {"1개월": 30, "3개월": 90, "6개월": 180}.get(period)
    plot_df = df[df["date"] >= pd.Timestamp(date.today()) - timedelta(days=cutoff)] if cutoff else df

    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        st.info("해당 기간에 데이터가 없습니다.")
        return

    fig = px.scatter(plot_df, x="date", y=metric, color="sport",
                     trendline="lowess",
                     labels={"date": "날짜", metric: metric},
                     title=f"{metric} 트렌드")
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 주간 볼륨 막대
    st.subheader("주간 훈련 볼륨 (km)")
    weekly = df.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    weekly_sum = weekly.groupby("week")["distance_km"].sum().reset_index()
    if cutoff:
        weekly_sum = weekly_sum[weekly_sum["week"] >= pd.Timestamp(date.today()) - timedelta(days=cutoff)]
    fig2 = px.bar(weekly_sum, x="week", y="distance_km", labels={"week": "주", "distance_km": "거리 (km)"})
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # 종목별 비율
    st.subheader("종목별 훈련 비율")
    sport_cnt = df["sport"].value_counts().reset_index()
    sport_cnt.columns = ["sport", "count"]
    fig3 = px.pie(sport_cnt, names="sport", values="count", hole=0.4)
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)


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
            st.success("삭제 완료. 페이지를 새로고침하세요.")


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="훈련 분석",
        page_icon="🏃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_db()
    max_hr, ftp, watch_dir = sidebar()
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
