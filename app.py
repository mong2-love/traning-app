import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import calendar

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
def parse_fit(path, max_hr, ftp):
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

    records = []
    session_data = {}

    for msg in ff.get_messages():
        name = msg.name
        if name == "record":
            row = {f.name: f.value for f in msg.fields}
            records.append(row)
        elif name == "session":
            session_data = {f.name: f.value for f in msg.fields}

    df = pd.DataFrame(records)
    if df.empty:
        return None

    sport = session_data.get("sport", "unknown")
    if isinstance(sport, str):
        sport = sport.lower()

    indoor_flag = int(session_data.get("sub_sport", "").lower() in ["indoor_cycling", "treadmill", "virtual_activity"]
                      if isinstance(session_data.get("sub_sport", ""), str) else False)

    ts_col = next((c for c in ["timestamp", "time"] if c in df.columns), None)
    start_time = None
    duration_sec = session_data.get("total_elapsed_time")
    if ts_col and len(df) > 1:
        try:
            times = pd.to_datetime(df[ts_col])
            start_time = times.iloc[0]
            if duration_sec is None:
                duration_sec = (times.iloc[-1] - times.iloc[0]).total_seconds()
        except Exception:
            pass

    date_str = (start_time.strftime("%Y-%m-%d") if start_time
                else session_data.get("start_time", datetime.today()).strftime("%Y-%m-%d")
                if hasattr(session_data.get("start_time", ""), "strftime") else str(date.today()))

    distance_km = session_data.get("total_distance", 0)
    if distance_km and distance_km > 1000:
        distance_km /= 1000.0

    hr_col = next((c for c in ["heart_rate"] if c in df.columns), None)
    pwr_col = next((c for c in ["power"] if c in df.columns), None)
    cad_col = next((c for c in ["cadence"] if c in df.columns), None)

    avg_hr = float(df[hr_col].mean()) if hr_col else None
    max_hr_val = float(df[hr_col].max()) if hr_col else None
    avg_power = float(df[pwr_col].mean()) if pwr_col else None
    max_power = float(df[pwr_col].max()) if pwr_col else None
    avg_cad = float(df[cad_col].mean()) if cad_col else None

    w_per_bpm = (avg_power / avg_hr) if avg_power and avg_hr else None

    # 드리프트: 후반부 평균HR / 전반부 평균HR
    drift = None
    if hr_col and len(df) > 10:
        mid = len(df) // 2
        h1 = df[hr_col].iloc[:mid].mean()
        h2 = df[hr_col].iloc[mid:].mean()
        drift = round((h2 - h1) / h1 * 100, 1) if h1 else None

    # 존 분포
    zones = hr_zones(max_hr)
    z_pcts = [0.0] * 5
    if hr_col:
        hr_series = df[hr_col].dropna()
        total = len(hr_series)
        if total > 0:
            for i, (lo, hi) in enumerate(zones):
                z_pcts[i] = round(((hr_series >= lo) & (hr_series <= hi)).sum() / total * 100, 1)

    avg_pace = None
    if distance_km and duration_sec and distance_km > 0:
        avg_pace = duration_sec / distance_km

    calories = session_data.get("total_calories")

    return dict(
        date=date_str,
        filename=os.path.basename(path),
        sport=sport,
        indoor=indoor_flag,
        distance_km=round(distance_km, 2) if distance_km else None,
        duration_sec=int(duration_sec) if duration_sec else None,
        avg_hr=round(avg_hr, 1) if avg_hr else None,
        max_hr=round(max_hr_val, 1) if max_hr_val else None,
        avg_power=round(avg_power, 1) if avg_power else None,
        max_power=round(max_power, 1) if max_power else None,
        avg_cadence=round(avg_cad, 1) if avg_cad else None,
        w_per_bpm=round(w_per_bpm, 3) if w_per_bpm else None,
        drift=drift,
        z1_pct=z_pcts[0], z2_pct=z_pcts[1], z3_pct=z_pcts[2],
        z4_pct=z_pcts[3], z5_pct=z_pcts[4],
        avg_pace=round(avg_pace, 1) if avg_pace else None,
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


def parse_file(path, max_hr, ftp):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fit":
        return parse_fit(path, max_hr, ftp)
    elif ext == ".gpx":
        return parse_gpx(path, max_hr, ftp)
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
    watch_dir = st.sidebar.text_input("감시 폴더 경로", value=os.path.expanduser("~/Downloads"))

    st.sidebar.markdown("---")
    st.sidebar.subheader("파일 업로드")
    uploaded = st.sidebar.file_uploader("FIT / GPX 파일 선택", type=["fit", "gpx"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            tmp = f"/tmp/{uf.name}"
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
                if fname.lower().endswith((".fit", ".gpx")):
                    fpath = os.path.join(watch_dir, fname)
                    data = parse_file(fpath, max_hr, ftp)
                    if data:
                        save_record(data)
                        count += 1
            st.sidebar.success(f"{count}개 파일 처리 완료")
        else:
            st.sidebar.error("폴더를 찾을 수 없습니다.")

    return max_hr, ftp, watch_dir


# ── 탭 1: 오늘의 훈련 ──────────────────────────────────────────────────────────
def tab_today(df: pd.DataFrame, max_hr: int, ftp: int):
    st.header("오늘의 훈련")
    today_str = str(date.today())
    today_df = df[df["date"] == today_str] if not df.empty else pd.DataFrame()

    if today_df.empty:
        st.info("오늘 등록된 훈련 데이터가 없습니다. 사이드바에서 파일을 업로드하세요.")
        return

    for _, row in today_df.iterrows():
        with st.expander(f"📋 {row['filename']} — {row['sport']}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("거리", f"{row['distance_km']:.1f} km" if row['distance_km'] else "-")
            c2.metric("시간", fmt_duration(row['duration_sec']))
            c3.metric("평균 심박", f"{row['avg_hr']:.0f} bpm" if row['avg_hr'] else "-")
            c4.metric("평균 파워", f"{row['avg_power']:.0f} W" if row['avg_power'] else "-")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("최대 심박", f"{row['max_hr']:.0f} bpm" if row['max_hr'] else "-")
            c6.metric("W/bpm", f"{row['w_per_bpm']:.3f}" if row['w_per_bpm'] else "-")
            c7.metric("드리프트", f"{row['drift']:+.1f}%" if row['drift'] is not None and not pd.isna(row['drift']) else "-")
            c8.metric("칼로리", f"{row['calories']:.0f} kcal" if row['calories'] else "-")

            # 존 차트
            zones_pct = [row[f"z{i}_pct"] for i in range(1, 6)]
            if any(v and v > 0 for v in zones_pct):
                zone_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]
                fig = go.Figure(go.Bar(
                    x=[f"Z{i}" for i in range(1, 6)],
                    y=zones_pct,
                    marker_color=zone_colors,
                    text=[f"{v:.1f}%" for v in zones_pct],
                    textposition="outside",
                ))
                fig.update_layout(title="심박 존 분포", yaxis_title="%", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)


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
    months_available = sorted(
        df["date"].str[:7].unique().tolist(), reverse=True
    )
    sel_month = st.selectbox("리포트 기간 (월)", months_available)

    if st.button("PDF 리포트 생성"):
        m_df = df[df["date"].str.startswith(sel_month)]

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
    zone_names = ["Z1 (회복)", "Z2 (유산소)", "Z3 (템포)", "Z4 (역치)", "Z5 (최대)"]
    zone_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]
    for name, (lo, hi), color in zip(zone_names, zones, zone_colors):
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
