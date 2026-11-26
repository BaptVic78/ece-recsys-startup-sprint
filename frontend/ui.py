"""Shared helpers for the StridePlan pages: API calls, formatting and the site style."""

import os

import requests
import streamlit as st

API = os.environ.get("MARATHON_API", "http://localhost:8000")
PACE_LABELS = {"much faster": "much faster", "faster": "faster", "slightly faster": "a little faster",
               "usual": "usual pace", "slightly easier": "a little easier", "easier": "easier"}

STYLE = """
<style>
  :root { --brand: #FC4C02; --brand-dark: #C93A00; --ink: #111827; --muted: #6B7280; }
  .stMainBlockContainer { max-width: 1160px; padding-top: 1.6rem; }
  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
  h1 { font-size: 2.6rem !important; letter-spacing: -.045em; font-weight: 700 !important; }
  h2, h3 { letter-spacing: -.02em; }
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 650; }
  button { border-radius: 10px !important; }
  button[kind="primary"] { background: var(--brand) !important; border-color: var(--brand) !important; }
  button[kind="primary"]:hover { background: var(--brand-dark) !important; border-color: var(--brand-dark) !important; }
  .brand { font-weight: 800; font-size: 1.35rem; letter-spacing: -.03em; }
  .brand span { color: var(--brand); }
  .hero { padding: 3.2rem 2.6rem; border-radius: 26px; color: #fff;
          background: linear-gradient(135deg, #FC4C02 0%, #FF7A3D 45%, #1F2937 100%); margin-bottom: 1.6rem; }
  .hero h1 { color: #fff !important; font-size: 3.1rem !important; margin: 0 0 .6rem 0; line-height: 1.05; }
  .hero p { font-size: 1.15rem; opacity: .95; max-width: 680px; margin: 0; }
  .pill { display: inline-block; padding: .25rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
          background: rgba(255,255,255,.18); color: #fff; margin-bottom: 1rem; }
  .feature h4 { margin: .2rem 0 .3rem 0; }
  .muted { color: var(--muted); font-size: .92rem; }
  .st-key-today_card { border-radius: 18px; padding: 1.2rem; border-left: 6px solid var(--brand) !important; }
  .st-key-completion_card { border-radius: 14px; }
  [data-testid="stTabs"] { margin-top: 1rem; }
  table.cal { width: 100%; border-collapse: separate; border-spacing: 6px; table-layout: fixed; }
  table.cal th { font-size: .8rem; color: var(--muted); font-weight: 600; text-align: left; padding: 0 .4rem; }
  table.cal td { vertical-align: top; height: 88px; padding: .45rem .5rem; border-radius: 12px;
                 background: rgba(127,127,127,.06); border: 1px solid rgba(127,127,127,.15); font-size: .82rem; }
  table.cal td.run { background: rgba(252,76,2,.10); border-color: rgba(252,76,2,.35); }
  table.cal td.today { box-shadow: inset 0 0 0 2px var(--brand); }
  table.cal td.race { background: var(--brand); color: #fff; font-weight: 700; }
  table.cal td.past { opacity: .35; }
  table.cal td.wk { background: transparent; border: none; font-weight: 700; vertical-align: middle; }
  table.cal .d { font-size: .72rem; color: var(--muted); margin-bottom: .15rem; }
  table.cal td.race .d { color: #fff; }
  table.cal .s { font-weight: 700; font-size: .95rem; }
  table.cal .p { color: var(--muted); font-size: .74rem; }
  table.cal .r { color: var(--brand-dark); font-size: .72rem; margin-top: .15rem; }
</style>
"""


def setup(title):
    st.set_page_config(page_title=title, page_icon="🏃", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)


def call(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=15, **kwargs)
    except requests.RequestException:
        st.error("The recommendation service is unavailable. Please try again.")
        with st.expander("Connection help"):
            st.write(f"Expected API: {API}")
            st.code("python -m uvicorn marathon.api:app --port 8000")
        st.stop()
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Please check your entries.")
        except ValueError:
            detail = "Please try again."
        if isinstance(detail, list):
            detail = " ".join(error.get("msg", "Invalid entry.").removeprefix("Value error, ") for error in detail)
        st.error(str(detail))
        st.stop()
    return response.json()


def duration_text(minutes):
    hours, remainder = divmod(round(minutes), 60)
    if hours == 0:
        return f"{remainder} min"
    return f"{hours} h {remainder:02d}" if remainder else f"{hours} h"


def clock(minutes):
    """Finish time as h:mm."""
    hours, remainder = divmod(round(minutes), 60)
    return f"{hours}:{remainder:02d}"


def session_title(item):
    """Distance and pace band of a session type, e.g. "11 km · a little faster"."""
    return f"{item['km']:g} km · {PACE_LABELS.get(item.get('intensity'), item.get('intensity', ''))}"


def rank_text(item):
    return f"#{item['rank']} · {item['probability']:.0%}"


def other_options(day, count=2):
    """The highest-ranked session types other than the one planned for the day."""
    return [item for item in day["recommendations"] if item["item"] != day["planned"]["item"]][:count]


def pace_text(minutes_per_km):
    seconds = round(minutes_per_km * 60)
    return f"{seconds // 60}:{seconds % 60:02d}/km"


def brand(sidebar=True):
    target = st.sidebar if sidebar else st
    target.markdown('<div class="brand">Stride<span>Plan</span></div>', unsafe_allow_html=True)
