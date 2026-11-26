"""StridePlan demo interface. All recommendations and interactions use the API."""

import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("MARATHON_API", "http://localhost:8000")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_KM, MAX_KM = 1.0, 100.0
MIN_MINUTES, MAX_MINUTES = 1.0, 900.0
# Each session type is a distance band crossed with a pace band relative to the runner's own
# median pace over the last 28 days. Titles name both, so two different types never share a title.
PACE_LABELS = {"much faster": "much faster", "faster": "faster", "slightly faster": "a little faster",
               "usual": "usual pace", "slightly easier": "a little easier", "easier": "easier"}

st.set_page_config(page_title="StridePlan", page_icon="🏃", layout="wide")
st.markdown("""
<style>
  .stMainBlockContainer { max-width: 1120px; padding-top: 2.5rem; }
  h1 { font-size: 2.65rem !important; letter-spacing: -.055em; font-weight: 650 !important; }
  h2, h3 { letter-spacing: -.025em; }
  [data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 600; }
  .st-key-today_card { border-radius: 18px; padding: 1.2rem; }
  .st-key-completion_card { border-radius: 14px; }
  button { border-radius: 9px !important; }
  [data-testid="stTabs"] { margin-top: 1.1rem; }
</style>
""", unsafe_allow_html=True)


def call(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=10, **kwargs)
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


def session_title(item):
    """Distance and pace band of a session type, e.g. "11 km · a little faster"."""
    return f"{item['km']:g} km · {PACE_LABELS.get(item.get('intensity'), item.get('intensity', ''))}"


def share(item):
    return f"{item['probability']:.0%}"


def rank_text(item):
    return f"#{item['rank']} · {share(item)}"


def other_options(day, count=2):
    """The highest-ranked session types other than the one planned for the day."""
    return [item for item in day["recommendations"] if item["item"] != day["planned"]["item"]][:count]


def pace_text(minutes_per_km):
    seconds = round(minutes_per_km * 60)
    return f"{seconds // 60}:{seconds % 60:02d}/km"


def activate_runner(user_id, today, label):
    st.session_state.user_id = user_id
    st.session_state.today = today
    st.session_state.runner_label = label
    st.session_state.show_completion = False
    st.session_state.pop("notice", None)


def log(kind, day, **fields):
    call("POST", "/log_interaction", json={"user_id": st.session_state.user_id, "type": kind,
                                           "date": str(day), **fields})
    st.session_state.today = day + timedelta(days=1)
    st.session_state.show_completion = False
    st.session_state.notice = "Run saved. Here is your next day." if kind == "completed" else "Day saved. Your plan is updated."
    st.rerun()


with st.sidebar:
    st.title("StridePlan")
    st.caption("Marathon training")
    mode = st.radio("Runner", ["Demo runner", "My profile"], label_visibility="collapsed")
    if mode == "Demo runner":
        runners = call("GET", "/demo/runners")
        if not runners:
            st.info("No demo runners are available. You can add your own profile.")
        else:
            labels = {f"{runner['race'].title()} · Runner {index + 1}": runner for index, runner in enumerate(runners)}
            choice = st.selectbox("Choose a runner", list(labels))
            if st.button("Load runner", type="primary", width="stretch"):
                loaded = call("POST", f"/demo/{labels[choice]['runner']}/load")
                activate_runner(loaded["user_id"], date.fromisoformat(loaded["today"]), loaded["race"].title())
    else:
        add_goal = st.checkbox("Add a target finish time")
        with st.form("profile"):
            user_id = st.text_input("Name", "lea")
            race_date = st.date_input("Marathon date", date.today() + timedelta(weeks=12))
            goal = st.number_input("Target finish (minutes)", 121, 479, 225) if add_goal else None
            if add_goal:
                st.caption("Shown for reference. Training paces follow your recent runs.")
            st.caption("Your recent runs")
            default_runs = pd.DataFrame({"date": [date.today() - timedelta(days=d) for d in (12, 10, 8, 5, 3, 1)],
                                         "km": [10., 8., 16., 10., 12., 8.],
                                         "minutes": [58., 46., 98., 56., 70., 47.]})
            runs = st.data_editor(default_runs, num_rows="dynamic", hide_index=True)
            submitted = st.form_submit_button("Save profile", type="primary", width="stretch")
        if submitted:
            sessions = [{"date": str(pd.Timestamp(row.date).date()), "km": float(row.km), "minutes": float(row.minutes)}
                        for row in runs.itertuples() if pd.notna(row.km) and pd.notna(row.minutes)
                        and row.km > 0 and row.minutes > 0]
            call("POST", "/users", json={"user_id": user_id, "race_date": str(race_date),
                                          "goal_minutes": goal, "sessions": sessions})
            activate_runner(user_id, date.today(), user_id)
    with st.expander("About the demo"):
        st.write("Demo runners use recorded marathon histories from 2019. Loading a runner restarts their demo.")
        st.caption("Session suggestions come from the trained model. A target finish time is displayed for reference only.")

if "user_id" not in st.session_state:
    st.title("Your next run, in focus.")
    st.write("A session that fits your recent running and the time left before your marathon.")
    with st.container(border=True):
        st.subheader("Start with a runner")
        st.write("Choose a demo runner in the sidebar and select **Load runner**. You can also add your own profile.")
    st.stop()

user = call("GET", f"/users/{st.session_state.user_id}")
today = st.session_state.today
recommendation = call("POST", "/recommend", json={"user_id": user["user_id"], "date": str(today), "top_k": 3})
session = recommendation["planned"]
st.title("Your next run")
runner_label = st.session_state.get("runner_label", user["user_id"])
race_date = date.fromisoformat(user["race_date"]) if user["race_date"] else None
context = f"{runner_label} · {today:%A %d %B %Y}"
if race_date:
    remaining = (race_date - today).days
    context += f" · {remaining} days to race" if remaining >= 0 else " · Race completed"
st.caption(context)
notice = st.session_state.pop("notice", None)
if notice:
    st.success(notice)

today_tab, week_tab, history_tab = st.tabs(["Today", "Your week", "History"])
with today_tab:
    with st.container(border=True, key="today_card"):
        st.caption(f"{recommendation['phase'].capitalize()} · {rank_text(session)} in the model's ranking")
        st.subheader(session_title(session))
        distance, pace, duration = st.columns(3)
        distance.metric("Distance", f"{session['km']:g} km")
        pace.metric("Pace", session["pace"])
        duration.metric("Time", duration_text(session["minutes"]))
        if recommendation["strategy"] == "phase_popularity":
            st.caption("Based on common runs at this stage. Add at least five recent runs for a personalised ranking.")
        else:
            st.caption("Ranked from your recent runs and your race date. Pace bands are relative to your median "
                       f"pace over the last 28 days ({pace_text(recommendation['reference_pace'])}).")
        done, missed = st.columns(2)
        if done.button("Log run", type="primary", width="stretch"):
            st.session_state.show_completion = True
        if missed.button("Missed today", width="stretch"):
            log("missed", today)
    if st.session_state.get("show_completion", False):
        with st.container(border=True, key="completion_card"):
            st.subheader("How did your run go?")
            with st.form(f"complete-{user['user_id']}-{today}"):
                distance_input, time_input = st.columns(2)
                km = distance_input.number_input("Distance (km)", MIN_KM, MAX_KM,
                                                 min(max(float(session["km"]), MIN_KM), MAX_KM))
                minutes = time_input.number_input("Time (min)", MIN_MINUTES, MAX_MINUTES,
                                                  min(max(float(session["minutes"]), MIN_MINUTES), MAX_MINUTES))
                feelings = {"Felt right": "ok", "Too easy": "too_easy", "Too hard": "too_hard"}
                feeling = st.selectbox("Effort", list(feelings))
                if st.form_submit_button("Save run", type="primary"):
                    log("completed", today, rec_id=recommendation["rec_id"], item=session["item"],
                        km=km, minutes=minutes, effort=feelings[feeling])
    others = other_options(recommendation)
    if others:
        st.markdown("**Other options the model ranks highly**")
        for column, option in zip(st.columns(len(others)), others):
            with column.container(border=True):
                st.caption(rank_text(option))
                st.markdown(f"**{session_title(option)}**")
                st.write(f"{option['pace']} · {duration_text(option['minutes'])}")
    if recommendation["strategy"] == "phase_popularity":
        st.caption("Percentages are shares of training sessions at this stage. Today's session is drawn from them.")
    else:
        st.caption("Percentages are the model's probability over 66 session types: it predicts what a runner with "
                   "your history is likely to do next, and no single session is very likely. Today's session is "
                   "drawn from those probabilities, as is every day of your week, so a week varies like real "
                   "training instead of repeating the single most likely session.")
    if st.button("Skip this session and move to tomorrow"):
        log("skipped", today, rec_id=recommendation["rec_id"], item=session["item"])

with week_tab:
    st.subheader("The next seven days")
    run_days = st.multiselect("Days you can run", WEEKDAYS, default=["Tue", "Thu", "Sat", "Sun"])
    if run_days:
        plan = call("POST", "/plan", json={"user_id": user["user_id"], "start_date": str(today), "days": 7,
                                          "run_weekdays": [WEEKDAYS.index(day) for day in run_days], "top_k": 3})
        if plan["days"]:
            rows = []
            for day in plan["days"]:
                planned = day["planned"]
                rows.append({"Day": pd.Timestamp(day["date"]).strftime("%a %d %b"),
                             "Session": session_title(planned), "Pace": planned["pace"],
                             "Time": duration_text(planned["minutes"]), "Model rank": rank_text(planned),
                             "Top-ranked options": " / ".join(session_title(item) for item in other_options(day))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Each session is drawn from the model's probabilities, then assumed done before the next day. "
                       "Model rank shows where it stands among 66 session types. The plan stays the same while you "
                       "follow it; log what happens to update it.")
        else:
            st.info("There are no training days in this period before your race.")
    else:
        st.info("Choose at least one day to see your week.")

with history_tab:
    st.subheader("Your recent running")
    history = pd.DataFrame(user["sessions"])
    if history.empty:
        st.info("Your completed runs will appear here.")
    else:
        history["date"] = pd.to_datetime(history["date"])
        recent = history[(history["date"] >= pd.Timestamp(today) - pd.Timedelta(weeks=8))
                         & (history["date"] < pd.Timestamp(today))]
        st.bar_chart(recent.set_index("date")["km"], height=220)
        with st.expander("View runs"):
            st.dataframe(history[["date", "km", "minutes"]].sort_values("date", ascending=False),
                         hide_index=True, width="stretch")
    with st.expander("Recorded feedback"):
        if user["interactions"]:
            st.dataframe(pd.DataFrame(user["interactions"])[["date", "type", "effort"]], hide_index=True, width="stretch")
        else:
            st.caption("No feedback yet.")

with st.sidebar:
    with st.expander("Profile details"):
        st.write(f"Runner: {runner_label}")
        st.write(f"Race date: {race_date:%d %B %Y}" if race_date else "No race date set")
        if user["goal_pace"]:
            st.write(f"Target race pace: {user['goal_pace']} (for reference)")
    with st.expander("Technical details"):
        st.caption(f"Model: {recommendation['model']}")
        st.caption(f"Strategy: {recommendation['strategy']}")
        st.caption(f"History used: {recommendation['history_sessions']} sessions")
        st.caption(f"Response: {recommendation['latency_ms']} ms")
