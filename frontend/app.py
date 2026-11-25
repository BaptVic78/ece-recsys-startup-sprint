"""StridePlan Streamlit frontend. Every number on this page comes from the FastAPI backend.

Run the API first (`python -m uvicorn marathon.api:app --port 8000`), then
`python -m streamlit run frontend/app.py` and open http://localhost:8501.
"""

import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("MARATHON_API", "http://localhost:8000")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="StridePlan", page_icon="🏃", layout="wide")


def call(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=10, **kwargs)
    except requests.ConnectionError:
        st.error(f"The API is not reachable at {API}. Start it with `python -m uvicorn marathon.api:app --port 8000`.")
        st.stop()
    if response.status_code >= 400:
        st.error(f"{path}: {response.status_code} {response.text}")
        st.stop()
    return response.json()


def log(kind, day, **fields):
    call("POST", "/log_interaction", json={"user_id": st.session_state.user_id, "type": kind, "date": str(day), **fields})
    st.session_state.today = day + timedelta(days=1)
    st.rerun()


st.title("StridePlan: your marathon training, learned from 15,000 real preparations")
st.caption("Context-aware sequential recommender (CARCA-style Transformer) trained on the public BMClab "
           "dataset of 2019 marathon runners. It ranks 66 session types for the day you plan to run.")

with st.sidebar:
    st.header("Runner")
    health = call("GET", "/health")
    st.caption(f"API {API} · model {health['model']}")
    mode = st.radio("Profile", ["Demo runner (test period)", "My own profile"])
    if mode.startswith("Demo"):
        runners = call("GET", "/demo/runners")
        labels = {f"Runner {r['runner']} · {r['race'].title()}": r for r in runners}
        choice = st.selectbox("Runner from the autumn 2019 majors", list(labels))
        if st.button("Load runner", type="primary"):
            loaded = call("POST", f"/demo/{labels[choice]['runner']}/load")
            st.session_state.user_id = loaded["user_id"]
            st.session_state.today = date.fromisoformat(loaded["today"])
    else:
        user_id = st.text_input("User name", "lea")
        race_date = st.date_input("Marathon date", date.today() + timedelta(weeks=12))
        goal = st.number_input("Goal time (minutes, shown as a pace only)", 150, 420, 225)
        st.caption("Enter your recent runs (at least 5 for personalised recommendations).")
        default_runs = pd.DataFrame({"date": [date.today() - timedelta(days=d) for d in (12, 10, 8, 5, 3, 1)],
                                     "km": [10.0, 8.0, 16.0, 10.0, 12.0, 8.0],
                                     "minutes": [58.0, 46.0, 98.0, 56.0, 70.0, 47.0]})
        runs = st.data_editor(default_runs, num_rows="dynamic")
        if st.button("Save profile", type="primary"):
            sessions = [{"date": str(pd.Timestamp(r.date).date()), "km": float(r.km), "minutes": float(r.minutes)}
                        for r in runs.itertuples() if r.km > 0 and r.minutes > 0]
            call("POST", "/users", json={"user_id": user_id, "race_date": str(race_date), "goal_minutes": float(goal),
                                         "sessions": sessions})
            st.session_state.user_id = user_id
            st.session_state.today = date.today()

if "user_id" not in st.session_state:
    st.info("Load a demo runner or save your profile in the sidebar.")
    st.stop()

user = call("GET", f"/users/{st.session_state.user_id}")
today = st.session_state.today
recommendation = call("POST", "/recommend", json={"user_id": user["user_id"], "date": str(today), "top_k": 10})

left, middle, right = st.columns(3)
left.metric("Today", today.strftime("%a %d %b %Y"))
middle.metric("Preparation phase", recommendation["phase"])
right.metric("Logged sessions", len(user["sessions"]))
if user["goal_pace"]:
    st.caption(f"Goal marathon pace: {user['goal_pace']}. Paces below are relative to your own recent median pace "
               f"({recommendation['reference_pace']:.2f} min/km).")

history = pd.DataFrame(user["sessions"])
if not history.empty:
    history["date"] = pd.to_datetime(history["date"])
    recent = history[history["date"] >= pd.Timestamp(today) - pd.Timedelta(weeks=8)]
    st.subheader("Last 8 weeks")
    st.bar_chart(recent.set_index("date")["km"], height=180)

st.subheader(f"Recommended sessions for {today:%A %d %B}")
st.caption(f"Strategy: {recommendation['strategy']} · {recommendation['history_sessions']} sessions of history · "
           f"answered in {recommendation['latency_ms']} ms")
table = pd.DataFrame(recommendation["recommendations"])[["rank", "title", "distance_range", "pace", "minutes", "score"]]
st.dataframe(table, hide_index=True, width="stretch")

top = recommendation["recommendations"][0]
st.markdown(f"**Log what you did on {today:%d %B}**")
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
km = c1.number_input("km", 1.0, 60.0, float(top["km"]))
minutes = c2.number_input("minutes", 5.0, 400.0, float(top["minutes"]))
effort = c3.selectbox("Effort", ["ok", "too_easy", "too_hard"])
with c4:
    done, skipped, missed = st.columns(3)
    if done.button("Done ✅"):
        log("completed", today, rec_id=recommendation["rec_id"], item=top["item"], km=km, minutes=minutes, effort=effort)
    if skipped.button("Skip top pick"):
        log("skipped", today, rec_id=recommendation["rec_id"], item=top["item"])
    if missed.button("Missed the day ❌"):
        log("missed", today)

st.subheader("Plan for the next 7 days")
run_days = st.multiselect("Days you can run", WEEKDAYS, default=["Tue", "Thu", "Sat", "Sun"])
if run_days:
    plan = call("POST", "/plan", json={"user_id": user["user_id"], "start_date": str(today), "days": 7,
                                       "run_weekdays": [WEEKDAYS.index(day) for day in run_days], "top_k": 3})
    rows = [{"day": pd.Timestamp(day["date"]).strftime("%a %d %b"), "phase": day["phase"],
             "session": day["recommendations"][0]["title"], "pace": day["recommendations"][0]["pace"],
             "minutes": day["recommendations"][0]["minutes"],
             "alternatives": " · ".join(item["title"] for item in day["recommendations"][1:])} for day in plan["days"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The plan is rolled forward with the model: each day's top session is assumed done before ranking the "
               "next one. Logging a missed day or a different session changes the next plan.")

with st.expander("Interaction log (stored in SQLite through /log_interaction)"):
    st.dataframe(pd.DataFrame(user["interactions"]), hide_index=True, width="stretch")
