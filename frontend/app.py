"""StridePlan web app. Every recommendation, plan and forecast comes from the API and the trained models."""

import json
import random
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import (call, clock, duration_text, other_options, pace_text, rank_text, session_title,  # noqa: E402
                setup)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_KM, MAX_KM = 1.0, 100.0
MIN_MINUTES, MAX_MINUTES = 1.0, 900.0
PROFILES = Path(__file__).resolve().parents[1] / "demo" / "mock_profiles.json"
APPS = ["Strava", "Garmin Connect", "Nike Run Club", "COROS", "Polar Flow"]

setup("StridePlan · Marathon coach")


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


def load_profiles():
    return json.loads(PROFILES.read_text(encoding="utf-8"))["profiles"] if PROFILES.exists() else []


def import_history(signup, source, profile):
    """The demo's "import": a synthetic history (demo/mock_profiles.json) dated to end yesterday."""
    today = date.today()
    sessions = [{"date": str(today - timedelta(days=s["days_ago"])), "km": s["km"], "minutes": s["minutes"]}
                for s in profile["sessions"]]
    slug = re.sub(r"[^a-z0-9]+", "-", signup["name"].lower()).strip("-") or "runner"
    user_id = f"{slug}-{random.randint(1000, 9999)}"
    progress = st.progress(0, text=f"Connecting to {source}...")
    for step, text in ((30, f"Authorised. Reading your {source} activities..."),
                       (70, f"Importing {len(sessions)} runs ({profile['total_km']:.0f} km, last 12 weeks)..."),
                       (100, "Building your training profile...")):
        time.sleep(0.35)
        progress.progress(step, text=text)
    call("POST", "/users", json={"user_id": user_id, "race_date": str(signup["race_date"]),
                                  "goal_minutes": signup["goal_minutes"], "sessions": sessions})
    st.session_state.profile_meta = {"source": source, "runs": len(sessions), "km": profile["total_km"],
                                     "level": profile["level"]}
    activate_runner(user_id, today, signup["name"].split()[0] if signup["name"].strip() else "Runner")
    st.session_state.pop("signup", None)
    st.rerun()


with st.sidebar:
    st.markdown('<div class="brand">Stride<span>Plan</span></div>', unsafe_allow_html=True)
    st.caption("Your AI marathon coach")
    if "user_id" in st.session_state:
        meta = st.session_state.get("profile_meta")
        st.write(f"Signed in as **{st.session_state.get('runner_label', st.session_state.user_id)}**")
        if meta:
            st.caption(f"Synced with {meta['source']} · {meta['runs']} runs imported")
        if st.button("Sign out", width="stretch"):
            for key in ("user_id", "today", "runner_label", "profile_meta", "show_completion", "notice", "signup"):
                st.session_state.pop(key, None)
            st.rerun()
    st.caption("Open **Staff** in the menu above for the model's reasoning.")

# ----------------------------------------------------------------------------- landing and onboarding
if "user_id" not in st.session_state:
    st.markdown("""
    <div class="hero">
      <div class="pill">Built on 16,121 real marathon preparations</div>
      <h1>Train smarter.<br/>Run your best marathon.</h1>
      <p>StridePlan tells you what to run today, plans your week and forecasts your finish time.
      It learns from your own runs and from thousands of marathoners who were where you are now.</p>
    </div>""", unsafe_allow_html=True)
    features = st.columns(3)
    for column, (icon, title, text) in zip(features, (
            ("🎯", "Your session, every day", "Distance and pace for today, ranked by a model trained on 2.2 million "
                                             "real training sessions."),
            ("🔁", "Adapts when life happens", "Missed a run or felt it was too hard? Log it and tomorrow changes."),
            ("⏱️", "Finish-time forecast", "See where your training puts you on race day, and how far from your goal."))):
        with column.container(border=True):
            st.markdown(f'<div class="feature"><h3>{icon}</h3><h4>{title}</h4><div class="muted">{text}</div></div>',
                        unsafe_allow_html=True)

    st.write("")
    signup = st.session_state.get("signup")
    if signup is None:
        st.subheader("Create your free account")
        with st.form("create_account"):
            left, right = st.columns(2)
            name = left.text_input("Full name", placeholder="Camille Martin")
            left.text_input("Email", placeholder="camille@example.com")
            race_date = right.date_input("Marathon date", date.today() + timedelta(weeks=10), min_value=date.today())
            hours, minutes = right.columns(2)
            goal_h = hours.selectbox("Goal time: hours", [2, 3, 4, 5, 6], index=1)
            goal_m = minutes.selectbox("minutes", list(range(0, 60, 5)), index=6)
            if st.form_submit_button("Create my account", type="primary"):
                if not name.strip():
                    st.error("Please enter your name.")
                else:
                    st.session_state.signup = {"name": name.strip(), "race_date": race_date,
                                               "goal_minutes": float(goal_h * 60 + goal_m)}
                    st.rerun()
    else:
        st.subheader(f"Welcome, {signup['name'].split()[0]}! Connect your running app")
        st.write("We import your last 12 weeks of runs to personalise your plan. Nothing is posted on your behalf.")
        profiles = load_profiles()
        with st.expander("Presenter options"):
            names = ["Random"] + [f"{p['name']} · {p['level']}" for p in profiles]
            choice = st.selectbox("Demo history to import", names)
        columns = st.columns(len(APPS))
        for column, source in zip(columns, APPS):
            if column.button(f"Connect {source}", key=f"connect-{source}", width="stretch",
                             type="primary" if source == "Strava" else "secondary"):
                profile = random.choice(profiles) if choice == "Random" else profiles[names.index(choice) - 1]
                import_history(signup, source, profile)
        st.caption("Demo: the import uses one of five synthetic marathon-training histories.")
    st.stop()

# ----------------------------------------------------------------------------- dashboard
user = call("GET", f"/users/{st.session_state.user_id}")
today = st.session_state.today
recommendation = call("POST", "/recommend", json={"user_id": user["user_id"], "date": str(today), "top_k": 3})
session = recommendation["planned"]
runner_label = st.session_state.get("runner_label", user["user_id"])
race_date = date.fromisoformat(user["race_date"]) if user["race_date"] else None

st.title(f"Hi {runner_label}, here is your training")
context = today.strftime("%A %d %B %Y")
if race_date:
    remaining = (race_date - today).days
    context += f" · Marathon on {race_date:%d %B}" if remaining >= 0 else " · Race completed"
st.caption(context)
notice = st.session_state.pop("notice", None)
if notice:
    st.success(notice)

forecast = call("GET", f"/users/{user['user_id']}/forecast", params={"date": str(today)})
history = pd.DataFrame(user["sessions"])
recent_km = 0.0
if not history.empty:
    history["date"] = pd.to_datetime(history["date"])
    recent_km = history.loc[history["date"] >= pd.Timestamp(today) - pd.Timedelta(weeks=4), "km"].sum() / 4
kpis = st.columns(4)
if forecast.get("available"):
    delta = None
    if user["goal_minutes"]:
        gap = forecast["minutes"] - user["goal_minutes"]
        delta = f"{'+' if gap > 0 else '-'}{duration_text(abs(gap))} vs goal"
    kpis[0].metric("Forecast finish", clock(forecast["minutes"]), delta, delta_color="inverse",
                   help=f"Likely range {clock(forecast['low'])} to {clock(forecast['high'])}. Learned from real "
                        f"2019 marathons; average error {forecast['test_mae_minutes']:.0f} min on held-out races.")
else:
    kpis[0].metric("Forecast finish", "–", help=forecast.get("reason"))
kpis[1].metric("Goal", clock(user["goal_minutes"]) if user["goal_minutes"] else "–")
kpis[2].metric("Days to race", (race_date - today).days if race_date and race_date >= today else "–")
kpis[3].metric("Weekly volume", f"{recent_km:.0f} km", help="Average of the last 4 weeks")

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
    st.caption("Percentages are the model's probabilities over 66 session types. Today's session is drawn from them, "
               "so your weeks vary like real training. The Staff page explains every number.")
    if st.button("Skip this session and move to tomorrow"):
        log("skipped", today, rec_id=recommendation["rec_id"], item=session["item"])

with week_tab:
    st.subheader("Your next seven days")
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
            st.metric("Planned this week", f"{sum(day['planned']['km'] for day in plan['days']):.0f} km "
                                           f"in {len(plan['days'])} runs")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Each session is drawn from the model's probabilities, then assumed done before the next day. "
                       "The plan stays the same while you follow it; log what happens to update it.")
        else:
            st.info("There are no training days in this period before your race.")
    else:
        st.info("Choose at least one day to see your week.")

with history_tab:
    st.subheader("Your recent running")
    if history.empty:
        st.info("Your completed runs will appear here.")
    else:
        weekly = (history[history["date"] < pd.Timestamp(today)].set_index("date")["km"]
                  .resample("W-SUN").sum().tail(12))
        st.caption("Weekly distance (km)")
        st.bar_chart(weekly, height=230, color="#FC4C02")
        with st.expander("View runs"):
            table = history[["date", "km", "minutes"]].sort_values("date", ascending=False).copy()
            table["pace"] = (table["minutes"] / table["km"]).map(pace_text)
            st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Recorded feedback"):
        if user["interactions"]:
            st.dataframe(pd.DataFrame(user["interactions"])[["date", "type", "effort"]], hide_index=True, width="stretch")
        else:
            st.caption("No feedback yet.")
