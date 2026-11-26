"""Staff page: how today's recommendation, the week plan and the forecast were produced."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import call, clock, duration_text, pace_text, session_title, setup  # noqa: E402

setup("StridePlan · Staff")
st.title("Staff: how the recommendations are made")
st.caption("Internal view for the team and the jury. Every number on this page comes from the API.")

health = call("GET", "/health")
with st.sidebar:
    st.markdown('<div class="brand">Stride<span>Plan</span> <span style="font-size:.8rem">staff</span></div>',
                unsafe_allow_html=True)
    st.caption(f"Serving model: {health['model']} · {health['items']} session types")
    st.subheader("Inspect a runner")
    runners = call("GET", "/demo/runners")
    if runners:
        labels = {f"{r['race'].title()} · runner {r['runner']}": r for r in runners}
        choice = st.selectbox("Real BMClab test runner (autumn 2019)", list(labels))
        if st.button("Load this runner", width="stretch"):
            loaded = call("POST", f"/demo/{labels[choice]['runner']}/load")
            st.session_state.user_id = loaded["user_id"]
            st.session_state.today = date.fromisoformat(loaded["today"])
            st.session_state.runner_label = loaded["race"].title()
            st.rerun()

with st.expander("The system in one minute", expanded="user_id" not in st.session_state):
    st.markdown("""
- **Recommender.** A context-aware sequential Transformer after CARCA (Rashed et al., RecSys 2022).
  - Self-attention encodes the runner's last 32 sessions: distance, pace relative to their own 28-day median, gap
    between runs, weekday, weeks to race and weeks since the last race.
  - Cross-attention scores **66 session types** (11 distance bands × 6 pace bands) for the planned day.
- **Training data.** Trained on 1.2 million sessions of 15,518 runners who ran a 2019 World Marathon Major (public BMClab
  dataset), with a temporal split: train before August, model selection in August, test from September.
- **Offline test (694,116 sessions).**
  - NDCG@10 is 0.3225, against 0.1992 for most-popular and 0.2555 for personal popularity.
  - Recall@20 is 0.7953, against 0.6240 for most-popular.
  - The race-calendar context adds +0.033 ± 0.004 NDCG@10 in the taper, over 3 seeds.
  - Details: `results/marathon/MARATHON_RESULTS.md` and `SEEDS.md`.
- **Weekly plans.** Each day is drawn from the model's probabilities (nucleus p = 0.9), seeded by runner and date. The
  planned run is fed back as done before the next day is ranked.
- **Finish-time forecast.** A linear model on the last 12 weeks: Riegel projection of the fastest run of 10 km or more,
  median pace, weekly volume, runs per week and longest run. It was fitted on the spring 2019 majors and tested on the
  autumn majors. More in the forecast section below.
""")

if "user_id" not in st.session_state:
    st.info("Sign up in the app (or load a BMClab test runner in the sidebar) to inspect a runner's recommendations.")
    st.stop()

user = call("GET", f"/users/{st.session_state.user_id}")
today = st.session_state.get("today", date.today())
st.subheader(f"Runner: {st.session_state.get('runner_label', user['user_id'])} · {today:%A %d %B %Y}")

ranking = call("POST", "/recommend", json={"user_id": user["user_id"], "date": str(today), "top_k": 66})
context = st.columns(5)
context[0].metric("Strategy", ranking["strategy"])
context[1].metric("Phase", ranking["phase"].split(",")[0])
context[2].metric("Days to race", ranking["days_to_race"] if ranking["days_to_race"] is not None else "–")
context[3].metric("History used", f"{ranking['history_sessions']} runs")
context[4].metric("Response", f"{ranking['latency_ms']:.0f} ms")
st.caption(f"Reference pace (median of the last 28 days): {pace_text(ranking['reference_pace'])}. "
           "Pace bands are relative to it. Strategy `model` means the Transformer ranked the sessions; "
           "`phase_popularity` is the cold-start fallback for runners with fewer than 5 runs.")

st.markdown("### 1. Today's ranking of all 66 session types")
planned = ranking["planned"]
st.write(f"Planned session: **{session_title(planned)}**, model rank #{planned['rank']} with probability "
         f"{planned['probability']:.1%}. It is drawn from the model's distribution, not always the top-ranked type.")
table = pd.DataFrame([{"Rank": item["rank"], "Session": session_title(item), "Distance band": item["distance_range"],
                       "Pace": item["pace"], "Time": duration_text(item["minutes"]),
                       "Probability": item["probability"], "Logit": round(item["score"], 3)}
                      for item in ranking["recommendations"]])
chart = table.head(15).set_index("Session")["Probability"]
st.bar_chart(chart, horizontal=True, height=420, color="#FC4C02")
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Probability": st.column_config.ProgressColumn(format="%.3f", min_value=0,
                                                                          max_value=float(table["Probability"].max()))})
st.caption(f"Top-10 types hold {table.head(10)['Probability'].sum():.0%} of the probability. "
           "No single session dominates, which is why plans sample instead of always taking #1.")

st.markdown("### 2. How the week is generated")
plan = call("POST", "/plan", json={"user_id": user["user_id"], "start_date": str(today), "days": 7, "top_k": 3})
st.caption(f"Strategy `{plan['strategy']}`, seed {plan['seed']} (a hash of the runner), run days Tue/Thu/Sat/Sun.")
st.dataframe(pd.DataFrame([{"Day": pd.Timestamp(day["date"]).strftime("%a %d %b"), "Phase": day["phase"],
                            "Planned": session_title(day["planned"]), "Model rank": day["planned"]["rank"],
                            "Probability": round(day["planned"]["probability"], 3),
                            "Top-ranked": session_title(day["recommendations"][0]),
                            "History fed in": day["history_sessions"]} for day in plan["days"]]),
             hide_index=True, width="stretch")
st.caption("History fed in grows by one each day: each planned run is assumed done before the next day is ranked. "
           "Over 500 real test weeks, sampled plans match real weeks' variety (0.92 distinct types per run, like real "
           "runners), whereas always taking #1 repeated sessions 44% of the time (results/marathon/PLAN_DIVERSITY.md).")

st.markdown("### 3. Finish-time forecast")
forecast = call("GET", f"/users/{user['user_id']}/forecast", params={"date": str(today)})
if forecast.get("available"):
    cols = st.columns(4)
    cols[0].metric("Forecast", clock(forecast["minutes"]))
    cols[1].metric("Likely range", f"{clock(forecast['low'])}–{clock(forecast['high'])}")
    cols[2].metric("Test error (MAE)", f"{forecast['test_mae_minutes']:.0f} min",
                   f"Riegel alone: {forecast['test_mae_riegel_minutes']:.0f} min", delta_color="off")
    cols[3].metric("Range coverage on test", f"{forecast['test_interval_coverage']:.0%}", "target 80%",
                   delta_color="off")
    names = {"riegel_minutes": "Riegel projection of fastest 10 km+ run (min)",
             "median_pace_x_marathon": "Median pace × 42.195 km (min)", "km_per_week": "Km per week",
             "runs_per_week": "Runs per week", "longest_km": "Longest run (km)"}
    st.dataframe(pd.DataFrame([{"Feature (last 12 weeks)": names[key], "Value": value}
                               for key, value in forecast["features"].items()]), hide_index=True)
    st.caption(f"{forecast['method']}, evaluated on {forecast['test_cases']:,} autumn 2019 marathons "
               "(features taken 3 weeks before each race). The range is the 10th–90th percentile of training "
               "errors; it covered fewer autumn results than intended, so read it as indicative.")
else:
    st.info(forecast.get("reason", "Forecast unavailable"))

st.markdown("### 4. What the model sees: recent history")
history = pd.DataFrame(user["sessions"])
if not history.empty:
    history = history[history["date"] < str(today)].tail(32).copy()
    history["pace"] = (history["minutes"] / history["km"]).map(pace_text)
    st.dataframe(history[["date", "km", "minutes", "pace", "source"]].iloc[::-1], hide_index=True, width="stretch")
    st.caption("The model reads the last 32 runs before the planned day. Each run is mapped to its session type with "
               "the same code as in training (marathon/data.py), so served histories match training data.")

st.markdown("### 5. Interaction log (`/log_interaction`, SQLite)")
if user["interactions"]:
    st.dataframe(pd.DataFrame(user["interactions"]), hide_index=True, width="stretch")
else:
    st.caption("No interaction logged yet for this runner.")
