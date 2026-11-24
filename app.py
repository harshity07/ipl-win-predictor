import pickle
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="IPL Win Predictor & Analytics",
    page_icon="🏏",
    layout="wide"
)

# 10 Modern IPL Franchises
TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]

CITIES = [
    "Abu Dhabi", "Ahmedabad", "Bengaluru", "Bloemfontein", "Cape Town",
    "Centurion", "Chandigarh", "Chennai", "Cuttack", "Delhi",
    "Dharamsala", "Dubai", "Durban", "East London", "Hyderabad",
    "Indore", "Jaipur", "Johannesburg", "Kanpur", "Kimberley",
    "Kochi", "Kolkata", "Mohali", "Mumbai", "Nagpur",
    "Port Elizabeth", "Pune", "Raipur", "Rajkot", "Ranchi",
    "Sharjah", "Visakhapatnam"
]

NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Lucknow Super Giants",
    "Rising Pune Supergiant": "Lucknow Super Giants",
    "Gujarat Lions": "Gujarat Titans"
}


@st.cache_resource
def load_model():
    with open("pipe.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_and_clean_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    for col in ["team1", "team2", "winner", "toss_winner"]:
        matches[col] = matches[col].replace(NAME_MAPPING)

    deliveries["batting_team"] = deliveries["batting_team"].replace(NAME_MAPPING)
    deliveries["bowling_team"] = deliveries["bowling_team"].replace(NAME_MAPPING)

    matches["city"] = matches["city"].fillna(
        matches["venue"].apply(lambda v: "Dubai" if "Dubai" in str(v) else "Unknown")
    )
    matches["city"] = matches["city"].replace({"Bangalore": "Bengaluru"})

    filtered_matches = matches[
        matches["team1"].isin(TEAMS) &
        matches["team2"].isin(TEAMS) &
        (matches["dl_applied"] == 0)
    ].copy()

    total_score = deliveries.groupby(["match_id", "inning"]).sum()["total_runs"].reset_index()
    first_inn_total = total_score[total_score["inning"] == 1].rename(columns={"total_runs": "target_score"})
    filtered_matches = filtered_matches.merge(first_inn_total[["match_id", "target_score"]], left_on="id", right_on="match_id")
    filtered_matches["target"] = filtered_matches["target_score"] + 1

    return filtered_matches, deliveries


pipe = load_model()
matches_df, deliveries_df = load_and_clean_data()

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.image("IPL.webp", width=110)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Feature",
    [
        "🏏 Match Win Predictor",
        "📊 IPL Analytics & Graphs",
        "📈 Match Chase Progression"
    ]
)

st.sidebar.divider()
st.sidebar.caption("🏏 IPL Predictor & Analytics Hub")
st.sidebar.caption("Trained on Ball-by-Ball Match Dynamics")


# =======================================================
# PAGE 1: MATCH WIN PREDICTOR
# =======================================================
if page == "🏏 Match Win Predictor":
    st.title("IPL Match Win Predictor")
    st.caption("Live second-innings chase probability estimator powered by machine learning.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Select Batting Team (Chasing)", sorted(TEAMS), index=5)
    with col2:
        bowling_team = st.selectbox("Select Bowling Team (Defending)", sorted(TEAMS), index=0)

    selected_city = st.selectbox(
        "Select Host City",
        sorted(CITIES),
        index=sorted(CITIES).index("Mumbai") if "Mumbai" in CITIES else 0
    )

    st.subheader("Match Situation")
    col_tgt, col_scr, col_ovr, col_wkt = st.columns(4)

    with col_tgt:
        target = st.number_input("Target Score", min_value=1, max_value=300, value=175, step=1)
    with col_scr:
        score = st.number_input("Current Runs Scored", min_value=0, max_value=300, value=95, step=1)
    with col_ovr:
        overs = st.number_input("Overs Completed", min_value=0.1, max_value=20.0, value=11.2, step=0.1)
    with col_wkt:
        wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=3, step=1)

    if st.button("Predict Winning Probability", type="primary"):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams cannot be the same. Please select different teams.")
            st.stop()

        progress_bar = st.progress(0)
        for pct in range(1, 101, 15):
            time.sleep(0.01)
            progress_bar.progress(pct)
        progress_bar.empty()

        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0.0
        rrr = (runs_left * 6.0) / balls_left if balls_left > 0 else 99.0

        if score >= target:
            st.success(f"{batting_team} have already chased the target and won the match!")
            win = 1.0
            loss = 0.0
        elif wickets >= 10:
            st.error(f"{batting_team} are all out! {bowling_team} won the match!")
            win = 0.0
            loss = 1.0
        elif balls_left <= 0:
            st.error(f"20 overs completed! {batting_team} failed to chase the target. {bowling_team} won!")
            win = 0.0
            loss = 1.0
        else:
            input_df = pd.DataFrame({
                "batting_team": [batting_team],
                "bowling_team": [bowling_team],
                "city": [selected_city],
                "runs_left": [runs_left],
                "balls_left": [balls_left],
                "wickets_left": [wickets_left],
                "total_runs_x": [target],
                "curr": [crr],
                "rr": [rrr]
            })

            probabilities = pipe.predict_proba(input_df)[0]
            loss = probabilities[0]
            win = probabilities[1]

        batting_win_pct = round(win * 100, 1)
        bowling_win_pct = round(loss * 100, 1)

        st.subheader("Match Verdict")
        if win >= 0.70:
            st.success(f"{batting_team} are strong favourites ({batting_win_pct}%) to win this chase.")
        elif win >= 0.50:
            st.info(f"Match slightly in favour of {batting_team} ({batting_win_pct}% vs {bowling_win_pct}%).")
        elif win >= 0.30:
            st.warning(f"{bowling_team} have the upper hand ({bowling_win_pct}% vs {batting_win_pct}%).")
        else:
            st.error(f"{bowling_team} are dominant favourites ({bowling_win_pct}%).")

        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=batting_win_pct,
                title={"text": f"{batting_team} Win %"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 40], "color": "#ffebee"},
                        {"range": [40, 65], "color": "#fff8e1"},
                        {"range": [65, 100], "color": "#e8f5e9"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            gauge_fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with vis_col2:
            comparison_df = pd.DataFrame({
                "Team": [batting_team, bowling_team],
                "Win Probability (%)": [batting_win_pct, bowling_win_pct],
                "Role": ["Chasing", "Defending"]
            })
            bar_fig = px.bar(
                comparison_df,
                x="Team",
                y="Win Probability (%)",
                color="Role",
                text="Win Probability (%)",
                color_discrete_map={"Chasing": "#1f77b4", "Defending": "#d62728"},
                title="Win Probability Comparison"
            )
            bar_fig.update_traces(texttemplate="%{text}%", textposition="outside")
            bar_fig.update_layout(yaxis=dict(range=[0, 115]), height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("Chase Equation & Run Rates")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Runs Needed", max(0, runs_left))
        with m_col2:
            st.metric("Balls Remaining", max(0, balls_left))
        with m_col3:
            st.metric("Wickets in Hand", wickets_left)
        with m_col4:
            st.metric("Required RR", f"{round(rrr, 2):.2f}")
        with m_col5:
            st.metric("Current RR", f"{round(crr, 2):.2f}", delta=f"{round(crr - rrr, 2):.2f} vs RRR")


# =======================================================
# PAGE 2: IPL ANALYTICS & GRAPHS (SEPARATE FEATURE)
# =======================================================
elif page == "📊 IPL Analytics & Graphs":
    st.title("IPL Analytics & Insights")
    st.caption("Deep-dive historical analysis across teams, stadiums, toss decisions, and players.")
    st.divider()

    tab_teams, tab_venues, tab_toss, tab_players = st.tabs([
        "Team Performance",
        "Stadiums & Pitches",
        "Toss Impact",
        "Player Leaderboards"
    ])

    # --- TAB 1: TEAM PERFORMANCE ---
    with tab_teams:
        st.header("Team Match Wins & Head-to-Head Analysis")

        team_wins = matches_df["winner"].value_counts().reset_index()
        team_wins.columns = ["Team", "Total Wins"]

        fig_team_wins = px.bar(
            team_wins,
            x="Total Wins",
            y="Team",
            orientation="h",
            color="Total Wins",
            color_continuous_scale="Viridis",
            title="Total IPL Match Wins by Franchise",
            text="Total Wins"
        )
        fig_team_wins.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        fig_team_wins.update_traces(textposition="outside")
        st.plotly_chart(fig_team_wins, use_container_width=True)

        st.subheader("Win Strategy: Defending (Batting 1st) vs Chasing (Batting 2nd)")
        bat_first_wins = matches_df[matches_df["win_by_runs"] > 0]["winner"].value_counts().rename("Defending Wins")
        chasing_wins = matches_df[matches_df["win_by_wickets"] > 0]["winner"].value_counts().rename("Chasing Wins")
        strategy_df = pd.concat([bat_first_wins, chasing_wins], axis=1).fillna(0).reset_index()
        strategy_df.columns = ["Team", "Defending Wins", "Chasing Wins"]
        strategy_df["Total Wins"] = strategy_df["Defending Wins"] + strategy_df["Chasing Wins"]
        strategy_df = strategy_df.sort_values(by="Total Wins", ascending=False)

        melted_strategy = strategy_df.melt(
            id_vars=["Team"],
            value_vars=["Defending Wins", "Chasing Wins"],
            var_name="Type",
            value_name="Wins"
        )

        fig_strategy = px.bar(
            melted_strategy,
            x="Team",
            y="Wins",
            color="Type",
            barmode="group",
            title="Wins While Batting 1st vs Chasing",
            color_discrete_map={"Defending Wins": "#ff7f0e", "Chasing Wins": "#1f77b4"}
        )
        fig_strategy.update_layout(height=420)
        st.plotly_chart(fig_strategy, use_container_width=True)

        st.divider()
        st.subheader("Head-to-Head Rivalry Comparison")
        h2h_col1, h2h_col2 = st.columns(2)
        with h2h_col1:
            team_a = st.selectbox("Select Team 1", TEAMS, index=5, key="h2h_t1")
        with h2h_col2:
            team_b = st.selectbox("Select Team 2", [t for t in TEAMS if t != team_a], index=0, key="h2h_t2")

        h2h_matches = matches_df[
            ((matches_df["team1"] == team_a) & (matches_df["team2"] == team_b)) |
            ((matches_df["team1"] == team_b) & (matches_df["team2"] == team_a))
        ]

        if len(h2h_matches) == 0:
            st.info(f"No historical matches found between {team_a} and {team_b}.")
        else:
            team_a_wins = (h2h_matches["winner"] == team_a).sum()
            team_b_wins = (h2h_matches["winner"] == team_b).sum()
            no_res = len(h2h_matches) - team_a_wins - team_b_wins

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Encounters", len(h2h_matches))
            with c2:
                st.metric(f"{team_a} Wins", team_a_wins)
            with c3:
                st.metric(f"{team_b} Wins", team_b_wins)

            h2h_summary = pd.DataFrame({
                "Result": [team_a, team_b],
                "Wins": [team_a_wins, team_b_wins]
            })
            if no_res > 0:
                h2h_summary = pd.concat([
                    h2h_summary,
                    pd.DataFrame({"Result": ["No Result / Tie"], "Wins": [no_res]})
                ])

            fig_h2h = px.pie(
                h2h_summary,
                names="Result",
                values="Wins",
                title=f"Head-to-Head Win Share: {team_a} vs {team_b}",
                hole=0.4
            )
            fig_h2h.update_layout(height=380)
            st.plotly_chart(fig_h2h, use_container_width=True)

    # --- TAB 2: STADIUMS & PITCHES ---
    with tab_venues:
        st.header("Venue Trends & Pitch Bias")

        venue_counts = matches_df["city"].value_counts().head(15).index
        venue_sub = matches_df[matches_df["city"].isin(venue_counts)]

        venue_stats = venue_sub.groupby("city").agg(
            total_matches=("id", "count"),
            chasing_wins=("win_by_wickets", lambda s: (s > 0).sum()),
            defending_wins=("win_by_runs", lambda s: (s > 0).sum()),
            avg_target=("target_score", "mean")
        ).reset_index()

        venue_stats["chase_win_pct"] = (venue_stats["chasing_wins"] / venue_stats["total_matches"] * 100).round(1)
        venue_stats["defend_win_pct"] = (venue_stats["defending_wins"] / venue_stats["total_matches"] * 100).round(1)
        venue_stats["avg_target"] = venue_stats["avg_target"].round(1)
        venue_stats = venue_stats.sort_values(by="total_matches", ascending=False)

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader("Chasing vs Defending Win % by City")
            v_melted = venue_stats.melt(
                id_vars=["city"],
                value_vars=["chase_win_pct", "defend_win_pct"],
                var_name="Advantage",
                value_name="Win Rate (%)"
            )
            v_melted["Advantage"] = v_melted["Advantage"].replace({
                "chase_win_pct": "Chasing (Bat 2nd)",
                "defend_win_pct": "Defending (Bat 1st)"
            })
            fig_venue_adv = px.bar(
                v_melted,
                x="city",
                y="Win Rate (%)",
                color="Advantage",
                barmode="group",
                title="Batting 1st vs Chasing Win Rate (%) across Cities",
                color_discrete_map={"Chasing (Bat 2nd)": "#2ca02c", "Defending (Bat 1st)": "#d62728"}
            )
            fig_venue_adv.update_layout(xaxis_tickangle=-45, height=430)
            st.plotly_chart(fig_venue_adv, use_container_width=True)

        with col_v2:
            st.subheader("Average 1st Innings Total by City")
            fig_avg_score = px.bar(
                venue_stats.sort_values(by="avg_target", ascending=False),
                x="city",
                y="avg_target",
                color="avg_target",
                color_continuous_scale="Plasma",
                title="Average 1st Innings Score across Major Stadiums",
                text="avg_target"
            )
            fig_avg_score.update_traces(textposition="outside")
            fig_avg_score.update_layout(xaxis_tickangle=-45, height=430, yaxis=dict(range=[100, 195]))
            st.plotly_chart(fig_avg_score, use_container_width=True)

        st.subheader("Venue Summary Table")
        st.dataframe(
            venue_stats.rename(columns={
                "city": "City",
                "total_matches": "Total Matches",
                "chasing_wins": "Chasing Wins",
                "defending_wins": "Defending Wins",
                "chase_win_pct": "Chasing Win %",
                "defend_win_pct": "Defending Win %",
                "avg_target": "Avg 1st Innings Score"
            }),
            use_container_width=True
        )

    # --- TAB 3: TOSS IMPACT ---
    with tab_toss:
        st.header("The Toss Factor in IPL")

        overall_toss_win_match_win = (matches_df["toss_winner"] == matches_df["winner"]).mean() * 100

        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("Total Analyzed Matches", len(matches_df))
        with t2:
            st.metric("Toss Winner Won Match %", f"{overall_toss_win_match_win:.1f}%")
        with t3:
            field_first_pct = (matches_df["toss_decision"] == "field").mean() * 100
            st.metric("Elected to Field First", f"{field_first_pct:.1f}%")

        col_toss1, col_toss2 = st.columns(2)

        with col_toss1:
            toss_decision_counts = matches_df["toss_decision"].value_counts().reset_index()
            toss_decision_counts.columns = ["Decision", "Count"]
            toss_decision_counts["Decision"] = toss_decision_counts["Decision"].replace({
                "field": "Field First",
                "bat": "Bat First"
            })
            fig_toss_dec = px.pie(
                toss_decision_counts,
                names="Decision",
                values="Count",
                title="Overall Toss Decision Distribution",
                hole=0.45,
                color="Decision",
                color_discrete_map={"Field First": "#1f77b4", "Bat First": "#ff7f0e"}
            )
            fig_toss_dec.update_layout(height=400)
            st.plotly_chart(fig_toss_dec, use_container_width=True)

        with col_toss2:
            toss_by_season = matches_df.groupby(["Season", "toss_decision"]).size().reset_index(name="Matches")
            toss_by_season["toss_decision"] = toss_by_season["toss_decision"].replace({
                "field": "Field First",
                "bat": "Bat First"
            })
            fig_toss_trend = px.bar(
                toss_by_season,
                x="Season",
                y="Matches",
                color="toss_decision",
                barmode="stack",
                title="Evolution of Toss Decision by Season",
                color_discrete_map={"Field First": "#1f77b4", "Bat First": "#ff7f0e"}
            )
            fig_toss_trend.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_toss_trend, use_container_width=True)

        st.subheader("Team-wise Toss Win to Match Win Conversion")
        toss_conversion = []
        for t in TEAMS:
            team_toss_matches = matches_df[matches_df["toss_winner"] == t]
            if len(team_toss_matches) > 0:
                toss_and_match_wins = (team_toss_matches["winner"] == t).sum()
                pct = (toss_and_match_wins / len(team_toss_matches)) * 100
                toss_conversion.append({
                    "Team": t,
                    "Tosses Won": len(team_toss_matches),
                    "Matches Won After Toss": toss_and_match_wins,
                    "Conversion Rate (%)": round(pct, 1)
                })

        toss_conv_df = pd.DataFrame(toss_conversion).sort_values(by="Conversion Rate (%)", ascending=False)
        fig_conv = px.bar(
            toss_conv_df,
            x="Team",
            y="Conversion Rate (%)",
            color="Conversion Rate (%)",
            color_continuous_scale="Tealgrn",
            title="Match Win Rate (%) When Winning the Toss",
            text="Conversion Rate (%)"
        )
        fig_conv.update_traces(textposition="outside")
        fig_conv.update_layout(xaxis_tickangle=-45, height=420, yaxis=dict(range=[0, 80]))
        st.plotly_chart(fig_conv, use_container_width=True)

    # --- TAB 4: PLAYER LEADERBOARDS ---
    with tab_players:
        st.header("IPL Hall of Fame & Player Statistics")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.subheader("Top 10 Run Scorers")
            top_batsmen = deliveries_df.groupby("batsman")["batsman_runs"].sum().reset_index()
            top_batsmen = top_batsmen.sort_values(by="batsman_runs", ascending=False).head(10)
            fig_batsmen = px.bar(
                top_batsmen,
                x="batsman_runs",
                y="batsman",
                orientation="h",
                color="batsman_runs",
                color_continuous_scale="Blues",
                title="Highest All-Time Run Scorers",
                text="batsman_runs"
            )
            fig_batsmen.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            fig_batsmen.update_traces(textposition="outside")
            st.plotly_chart(fig_batsmen, use_container_width=True)

        with col_p2:
            st.subheader("Top 10 Wicket Takers")
            dismissal_types = ["bowled", "caught", "caught and bowled", "lbw", "stumped"]
            wickets_df = deliveries_df[deliveries_df["dismissal_kind"].isin(dismissal_types)]
            top_bowlers = wickets_df.groupby("bowler")["player_dismissed"].count().reset_index()
            top_bowlers = top_bowlers.sort_values(by="player_dismissed", ascending=False).head(10)
            fig_bowlers = px.bar(
                top_bowlers,
                x="player_dismissed",
                y="bowler",
                orientation="h",
                color="player_dismissed",
                color_continuous_scale="Reds",
                title="Highest All-Time Wicket Takers",
                text="player_dismissed"
            )
            fig_bowlers.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            fig_bowlers.update_traces(textposition="outside")
            st.plotly_chart(fig_bowlers, use_container_width=True)

        st.subheader("Boundary Masters: Most Sixes in IPL")
        sixes_df = deliveries_df[deliveries_df["batsman_runs"] == 6].groupby("batsman")["batsman_runs"].count().reset_index()
        sixes_df.columns = ["Batsman", "Sixes"]
        top_sixes = sixes_df.sort_values(by="Sixes", ascending=False).head(10)

        fig_sixes = px.bar(
            top_sixes,
            x="Batsman",
            y="Sixes",
            color="Sixes",
            color_continuous_scale="Inferno",
            title="Top 10 Players with Most Sixes",
            text="Sixes"
        )
        fig_sixes.update_traces(textposition="outside")
        fig_sixes.update_layout(xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig_sixes, use_container_width=True)


# =======================================================
# PAGE 3: HISTORICAL MATCH CHASE PROGRESSION
# =======================================================
elif page == "📈 Match Chase Progression":
    st.title("Historical Match Progression Simulator")
    st.caption("Replay the over-by-over win probability swing, runs scored, and wickets lost in any historical IPL chase.")
    st.divider()

    seasons = sorted(matches_df["Season"].dropna().unique())
    selected_season = st.selectbox("Select IPL Season", seasons, index=len(seasons) - 1)

    season_matches = matches_df[matches_df["Season"] == selected_season].copy()
    match_labels = [
        f"Match {row['id']}: {row['team1']} vs {row['team2']} ({row['city']}) - Winner: {row['winner']}"
        for _, row in season_matches.iterrows()
    ]

    selected_match_label = st.selectbox("Select Match to Replay Chase", match_labels)
    selected_match_id = int(selected_match_label.split(":")[0].replace("Match", "").strip())
    match_row = season_matches[season_matches["id"] == selected_match_id].iloc[0]

    st.subheader("Match Summary")
    sm_col1, sm_col2, sm_col3, sm_col4 = st.columns(4)
    with sm_col1:
        st.metric("Teams", f"{match_row['team1']} vs {match_row['team2']}")
    with sm_col2:
        st.metric("Target Set (1st Inn)", match_row["target"])
    with sm_col3:
        st.metric("Winner", match_row["winner"])
    with sm_col4:
        margin = f"{match_row['win_by_runs']} runs" if match_row["win_by_runs"] > 0 else f"{match_row['win_by_wickets']} wickets"
        st.metric("Victory Margin", margin)

    match_deliveries = deliveries_df[
        (deliveries_df["match_id"] == selected_match_id) &
        (deliveries_df["inning"] == 2)
    ].copy()

    if len(match_deliveries) == 0:
        st.warning("Second innings delivery data is not available for this match.")
    else:
        batting_team = match_deliveries["batting_team"].iloc[0]
        bowling_team = match_deliveries["bowling_team"].iloc[0]
        city = match_row["city"]
        target = match_row["target"]

        match_deliveries["current_score"] = match_deliveries["total_runs"].cumsum()
        match_deliveries["runs_left"] = target - match_deliveries["current_score"]
        match_deliveries["balls_left"] = 126 - (match_deliveries["over"] * 6 + match_deliveries["ball"])

        match_deliveries["player_dismissed"] = match_deliveries["player_dismissed"].fillna("0").apply(lambda x: 0 if x == "0" else 1)
        match_deliveries["wickets_left"] = 10 - match_deliveries["player_dismissed"].cumsum()

        over_progression = []
        for over_num, over_df in match_deliveries.groupby("over"):
            runs_in_over = over_df["total_runs"].sum()
            wickets_in_over = over_df["player_dismissed"].sum()
            last_ball = over_df.iloc[-1]

            score_at_end = last_ball["current_score"]
            runs_left_at_end = max(0, target - score_at_end)
            balls_left_at_end = max(0, 120 - (over_num * 6))
            wickets_left_at_end = last_ball["wickets_left"]

            crr_at_end = score_at_end / over_num
            rrr_at_end = (runs_left_at_end * 6.0) / balls_left_at_end if balls_left_at_end > 0 else 0.0

            if score_at_end >= target:
                win_p = 1.0
            elif wickets_left_at_end <= 0 or (balls_left_at_end <= 0 and score_at_end < target):
                win_p = 0.0
            else:
                input_row = pd.DataFrame({
                    "batting_team": [batting_team],
                    "bowling_team": [bowling_team],
                    "city": [city],
                    "runs_left": [max(1, runs_left_at_end)],
                    "balls_left": [max(1, balls_left_at_end)],
                    "wickets_left": [wickets_left_at_end],
                    "total_runs_x": [target],
                    "curr": [crr_at_end],
                    "rr": [rrr_at_end]
                })
                win_p = pipe.predict_proba(input_row)[0][1]

            over_progression.append({
                "Over": over_num,
                "Runs in Over": runs_in_over,
                "Wickets in Over": wickets_in_over,
                "Cumulative Score": score_at_end,
                "Wickets Left": wickets_left_at_end,
                "Runs Left": runs_left_at_end,
                "Win Probability (%)": round(win_p * 100, 1),
                "Loss Probability (%)": round((1.0 - win_p) * 100, 1)
            })

        prog_df = pd.DataFrame(over_progression)

        st.subheader(f"Chase Trajectory: {batting_team} Chasing {target}")

        prog_fig = go.Figure()

        prog_fig.add_trace(go.Bar(
            x=prog_df["Over"],
            y=prog_df["Runs in Over"],
            name="Runs in Over",
            marker_color="rgba(31, 119, 180, 0.45)",
            yaxis="y2"
        ))

        prog_fig.add_trace(go.Scatter(
            x=prog_df["Over"],
            y=prog_df["Win Probability (%)"],
            name=f"{batting_team} Win %",
            mode="lines+markers",
            line=dict(color="#2ca02c", width=3.5),
            marker=dict(size=7)
        ))

        prog_fig.add_trace(go.Scatter(
            x=prog_df["Over"],
            y=prog_df["Loss Probability (%)"],
            name=f"{bowling_team} Win %",
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dash")
        ))

        wicket_overs = prog_df[prog_df["Wickets in Over"] > 0]
        if len(wicket_overs) > 0:
            prog_fig.add_trace(go.Scatter(
                x=wicket_overs["Over"],
                y=wicket_overs["Win Probability (%)"],
                name="Wicket Fallen",
                mode="markers+text",
                marker=dict(color="red", size=13, symbol="x"),
                text=[f"W-{w}" for w in wicket_overs["Wickets in Over"]],
                textposition="top center"
            ))

        prog_fig.update_layout(
            title=f"Over-by-Over Win Probability Swing ({batting_team} vs {bowling_team})",
            xaxis=dict(title="Over Completed", tickmode="linear", dtick=1),
            yaxis=dict(title="Win Probability (%)", range=[0, 105]),
            yaxis2=dict(
                title="Runs in Over",
                overlaying="y",
                side="right",
                range=[0, max(25, prog_df["Runs in Over"].max() + 5)],
                showgrid=False
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=480
        )
        st.plotly_chart(prog_fig, use_container_width=True)

        st.subheader("Over-by-Over Breakdown")
        st.dataframe(prog_df, use_container_width=True)


# ----------------- FOOTER -----------------
st.divider()
st.caption("IPL Win Predictor & Analytics Hub using Python, Scikit-Learn & Streamlit")
