
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}

h1 {
    text-align:center;
    color:#FF5722;
}
</style>
""", unsafe_allow_html=True)

teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Gujrat Titans',
 'Lucknow Super Gaints']



cities=['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
       'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
       'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali',
       'Bengaluru']

pipe= pickle.load(open('pipe.pkl','rb'))
st.image(
    "IPL.webp",
    width=80
)



st.title("      "+"IPL WIN PREDICTOR")

st.sidebar.title("Menu")
team = st.sidebar.selectbox(
    "Team",
    teams
)

col1, col2 = st.columns(2)

with col1:
    batting_team=st.selectbox("Select Batting Team",sorted(teams))
with col2:
    bowling_team=st.selectbox("Select Bowling Team",sorted(teams))

selected_city= st.selectbox("Select host City",sorted(cities))

target=st.number_input("enter Target Runs")
col3,col4,col5 = st.columns(3)
with col3:
    score=st.number_input("runs scored")
with col4:
    overs=st.number_input("overs completed")
with col5:
    wickets=st.number_input("wickets out")

if st.button("predict"):

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets_left=10-wickets
    crr = score / overs if overs > 0 else 0
    rrr=(runs_left*6)/balls_left
    win=0.5
    loss=0.5
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same.")
        st.stop()
    elif wickets>=10 or overs>=20:
        st.warning(batting_team + " lost")
        win=0
        loss=1
    elif score >= target:
        st.warning(batting_team+" won")
        win=1
        loss=0
    else:
        input_df = pd.DataFrame({'batting_team':[batting_team],
                             'bowling_team':[bowling_team],
                             'city':[selected_city],
                             'runs_left':[runs_left],
                             'balls_left':[balls_left],
                             'wickets_left':[wickets_left],
                             'total_runs_x':[target],
                             'curr':[crr],
                             'rr':[rrr]})
        result =pipe.predict_proba(input_df)
        loss=result[0][0]
        win=result[0][1]

    if win > 0.7:
        st.success(f"{batting_team} are favourites to win 🏏")
    elif win > 0.5:
        st.info("Match slightly in favour of batting side")
    else:
        st.warning(f"{bowling_team} have the upper hand")

    import plotly.graph_objects as go



    fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win * 100,
            title={'text': batting_team + " Win %"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 40]},
                    {"range": [40, 70]},
                    {"range": [70, 100]}
                ]
            }
        ))


    st.plotly_chart(fig)

    col1, col2, col3 = st.columns(3)

    with col1:
            st.metric("Runs Left", runs_left)

    with col2:
            st.metric("Required RR", round(rrr, 2))

    with col3:
            st.metric("Current RR", round(crr, 2))


st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:#888888; font-size:14px;'>
        🏏 Developed by Harshit Yadav | IPL Win Predictor using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

