# =========================================================
# STREAMLIT APP - FINANCIAL INVESTMENT ADVISOR
# =========================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💰",
    layout="wide"
)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

model = joblib.load("financial_advisor_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.sub-title {
    font-size: 18px;
    color: #B0B0B0;
    text-align: center;
    margin-bottom: 40px;
}

.card {
    background-color: #1E1E1E;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.05);
}

.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.risk-low {
    color: #00FF99;
    font-size: 32px;
    font-weight: bold;
}

.risk-medium {
    color: #FFD700;
    font-size: 32px;
    font-weight: bold;
}

.risk-high {
    color: #FF4B4B;
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="big-title">💰 AI Financial Investment Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Get personalized investment risk analysis using Machine Learning</div>',
    unsafe_allow_html=True
)

# =========================================================
# LAYOUT
# =========================================================

col1, col2 = st.columns([1, 1])

# =========================================================
# INPUT SECTION
# =========================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 Enter Your Details")

    age = st.slider(
        "Select Your Age",
        min_value=18,
        max_value=65,
        value=25
    )

    savings = st.slider(
        "Monthly Savings (₹)",
        min_value=1000,
        max_value=100000,
        value=25000,
        step=1000
    )

    goal = st.selectbox(
        "Financial Goal",
        ["Short-Term", "Long-Term"]
    )

    goal_value = 1 if goal == "Short-Term" else 2

    predict_button = st.button("🔍 Analyze My Risk Profile")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDICTION SECTION
# =========================================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📈 Prediction Result")

    if predict_button:

        # Prepare Input
        input_data = np.array([[age, goal_value, savings]])

        # Scale Input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Prediction Probabilities
        probabilities = model.predict_proba(input_scaled)[0]

        # =================================================
        # RISK LABELS
        # =================================================

        risk_map = {
            0: "Low Risk",
            1: "Medium Risk",
            2: "High Risk"
        }

        risk = risk_map[prediction]

        # =================================================
        # DISPLAY RISK
        # =================================================

        if prediction == 0:
            st.markdown(
                '<p class="risk-low">🟢 LOW RISK</p>',
                unsafe_allow_html=True
            )

            recommendation = """
### Recommended Investments
- Fixed Deposits
- PPF
- Government Bonds
- Savings Schemes
"""

        elif prediction == 1:
            st.markdown(
                '<p class="risk-medium">🟡 MEDIUM RISK</p>',
                unsafe_allow_html=True
            )

            recommendation = """
### Recommended Investments
- Mutual Funds
- Index Funds
- SIP Investments
- Hybrid Funds
"""

        else:
            st.markdown(
                '<p class="risk-high">🔴 HIGH RISK</p>',
                unsafe_allow_html=True
            )

            recommendation = """
### Recommended Investments
- Stocks
- Equity Funds
- ETFs
- Cryptocurrency
"""

        st.markdown(recommendation)

        # =================================================
        # PROBABILITY CHART
        # =================================================

        st.subheader("📊 Prediction Confidence")

        prob_df = pd.DataFrame({
            "Risk Type": ["Low", "Medium", "High"],
            "Probability": probabilities
        })

        fig = px.bar(
            prob_df,
            x="Risk Type",
            y="Probability",
            text_auto='.2%',
            height=350
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.info("Enter your details and click the button to analyze.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# DASHBOARD SECTION
# =========================================================

st.markdown("## 📌 Financial Insights Dashboard")

dash1, dash2, dash3 = st.columns(3)

with dash1:
    st.markdown("""
    <div class="metric-card">
        <h3>💵 Savings Entered</h3>
        <h2>₹ {:,}</h2>
    </div>
    """.format(savings), unsafe_allow_html=True)

with dash2:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Goal Type</h3>
        <h2>{}</h2>
    </div>
    """.format(goal), unsafe_allow_html=True)

with dash3:
    st.markdown("""
    <div class="metric-card">
        <h3>👤 Age</h3>
        <h2>{}</h2>
    </div>
    """.format(age), unsafe_allow_html=True)

# =========================================================
# INVESTMENT DISTRIBUTION PIE CHART
# =========================================================

st.markdown("## 🥧 Suggested Portfolio Distribution")

portfolio_df = pd.DataFrame({
    "Investment": ["FD", "Mutual Funds", "Stocks", "Gold"],
    "Percentage": [25, 35, 30, 10]
})

pie_fig = px.pie(
    portfolio_df,
    names="Investment",
    values="Percentage",
    hole=0.45
)

pie_fig.update_layout(
    template="plotly_dark",
    height=450
)

st.plotly_chart(pie_fig, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>
Made with ❤️ using Machine Learning, Streamlit & Scikit-learn
</center>
""", unsafe_allow_html=True)