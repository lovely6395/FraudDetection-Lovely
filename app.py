import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ── Load data & model ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dashboard_data.csv')

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

tier_filter = st.sidebar.multiselect(
    "Risk Tier",
    options=["Critical Risk", "Suspicious", "Clear"],
    default=["Critical Risk", "Suspicious", "Clear"]
)

hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value=0, max_value=23,
    value=(0, 23)
)

# Apply filters
filtered_df = df[
    (df['RiskTier'].isin(tier_filter)) &
    (df['HourOfDay'] >= hour_range[0]) &
    (df['HourOfDay'] <= hour_range[1])
]

# ── Color map ─────────────────────────────────────────────────
tier_colors = {
    'Critical Risk': '#F44336',
    'Suspicious'   : '#FF9800',
    'Clear'        : '#4CAF50'
}

# ═════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Detection — Overview")
    st.markdown("---")

    # KPI cards
    total_tx      = len(filtered_df)
    total_fraud   = int(filtered_df['actual_fraud'].sum())
    detection_rate= filtered_df['actual_fraud'].mean() * 100
    avg_fraud_amt = filtered_df[filtered_df['actual_fraud'] == 1]['TransactionAmt'].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions",  f"{total_tx:,}")
    col2.metric("Total Fraud Cases",   f"{total_fraud:,}")
    col3.metric("Detection Rate",      f"{detection_rate:.2f}%")
    col4.metric("Avg Fraud Amount",    f"${avg_fraud_amt:.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Risk tier donut
    with col_left:
        st.subheader("Risk Tier Distribution")
        tier_counts = filtered_df['RiskTier'].value_counts().reset_index()
        tier_counts.columns = ['RiskTier', 'Count']
        fig_donut = px.pie(
            tier_counts,
            names='RiskTier', values='Count',
            hole=0.5,
            color='RiskTier',
            color_discrete_map=tier_colors
        )
        fig_donut.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    # Fraud rate by hour
    with col_right:
        st.subheader("Fraud Rate by Hour of Day")
        hour_fraud = filtered_df.groupby('HourOfDay')['actual_fraud'].mean() * 100
        fig_hour = px.bar(
            x=hour_fraud.index,
            y=hour_fraud.values,
            labels={'x': 'Hour of Day', 'y': 'Fraud Rate (%)'},
            color=hour_fraud.values,
            color_continuous_scale='Reds'
        )
        fig_hour.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("---")

    # Transaction amount distribution
    st.subheader("Transaction Amount Distribution by Risk Tier")
    fig_amt = px.histogram(
        filtered_df,
        x='TransactionAmt',
        color='RiskTier',
        nbins=80,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map=tier_colors,
        log_x=True,
        labels={'TransactionAmt': 'Transaction Amount (log scale)'}
    )
    st.plotly_chart(fig_amt, use_container_width=True)

    # Scatter: TransactionAmt vs HourOfDay colored by fraud prob
    st.subheader("Transaction Amount vs Hour of Day")
    sample = filtered_df.sample(min(2000, len(filtered_df)), random_state=42)
    fig_scatter = px.scatter(
        sample,
        x='HourOfDay',
        y='TransactionAmt',
        color='fraud_prob',
        color_continuous_scale='RdYlGn_r',
        opacity=0.5,
        labels={
            'HourOfDay'    : 'Hour of Day',
            'TransactionAmt': 'Transaction Amount',
            'fraud_prob'   : 'Fraud Probability'
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION EXPLORER
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("---")

    # Search by TransactionID
    search_id = st.text_input("Search by TransactionID", placeholder="Enter TransactionID...")

    if search_id:
        result = df[df['TransactionID'].astype(str).str.contains(search_id)]
        if len(result) == 0:
            st.warning("No transaction found with that ID.")
        else:
            st.success(f"Found {len(result)} transaction(s)")
            row = result.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transaction ID",  str(row['TransactionID']))
            c2.metric("Amount",          f"${row['TransactionAmt']:.2f}")
            c3.metric("Fraud Probability", f"{row['fraud_prob']:.4f}")

            tier  = row['RiskTier']
            color = {"Critical Risk": "🔴", "Suspicious": "🟡", "Clear": "🟢"}
            c4.metric("Risk Tier", f"{color[tier]} {tier}")

    st.markdown("---")
    st.subheader("All Transactions")

    # Sort & display
    display_cols = ['TransactionID', 'TransactionAmt', 'HourOfDay',
                    'fraud_prob', 'RiskTier', 'actual_fraud']
    available    = [c for c in display_cols if c in filtered_df.columns]

    sort_col = st.selectbox("Sort by", options=['fraud_prob', 'TransactionAmt', 'HourOfDay'])
    sort_asc = st.checkbox("Ascending", value=False)

    display_data = filtered_df[available].sort_values(sort_col, ascending=sort_asc)
    st.dataframe(
        display_data.head(500).style.background_gradient(
            subset=['fraud_prob'], cmap='RdYlGn_r'
        ),
        use_container_width=True,
        height=500
    )

    st.caption(f"Showing top 500 of {len(filtered_df):,} filtered transactions")


# ═════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINER
# ═════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer")
    st.markdown("---")

    tx_input = st.text_input(
        "Enter TransactionID to explain",
        placeholder="e.g. 2987004"
    )

    if tx_input:
        row_match = df[df['TransactionID'].astype(str) == tx_input.strip()]

        if len(row_match) == 0:
            st.warning("TransactionID not found.")
        else:
            row      = row_match.iloc[0]
            prob     = row['fraud_prob']
            tier     = row['RiskTier']
            tier_icon= {"Critical Risk": "🔴", "Suspicious": "🟡", "Clear": "🟢"}

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{prob:.4f}")
            c2.metric("Risk Tier",  f"{tier_icon[tier]} {tier}")
            c3.metric("Actual Label", "🚨 FRAUD" if row['actual_fraud'] == 1 else "✅ Legit")

            st.markdown("---")
            st.subheader("SHAP Waterfall Plot")

            # Get feature columns (exclude non-feature cols)
            exclude = ['TransactionID','actual_fraud','fraud_prob','RiskTier']
            feat_cols = [c for c in df.columns if c not in exclude]
            X_row     = row[feat_cols].values.reshape(1, -1)
            X_row_df  = pd.DataFrame(X_row, columns=feat_cols)

            with st.spinner("Computing SHAP values..."):
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X_row_df)
                sv         = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                base_val   = explainer.expected_value[1] if isinstance(
                             explainer.expected_value, list) else explainer.expected_value

                explanation = shap.Explanation(
                    values      = sv,
                    base_values = base_val,
                    data        = X_row_df.iloc[0].values,
                    feature_names=feat_cols
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(explanation, max_display=15, show=False)
                st.pyplot(fig)
                plt.close()

            # Plain English explanation
            st.markdown("---")
            st.subheader("Plain-English Explanation")

            top_idx      = np.argsort(np.abs(sv))[-3:][::-1]
            top_features = [feat_cols[i] for i in top_idx]
            top_values   = [sv[i] for i in top_idx]

            if prob >= 0.75:
                verdict = "🔴 **This transaction is highly likely to be fraudulent.**"
            elif prob >= 0.40:
                verdict = "🟡 **This transaction shows mixed signals — manual review recommended.**"
            else:
                verdict = "🟢 **This transaction appears legitimate.**"

            st.markdown(verdict)
            st.markdown(f"The model's baseline fraud probability is **{base_val:.3f}**. "
                        f"For this transaction it moved to **{prob:.3f}**.")
            st.markdown("**Top contributing features:**")
            for feat, val in zip(top_features, top_values):
                direction = "↑ increased" if val > 0 else "↓ decreased"
                st.markdown(f"- `{feat}` → {direction} fraud probability by **{abs(val):.4f}**")