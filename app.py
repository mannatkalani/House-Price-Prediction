import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HouseIQ · Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --ink:     #0f1117;
    --paper:   #f7f6f2;
    --slate:   #1e2535;
    --accent:  #2563eb;
    --gold:    #d97706;
    --emerald: #059669;
    --rose:    #e11d48;
    --muted:   #6b7280;
    --border:  #e5e7eb;
    --card:    #ffffff;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--slate);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--accent) !important; }
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

/* Main header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    color: var(--slate);
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
}
.metric-card.gold::before  { background: var(--gold); }
.metric-card.green::before { background: var(--emerald); }
.metric-card.rose::before  { background: var(--rose); }

.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-weight: 600;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--slate);
    line-height: 1.2;
    margin-top: 0.25rem;
}
.metric-delta {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.15rem;
}

/* Prediction box */
.prediction-box {
    background: var(--slate);
    border-radius: 16px;
    padding: 2rem 2.4rem;
    text-align: center;
    color: white;
}
.pred-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 600;
}
.pred-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #ffffff;
    line-height: 1.1;
    margin: 0.5rem 0;
}
.pred-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--slate);
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
}

/* Feature badge */
.feature-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f3f4f6;
}
.feature-name { font-size: 0.875rem; color: var(--slate); flex: 1; }
.feature-val  { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--accent); font-weight: 600; }

/* Divider */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("house.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_m"] = df["price"] / 1_000_000
    df["price_k"] = df["price"] / 1_000
    return df

model = load_model()
df    = load_data()

FEATURES = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors",
            "waterfront","view","condition","sqft_above","sqft_basement",
            "yr_built","yr_renovated"]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏠 HouseIQ")
    st.markdown("<p style='color:#64748b;font-size:0.8rem;'>Configure a property to predict its value.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🛏 Interior**")
    bedrooms    = st.slider("Bedrooms",    1, 9,  3)
    bathrooms   = st.slider("Bathrooms",  1.0, 6.0, 2.0, 0.25)
    sqft_living = st.slider("Living Area (sqft)", 500, 10000, 2000, 50)
    sqft_above  = st.slider("Above Ground (sqft)", 500, 8000, 1500, 50)
    sqft_basement = st.slider("Basement (sqft)", 0, 3000, 0, 50)

    st.markdown("**🌿 Lot**")
    sqft_lot    = st.slider("Lot Size (sqft)", 1000, 50000, 8000, 500)
    floors      = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
    waterfront  = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("**⭐ Condition & View**")
    view        = st.slider("View Score",      0, 4, 0)
    condition   = st.slider("Condition Score", 1, 5, 3)

    st.markdown("**🗓 History**")
    yr_built      = st.slider("Year Built",      1900, 2014, 1975)
    yr_renovated  = st.slider("Year Renovated",  0,    2014, 0)

    st.markdown("---")
    predict_btn = st.button("✦ Get Prediction", use_container_width=True, type="primary")

# ─── Prediction ────────────────────────────────────────────────────────────────
input_data = pd.DataFrame([[
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, condition, sqft_above, sqft_basement,
    yr_built, yr_renovated
]], columns=FEATURES)

predicted_price = model.predict(input_data)[0]
predicted_price = max(0, predicted_price)

# ─── Main Layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    st.markdown("""
    <div class='hero-title'>House Price<br><em>Intelligence</em></div>
    <div class='hero-subtitle'>Linear regression model · King County, WA · 4,600 properties</div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown(f"""
    <div class='prediction-box'>
        <div class='pred-label'>Predicted Value</div>
        <div class='pred-value'>${predicted_price:,.0f}</div>
        <div class='pred-mono'>LinearRegression · sklearn 1.6.1</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─── KPI Row ───────────────────────────────────────────────────────────────────
avg_price    = df["price"].mean()
median_price = df["price"].median()
max_price    = df["price"].max()
total_homes  = len(df)
pct_vs_median = ((predicted_price - median_price) / median_price) * 100

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Dataset Size</div>
        <div class='metric-value'>{total_homes:,}</div>
        <div class='metric-delta'>King County homes</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='metric-card gold'>
        <div class='metric-label'>Average Price</div>
        <div class='metric-value'>${avg_price/1000:.0f}K</div>
        <div class='metric-delta'>Across all properties</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='metric-card green'>
        <div class='metric-label'>Median Price</div>
        <div class='metric-value'>${median_price/1000:.0f}K</div>
        <div class='metric-delta'>50th percentile</div>
    </div>""", unsafe_allow_html=True)

with k4:
    color = "green" if pct_vs_median >= 0 else "rose"
    arrow = "▲" if pct_vs_median >= 0 else "▼"
    st.markdown(f"""
    <div class='metric-card {color}'>
        <div class='metric-label'>vs. Median</div>
        <div class='metric-value'>{arrow} {abs(pct_vs_median):.1f}%</div>
        <div class='metric-delta'>Your prediction vs market median</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="large")

with c1:
    st.markdown("<div class='section-header'>Price Distribution</div><div class='section-sub'>Where your prediction falls in the market</div>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        df[df["price"] < 3_000_000], x="price",
        nbins=80,
        color_discrete_sequence=["#3b82f6"],
        template="plotly_white",
    )
    fig_hist.add_vline(
        x=predicted_price,
        line_dash="dash", line_color="#d97706", line_width=2.5,
        annotation_text=f"  Your house: ${predicted_price:,.0f}",
        annotation_font_color="#d97706",
        annotation_font_size=12,
    )
    fig_hist.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Price (USD)", yaxis_title="Count",
        font_family="Inter",
        showlegend=False,
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_hist.update_xaxes(tickformat="$,.0f", tickfont_size=11)
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.markdown("<div class='section-header'>Your Property</div><div class='section-sub'>Key inputs at a glance</div>", unsafe_allow_html=True)
    features_display = {
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Living Area": f"{sqft_living:,} sqft",
        "Lot Size": f"{sqft_lot:,} sqft",
        "Floors": floors,
        "Waterfront": "Yes" if waterfront else "No",
        "View Score": f"{view}/4",
        "Condition": f"{condition}/5",
        "Year Built": yr_built,
        "Renovated": yr_renovated if yr_renovated > 0 else "Never",
    }
    html_rows = ""
    for k, v in features_display.items():
        html_rows += f"<div class='feature-row'><span class='feature-name'>{k}</span><span class='feature-val'>{v}</span></div>"
    st.markdown(f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem 1.2rem;'>{html_rows}</div>", unsafe_allow_html=True)

# ─── Charts Row 2 ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c3, c4 = st.columns(2, gap="large")

with c3:
    st.markdown("<div class='section-header'>Price vs Living Area</div><div class='section-sub'>Size drives value — your house highlighted</div>", unsafe_allow_html=True)
    sample = df[df["price"] < 3_000_000].sample(min(1500, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="sqft_living", y="price",
        color_discrete_sequence=["#93c5fd"],
        opacity=0.45,
        template="plotly_white",
        height=300,
    )
    fig_scatter.add_scatter(
        x=[sqft_living], y=[predicted_price],
        mode="markers",
        marker=dict(size=14, color="#d97706", symbol="star", line=dict(color="white", width=1.5)),
        name="Your House",
        showlegend=False,
    )
    fig_scatter.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Living Area (sqft)", yaxis_title="Price (USD)",
        font_family="Inter",
        yaxis_tickformat="$,.0f",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with c4:
    st.markdown("<div class='section-header'>Avg Price by Bedrooms</div><div class='section-sub'>Market value segmented by room count</div>", unsafe_allow_html=True)
    bed_avg = (df[df["bedrooms"] <= 8]
               .groupby("bedrooms")["price"]
               .mean()
               .reset_index())
    colors = ["#2563eb" if int(b) == bedrooms else "#bfdbfe" for b in bed_avg["bedrooms"]]
    fig_bar = go.Figure(go.Bar(
        x=bed_avg["bedrooms"].astype(int),
        y=bed_avg["price"],
        marker_color=colors,
        text=[f"${v/1000:.0f}K" for v in bed_avg["price"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Bedrooms", yaxis_title="Avg Price (USD)",
        font_family="Inter",
        yaxis_tickformat="$,.0f",
        height=300,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── Charts Row 3 ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c5, c6 = st.columns(2, gap="large")

with c5:
    st.markdown("<div class='section-header'>Top Cities by Avg Price</div><div class='section-sub'>King County market leaders</div>", unsafe_allow_html=True)
    city_avg = (df.groupby("city")["price"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index())
    fig_city = px.bar(
        city_avg, x="price", y="city",
        orientation="h",
        color="price",
        color_continuous_scale=["#dbeafe","#1d4ed8"],
        template="plotly_white",
        height=320,
    )
    fig_city.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_tickformat="$,.0f",
        xaxis_title="Avg Price", yaxis_title="",
        font_family="Inter",
        coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_city, use_container_width=True)

with c6:
    st.markdown("<div class='section-header'>Avg Price by Condition</div><div class='section-sub'>How property condition impacts value</div>", unsafe_allow_html=True)
    cond_avg = (df.groupby("condition")["price"]
                .mean()
                .reset_index())
    colors2 = ["#2563eb" if int(c) == condition else "#bfdbfe" for c in cond_avg["condition"]]
    labels = {1:"Poor",2:"Fair",3:"Average",4:"Good",5:"Excellent"}
    fig_cond = go.Figure(go.Bar(
        x=[labels[int(c)] for c in cond_avg["condition"]],
        y=cond_avg["price"],
        marker_color=colors2,
        text=[f"${v/1000:.0f}K" for v in cond_avg["price"]],
        textposition="outside",
    ))
    fig_cond.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Condition", yaxis_title="Avg Price (USD)",
        font_family="Inter",
        yaxis_tickformat="$,.0f",
        height=320,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_cond, use_container_width=True)

# ─── Model Coefficients ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Model Feature Weights</div><div class='section-sub'>How each feature influences the predicted price (LinearRegression coefficients)</div>", unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": model.coef_,
    "Impact": ["Positive" if c > 0 else "Negative" for c in model.coef_],
}).sort_values("Coefficient", ascending=True)

fig_coef = px.bar(
    coef_df, x="Coefficient", y="Feature",
    orientation="h",
    color="Impact",
    color_discrete_map={"Positive": "#2563eb", "Negative": "#e11d48"},
    template="plotly_white",
    height=380,
    text=coef_df["Coefficient"].apply(lambda x: f"{x:+,.0f}"),
)
fig_coef.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    font_family="Inter",
    xaxis_title="Coefficient Value ($ per unit)",
    yaxis_title="",
    legend_title="",
    plot_bgcolor="white", paper_bgcolor="white",
)
st.plotly_chart(fig_coef, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;color:#9ca3af;font-size:0.78rem;'>
  HouseIQ · LinearRegression model · sklearn 1.6.1 · King County, WA dataset · 4,600 records
</p>
""", unsafe_allow_html=True)
