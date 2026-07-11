import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
import os

# Internal Imports
import utils
import agents

# Load variables from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ──────────────────────────────────────────────────────────────────────────────
# Global page configuration & UI styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextGen Market Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { width: 0 !important; min-width: 0 !important; }
h1, h2, h3 { letter-spacing: .2px; }
[data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] { gap: .25rem !important; }
.stPlotlyChart, .element-container { margin-top: .2rem; margin-bottom: .2rem; }
ul { margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# Layout Header
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("### 📊")
with col_title:
    st.markdown("# NextGen Market Analyzer")

tab_port, tab_stock = st.tabs(["Portfolio Analyzer", "Stock Evaluator"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Portfolio Analyzer
# ──────────────────────────────────────────────────────────────────────────────
with tab_port:
    st.markdown("## 📈 Portfolio Diversification Checker")

    if not API_KEY:
        st.error("❌ `GEMINI_API_KEY` was not detected in your environment setup or `.env` file.")
        st.stop()

    # Load dynamic JSON file safely
    try:
        with open(utils.PORTFOLIO_JSON_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        st.error(f"❌ Target context JSON cannot be read at path `{utils.PORTFOLIO_JSON_PATH}`: {e}")
        st.stop()

    ok, msg = utils.validate_client_payload(payload)
    if not ok:
        st.error(msg)
        st.stop()

    clients = payload if isinstance(payload, list) else [payload]
    client_ids = [c["clientId"] for c in clients]
    sel = st.selectbox("Choose client", client_ids, index=0)
    client = next(c for c in clients if c["clientId"] == sel)

    # Computations
    total_val = utils.total_portfolio_value(client)
    mix = utils.weighted_sector_mix(client)
    sector_hhi = utils.hhi_from_mix(mix)
    risk_level = utils.risk_bucket_from_hhi(sector_hhi)
    df_mat, avg_ov = utils.fund_overlap_matrix(client)

    st.markdown(f"**Client:** {client['clientId']} · **Currency:** {client['currency']}")

    # Metrics Display Metrics Row
    hdr_l, hdr_c, hdr_r = st.columns([1.1, 1.2, 0.9], gap="large")
    with hdr_l:
        st.metric("Total Portfolio Value", f"{total_val:,.2f}")
    with hdr_c:
        st.markdown("**Risk Level**")
        st.plotly_chart(utils.create_riskometer_gauge(risk_level), use_container_width=True, config={"displayModeBar": False})
    with hdr_r:
        st.metric("Sector HHI", f"{sector_hhi:.3f}")

    st.divider()

    # DataFrames Display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Funds")
        df_funds = pd.DataFrame([{
            "fundCode": f["fundCode"],
            "amount": f["amount"],
            "top_holdings": ", ".join([f"{k}:{v*100:.0f}%" for k, v in sorted(f["holdings"].items(), key=lambda kv: kv[1], reverse=True)[:3]])
        } for f in client["funds"]])
        df_funds.insert(0, "S.No", np.arange(1, len(df_funds) + 1))
        st.dataframe(df_funds, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Sector Mix (Weighted)")
        df_mix = pd.DataFrame([{"sector": k, "weight_pct": round(v*100, 4)} for k, v in sorted(mix.items(), key=lambda kv: kv[1], reverse=True)])
        df_mix.insert(0, "S.No", np.arange(1, len(df_mix) + 1))
        st.dataframe(df_mix, use_container_width=True, hide_index=True)

    st.markdown("## Visual Analysis")

    # Visualizations Row
    row1_c1, row1_c2 = st.columns([1, 1], gap="large")
    with row1_c1:
        donut = utils.create_sector_donut_chart(mix)
        if donut: st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
    with row1_c2:
        fig_venn = utils.render_fund_overlap_venn(client, df_mat)
        st.pyplot(fig_venn, use_container_width=True)
        plt.close(fig_venn)  # Closes resource properly to prevent memory leaks

    overlap_score = utils.score_from_overlap(avg_ov)
    sector_score = utils.score_from_hhi(sector_hhi)
    
    row2_sp1, row2_main, row2_sp2 = st.columns([1, 2, 1])
    with row2_main:
        st.markdown("**Sector Score**")
        st.plotly_chart(utils.create_sector_score_gauge(sector_score), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Calculation Output Metric Grid
    final_score = utils.final_diversification_score(overlap_score, sector_score)
    st.markdown("## Scores")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Avg Overlap", f"{avg_ov*100:.2f}%")
    sc2.metric("Overlap Score", f"{overlap_score:.2f}")
    sc3.metric("Sector HHI", f"{sector_hhi:.3f}")
    sc4.metric("Sector Score", f"{sector_score:.2f}")
    st.success(f"*Final Diversification Score*: **{final_score:.2f} / 100**")

    st.divider()
    st.markdown("## Advisor Note (Agentic Debate)")

    metrics_payload = {
        "clientId": client["clientId"],
        "currency": client["currency"],
        "total_value": total_val,
        "weighted_sector_mix": mix,
        "sector_hhi": sector_hhi,
        "avg_overlap": avg_ov,
        "overlap_score": overlap_score,
        "sector_score": sector_score,
        "final_score": final_score,
    }

    if st.button("🤖 Generate Agent Advisory Debate", type="secondary"):
        with st.spinner("Running financial simulations via agents..."):
            try:
                results = agents.run_three_agent_debate(metrics_payload, API_KEY)
                st.markdown("#### 🔎 Agent Verdicts")
                st.info(results["risk_verdict"])
                st.success(results["growth_verdict"])
                st.warning(results["lead_verdict"])

                st.markdown("#### ✅ Final Advisory Note")
                st.write(results["final_note"])
            except Exception as e:
                st.error(f"Agentic Execution Error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Stock Evaluator
# ──────────────────────────────────────────────────────────────────────────────
with tab_stock:
    st.markdown("## 📊 Stock Evaluator")

    # Safe Cache Implementation
    @st.cache_data
    def load_stock_json_data():
        try:
            with open(utils.STOCK_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            st.error(f"Could not read stock symbols data target at relative path: {e}")
            return []

    stocks_data = load_stock_json_data()

    if not stocks_data:
        st.warning("⚠ No stock data loaded. Verify dataset configuration rules.")
    else:
        symbols = [s.get("stockSymbol", f"Stock{i+1}") for i, s in enumerate(stocks_data)]
        selected_symbol = st.selectbox("Choose a stock", symbols, index=0)
        selected_stock_dict = next((s for s in stocks_data if s.get("stockSymbol") == selected_symbol), None)

        if selected_stock_dict and st.button("🔎 Run Analysis", type="primary"):
            try:
                stock_parsed = utils.StockInput(**selected_stock_dict)
                with st.spinner("Evaluating structural metrics..."):
                    result = utils.deterministic_evaluate(stock_parsed)
                
                st.success("✅ Analysis complete")
                st.subheader("📄 Detailed Analysis Report")
                st.write(f"**Stock Symbol:** {result.get('stockSymbol')}")

                # Build Metrics Display Table
                p = stock_parsed.parameters
                metrics_df = pd.DataFrame([
                    ("P/E ratio", f"{p.priceEarningsRatio:.2f}"),
                    ("Earnings per share (EPS)", f"{p.earningsPerShare:.2f}"),
                    ("Dividend yield", f"{p.dividendYield:.2f}%"),
                    ("Market cap", f"{p.marketCap/1_000_000_000_000:,.2f}T"),
                    ("Debt-to-Equity (D/E)", f"{p.debtToEquityRatio:.2f}"),
                    ("Return on Equity (ROE)", f"{p.returnOnEquity*100:.2f}%"),
                    ("Return on Assets (ROA)", f"{p.returnOnAssets*100:.2f}%"),
                    ("Current ratio", f"{p.currentRatio:.2f}"),
                    ("Quick ratio", f"{p.quickRatio:.2f}"),
                    ("Book value per share (BVPS)", f"{p.bookValuePerShare:.2f}"),
                ], columns=["Metric", "Value"])
                metrics_df.insert(0, "S.No", np.arange(1, len(metrics_df) + 1))
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                # Feedback Mapping Logic
                nice_labels = {
                    "priceEarningsRatio": "P/E ratio",
                    "earningsPerShare": "Earnings per share",
                    "dividendYield": "Dividend yield",
                    "marketCap": "Market cap",
                    "debtToEquityRatio": "Debt-to-Equity (D/E)",
                    "returnOnEquity": "Return on Equity (ROE)",
                    "returnOnAssets": "Return on Assets (ROA)",
                    "currentRatio": "Current ratio",
                    "quickRatio": "Quick ratio",
                    "bookValuePerShare": "Book value per share (BVPS)",
                }
                
                feedback = result.get("feedback", {})
                st.markdown("**Detailed Feedback by Parameter:**")
                items = [f"- **{nice_labels.get(k, k)}**: {sentence}" for k, sentence in feedback.items()]
                st.markdown("\n".join(items))

                st.markdown("**Overall Summary (dynamic):**")
                st.write(result.get("summary"))
                
            except Exception as err:
                st.error(f"❌ Input validation or rule evaluation failed: {err}")