# nextgen_market_analyzer.py
# ──────────────────────────────────────────────────────────────────────────────
# NextGen Market Analyzer — Unified Dashboard (Streamlit)
# Tabs:
#   1) Portfolio Analyzer (Autogen-based notes, visuals; reads JSON from fixed path)
#   2) Stock Evaluator (deterministic rules + dynamic number-rich summary)
# UI updates:
#   • No left sidebar (collapsed); top tabs used instead
#   • Clean layout: header row (metric + gauge + metric), then tables, then 2-row visual analysis
#   • Sector Score gauge centered below donut & Venn
#   • Tight chart margins/heights; hidden Plotly modebar
#   • S.No columns in tables
#   • Human-readable feedback labels in Stock Evaluator
#   • Hardcoded API keys kept as-is
# ──────────────────────────────────────────────────────────────────────────────

import json, warnings, re
from typing import Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Extra viz libs (Portfolio Analyzer)
from matplotlib_venn import venn2, venn3
import plotly.graph_objects as go

# Optional (Stock Evaluator)
from pydantic import BaseModel, Field
from jsonschema import validate as jsonschema_validate, Draft7Validator
from jsonschema.exceptions import ValidationError as JSONSchemaError

# ──────────────────────────────────────────────────────────────────────────────
# Global page & style
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextGen Market Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS polish
st.markdown("""
<style>
/* tighten vertical rhythm */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { width: 0 !important; min-width: 0 !important; }
/* headings */
h1, h2, h3 { letter-spacing: .2px; }
/* compact tables */
[data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] { gap: .25rem !important; }
/* reduce padding around plotly charts */
.css-1kyxreq, .stPlotlyChart, .element-container { margin-top: .2rem; margin-bottom: .2rem; }
/* bullets nicer spacing */
ul { margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# Header
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("### 📊")
with col_title:
    st.markdown("# NextGen Market Analyzer")

# Tabs instead of sidebar
tab_port, tab_stock = st.tabs(["Portfolio Analyzer", "Stock Evaluator"])

# Common tiny helper
def clean_json_string(s: str) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Portfolio Analyzer
# ──────────────────────────────────────────────────────────────────────────────
with tab_port:
    st.markdown("## 📈 Portfolio Diversification Checker")

    # Requirements (Autogen + API key)
    try:
        import autogen
        from autogen import AssistantAgent
        AUTOGEN_OK = True
    except Exception:
        AUTOGEN_OK = False
        st.error("❌ The 'pyautogen' package is required. Install with: `pip install pyautogen`")
        st.stop()

    API_KEY = "your api key"
    if not API_KEY:
        st.error("❌ OPENAI_API_KEY is required in this demo.")
        st.stop()

    # Data validation & metrics
    def validate_client_payload(data: Any) -> Tuple[bool, str]:
        def _is_client(obj):
            if not isinstance(obj, dict): return False
            keys = {"clientId", "currency", "funds"}
            if not keys.issubset(set(obj.keys())): return False
            if not isinstance(obj["funds"], list) or len(obj["funds"]) == 0: return False
            for f in obj["funds"]:
                if not isinstance(f, dict): return False
                if not {"fundCode", "amount", "holdings", "sectors"}.issubset(set(f.keys())): return False
                if not isinstance(f["amount"], (int, float)): return False
                if not isinstance(f["holdings"], dict): return False
                if not isinstance(f["sectors"], dict): return False
                if any(v < 0 or v > 1 for v in f["holdings"].values()): return False
                if any(v < 0 or v > 1 for v in f["sectors"].values()): return False
            return True

        if isinstance(data, list):
            for obj in data:
                if not _is_client(obj):
                    return False, "One or more client objects are malformed."
            return True, ""
        elif isinstance(data, dict):
            if _is_client(data): return True, ""
            else: return False, "Client object is malformed."
        else:
            return False, "Payload must be a JSON object or array of client objects."

    def total_portfolio_value(client: Dict) -> float:
        return float(sum(f["amount"] for f in client["funds"]))

    def weighted_sector_mix(client: Dict) -> Dict[str, float]:
        total = total_portfolio_value(client)
        mix: Dict[str, float] = {}
        for f in client["funds"]:
            w = f["amount"] / total if total > 0 else 0.0
            for sector, pct in f["sectors"].items():
                mix[sector] = mix.get(sector, 0.0) + w * float(pct)
        s = sum(mix.values())
        if s > 0:
            for k in list(mix.keys()):
                mix[k] = mix[k] / s
        return mix

    def hhi_from_mix(mix: Dict[str, float]) -> float:
        return float(sum((v ** 2) for v in mix.values()))

    def pairwise_overlap(hold_a: Dict[str, float], hold_b: Dict[str, float]) -> float:
        tickers = set(hold_a) | set(hold_b)
        return float(sum(min(hold_a.get(t, 0.0), hold_b.get(t, 0.0)) for t in tickers))

    def fund_overlap_matrix(client: Dict) -> Tuple[pd.DataFrame, float]:
        funds = client["funds"]
        names = [f["fundCode"] for f in funds]
        n = len(funds)
        mat = np.zeros((n, n), dtype=float)
        overlaps = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = 1.0
                else:
                    ov = pairwise_overlap(funds[i]["holdings"], funds[j]["holdings"])
                    mat[i, j] = ov
                    if j > i:
                        overlaps.append(ov)
        avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        df = pd.DataFrame(mat, index=names, columns=names)
        return df, avg_overlap

    def score_from_overlap(avg_overlap: float) -> float:
        return round((1.0 - avg_overlap) * 100.0, 2)

    def score_from_hhi(hhi: float) -> float:
        return round((1.0 - hhi) * 100.0, 2)

    def final_diversification_score(overlap_score: float, sector_score: float) -> float:
        return round(0.5 * overlap_score + 0.5 * sector_score, 2)

    def risk_bucket_from_hhi(hhi: float) -> str:
        if hhi >= 0.40: return "High"
        if hhi >= 0.25: return "Moderate"
        return "Low"

    # Visuals (tight margins)
    def create_riskometer_gauge(risk_level: str):
        risk_map = {"Low": 1, "Moderate": 3, "High": 5}
        risk_value = risk_map.get(risk_level, 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_value,
            gauge={
                'axis': {'range': [0, 6], 'tickvals': [1, 3, 5], 'ticktext': ['Low', 'Moderate', 'High'], 'tickfont': {'size': 11}},
                'bar': {'color': "black", 'thickness': 0.25},
                'steps': [
                    {'range': [0, 2], 'color': 'rgba(40, 167, 69, .70)'},
                    {'range': [2, 4], 'color': 'rgba(255, 193, 7, .70)'},
                    {'range': [4, 6], 'color': 'rgba(220, 53, 69, .70)'}],
            }))
        fig.update_layout(height=160, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    def create_sector_score_gauge(sector_score: float):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(sector_score),
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'black', 'thickness': 0.30},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(220, 53, 69, .6)'},
                    {'range': [50, 75], 'color': 'rgba(255, 193, 7, .6)'},
                    {'range': [75, 100], 'color': 'rgba(40, 167, 69, .6)'},
                ],
            }
        ))
        fig.update_layout(height=160, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    def plot_sector_donut_chart(mix: Dict[str, float]):
        if not mix:
            st.warning("No sector data to display.")
            return
        labels = list(mix.keys())
        values = list(mix.values())
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.45,
            textinfo='label+percent', hoverinfo='label+percent+value'
        )])
        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    def plot_fund_overlap_venn(client: Dict, df_mat: pd.DataFrame):
        funds = client.get("funds", [])
        num_funds = len(funds)
        fund_names = [f["fundCode"] for f in funds]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        plt.style.use('seaborn-v0_8-whitegrid')

        if num_funds not in [2, 3]:
            ax.text(0.5, 0.5, "Venn diagram is only available for 2 or 3 funds.",
                    ha='center', va='center', fontsize=12, wrap=True)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            return

        if num_funds == 2:
            overlap_ab = df_mat.iloc[0, 1]
            subsets = (1 - overlap_ab, 1 - overlap_ab, overlap_ab)
            v = venn2(subsets, set_labels=fund_names, ax=ax, set_colors=('skyblue', 'lightgreen'), alpha=0.7)
        else:
            ov_ab = df_mat.iloc[0, 1]
            ov_ac = df_mat.iloc[0, 2]
            ov_bc = df_mat.iloc[1, 2]
            holdings = [f['holdings'] for f in funds]
            all_tickers = set(holdings[0].keys()) | set(holdings[1].keys()) | set(holdings[2].keys())
            ov_abc = sum(min(h.get(t, 0) for h in holdings) for t in all_tickers)
            subsets = (
                1 - (ov_ab + ov_ac - ov_abc),
                1 - (ov_ab + ov_bc - ov_abc),
                ov_ab - ov_abc,
                1 - (ov_ac + ov_bc - ov_abc),
                ov_ac - ov_abc,
                ov_bc - ov_abc,
                ov_abc
            )
            v = venn3(subsets, set_labels=fund_names, ax=ax, set_colors=('skyblue', 'lightgreen', 'salmon'), alpha=0.7)

        if v is not None:
            for text in v.set_labels:
                if text: text.set_fontsize(12)
            for text in v.subset_labels:
                if text:
                    try:
                        val = float(text.get_text())
                        text.set_text(f"{val:.1%}" if val > 0.001 else "")
                    except Exception:
                        pass
        ax.set_title("Fund Overlap Analysis", fontsize=14, pad=6)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Agentic notes (3 calls)
    def _last_content(chat_result) -> str:
        if hasattr(chat_result, "chat_history") and chat_result.chat_history:
            last = chat_result.chat_history[-1]
            if hasattr(last, "content"):
                return last.content or ""
            if isinstance(last, dict) and "content" in last:
                return last["content"] or ""
        return ""

    def run_three_agent_debate(metrics: Dict[str, Any]) -> Dict[str, str]:
        assert AUTOGEN_OK and API_KEY, "Autogen and API key required."
        llm_config = {
            "config_list": [{"model": "gpt-4o-mini", "api_key": API_KEY}],
            "temperature": 0.35,
            "timeout": 120,
        }
        try:
            risk_analyst = AssistantAgent(
                name="risk_analyst",
                system_message=("You are a conservative financial analyst. "
                                "Write a compact verdict (80–120 words) covering concentration risks, sector imbalances, "
                                "and potential drawdowns. Offer 1–2 cautionary steps. No hype, no guarantees."),
                llm_config=llm_config,
            )
            growth_strategist = AssistantAgent(
                name="growth_strategist",
                system_message=("You are a growth-focused strategist. "
                                "Write a compact verdict (80–120 words) highlighting opportunities, underweighted growth areas, "
                                "and 2 practical steps to improve upside. Keep it professional, realistic, and specific."),
                llm_config=llm_config,
            )
            lead_analyst = AssistantAgent(
                name="lead_analyst",
                system_message=("You are the lead analyst and moderator. "
                                "Given two expert notes (risk and growth) and portfolio metrics, write a final advisory note "
                                "(120–180 words) that balances risk and opportunity, summarizes overlap/HHI, and lists 2–3 "
                                "actionable diversification steps. No promises."),
                llm_config=llm_config,
            )
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy", human_input_mode="NEVER",
                code_execution_config=False, llm_config=llm_config,
            )
            metrics_json = json.dumps(metrics, indent=2)
            risk_result = user_proxy.initiate_chat(
                risk_analyst, message=f"Portfolio metrics:\njson\n{metrics_json}\n\n\nWrite your *risk verdict* now.", max_turns=1)
            risk_verdict = _last_content(risk_result).strip()

            growth_result = user_proxy.initiate_chat(
                growth_strategist, message=f"Portfolio metrics:\njson\n{metrics_json}\n\n\nWrite your *growth verdict* now.", max_turns=1)
            growth_verdict = _last_content(growth_result).strip()

            lead_msg = ("Here are the portfolio metrics and two expert notes.\n\n"
                        f"METRICS:\njson\n{metrics_json}\n\n\n"
                        f"RISK ANALYST NOTE:\n{risk_verdict}\n\n"
                        f"GROWTH STRATEGIST NOTE:\n{growth_verdict}\n\n"
                        "Write the final advisory note as instructed in your system message.")
            lead_result = user_proxy.initiate_chat(lead_analyst, message=lead_msg, max_turns=1)
            lead_verdict = _last_content(lead_result).strip()

            return {
                "risk_verdict": risk_verdict,
                "growth_verdict": growth_verdict,
                "lead_verdict": lead_verdict,
                "final_note": lead_verdict,
            }
        except Exception as e:
            st.error(f"Autogen error: {e}")
            warnings.warn(f"Autogen failure: {e}")
            raise

    # Load fixed JSON
    JSON_PATH = r"D:\NGM_Project\NGM\test.json"


    def load_portfolio_json():
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"❌ Could not read portfolio JSON from {JSON_PATH}: {e}")
            st.stop()

    payload = load_portfolio_json()
    ok, msg = validate_client_payload(payload)
    if not ok:
        st.error(msg)
        st.stop()

    clients = payload if isinstance(payload, list) else [payload]
    client_ids = [c["clientId"] for c in clients]
    sel = st.selectbox("Choose client", client_ids, index=0, label_visibility="visible")
    client = next(c for c in clients if c["clientId"] == sel)

    # Compute metrics
    total_val = total_portfolio_value(client)
    mix = weighted_sector_mix(client)
    sector_hhi = hhi_from_mix(mix)
    risk_level = risk_bucket_from_hhi(sector_hhi)
    df_mat, avg_ov = fund_overlap_matrix(client)

    st.markdown(f"**Client:** {client['clientId']} · **Currency:** {client['currency']}")

    # Header row (prevents overlap)
    hdr_l, hdr_c, hdr_r = st.columns([1.1, 1.2, 0.9], gap="large")
    with hdr_l:
        st.metric("Total Portfolio Value", f"{total_val:,.2f}")
    with hdr_c:
        st.markdown("**Risk Level**")
        create_riskometer_gauge(risk_level)
    with hdr_r:
        st.metric("Sector HHI", f"{sector_hhi:.3f}")

    st.divider()

    # Data tables
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

    # Row 1: Donut + Venn
    row1_c1, row1_c2 = st.columns([1, 1], gap="large")
    with row1_c1:
        plot_sector_donut_chart(mix)
    with row1_c2:
        plot_fund_overlap_venn(client, df_mat)

    # Row 2: centered gauge
    overlap_score = score_from_overlap(avg_ov)
    sector_score = score_from_hhi(sector_hhi)
    row2_sp1, row2_main, row2_sp2 = st.columns([1, 2, 1])
    with row2_main:
        st.markdown("**Sector Score**")
        create_sector_score_gauge(sector_score)

    st.divider()

    # Scores
    final_score = final_diversification_score(overlap_score, sector_score)
    st.markdown("## Scores")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Avg Overlap", f"{avg_ov*100:.2f}%")
    sc2.metric("Overlap Score", f"{overlap_score:.2f}")
    sc3.metric("Sector HHI", f"{sector_hhi:.3f}")
    sc4.metric("Sector Score", f"{sector_score:.2f}")
    st.success(f"*Final Diversification Score*: **{final_score:.2f} / 100**")

    st.divider()
    st.markdown("## Advisor Note (Agentic Debate)")

    metrics = {
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

    with st.spinner("🤖 Running Risk, Growth, and Lead agents..."):
        results = run_three_agent_debate(metrics)

    st.markdown("#### 🔎 Agent Verdicts")
    st.info(results["risk_verdict"] or "(no output)")
    st.success(results["growth_verdict"] or "(no output)")
    st.warning(results["lead_verdict"] or "(no output)")

    st.markdown("#### ✅ Final Advisory Note")
    st.write(results["final_note"] or "(no output)")

    st.caption("© NextGen Market Analyzer — demo app. For education only; not investment advice.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Stock Evaluator (clean UI + dynamic summary)
# ──────────────────────────────────────────────────────────────────────────────
with tab_stock:
    st.markdown("## 📊 Stock Evaluator")

    # Hard-coded API key kept as-is (not used by deterministic evaluator)
    HARDCODED_DEFAULT_API_KEY = "your api key"

    # Schema models
    class StockParameters(BaseModel):
        priceEarningsRatio: float
        earningsPerShare: float
        dividendYield: float
        marketCap: float
        debtToEquityRatio: float
        returnOnEquity: float   # decimal
        returnOnAssets: float   # decimal
        currentRatio: float
        quickRatio: float
        bookValuePerShare: float

    class StockInput(BaseModel):
        stockSymbol: str = Field(..., min_length=1)
        parameters: StockParameters

    RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["stockSymbol", "feedback", "summary"],
        "properties": {
            "stockSymbol": {"type": "string"},
            "feedback": {"type": "object"},
            "summary": {"type": "string"}
        },
        "additionalProperties": True,
    }

    def validate_response_schema(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            jsonschema_validate(instance=obj, schema=RESPONSE_JSON_SCHEMA, cls=Draft7Validator)
            return True, None
        except JSONSchemaError as e:
            return False, str(e)

    # Feedback helpers
    def rule_comment_pe(v: float) -> str:
        if v < 15:   return f"P/E {v:g}: appears inexpensive versus earnings (possible undervaluation or low-growth pricing)."
        if v <= 30:  return f"P/E {v:g}: around market norms; pricing looks reasonable."
        return f"P/E {v:g}: premium valuation; implies higher growth expectations and risk."

    def rule_comment_eps(v: float) -> str:
        if v < 1:    return f"EPS {v:g}: thin profitability."
        if v < 5:    return f"EPS {v:g}: moderate profitability."
        return f"EPS {v:g}: strong earnings power."

    def rule_comment_divy(v: float) -> str:
        if v < 1:    return f"Dividend yield {v:g}%: low income."
        if v <= 3:   return f"Dividend yield {v:g}%: in a balanced range."
        return f"Dividend yield {v:g}%: high income, check sustainability."

    def rule_comment_mcap(v: float) -> str:
        trillion = 1_000_000_000_000
        trill = v / trillion
        if v >= 100 * trillion: return f"Market cap ${trill:,.2f}T: mega-cap scale and stability."
        if v >= 1 * trillion:   return f"Market cap ${trill:,.2f}T: large-cap profile."
        return f"Market cap ${trill:,.2f}T: mid/smaller-cap profile."

    def rule_comment_de(v: float) -> str:
        if v < 0.5:  return f"D/E {v:g}: conservative leverage."
        if v <= 1.5: return f"D/E {v:g}: balanced leverage."
        return f"D/E {v:g}: high leverage risk."

    def rule_comment_roe(v: float) -> str:
        pct = v * 100
        if pct < 8:  return f"ROE {pct:.1f}%: weak efficiency."
        if pct <= 15:return f"ROE {pct:.1f}%: healthy efficiency."
        return f"ROE {pct:.1f}%: excellent efficiency."

    def rule_comment_roa(v: float) -> str:
        pct = v * 100
        if pct < 5:  return f"ROA {pct:.1f}%: modest asset productivity."
        if pct <= 10:return f"ROA {pct:.1f}%: solid productivity."
        return f"ROA {pct:.1f}%: very strong productivity."

    def rule_comment_current(v: float) -> str:
        if v < 1:    return f"Current ratio {v:g}: potential liquidity stress."
        if v <= 2:   return f"Current ratio {v:g}: healthy liquidity."
        return f"Current ratio {v:g}: very high, capital may be idle."

    def rule_comment_quick(v: float) -> str:
        if v < 1:    return f"Quick ratio {v:g}: tight immediate liquidity."
        if v <= 2:   return f"Quick ratio {v:g}: strong immediate liquidity."
        return f"Quick ratio {v:g}: very high, conservative working capital."

    def rule_comment_bvps(v: float) -> str:
        return f"BVPS {v:g}: per-share net assets; market often trades at a premium/discount to this."

    # Dynamic summary helpers
    def _label_pe(v: float) -> str:
        return "cheap" if v < 15 else ("fairly valued" if v <= 30 else "expensive")

    def _label_de(v: float) -> str:
        return "low" if v < 0.5 else ("balanced" if v <= 1.5 else "high")

    def _label_roe(v: float) -> str:
        pct = v * 100
        return "excellent" if pct > 15 else ("healthy" if pct >= 8 else "weak")

    def _label_roa(v: float) -> str:
        pct = v * 100
        return "very strong" if pct > 10 else ("solid" if pct >= 5 else "modest")

    def _label_liquidity(cr: float, qr: float) -> str:
        if cr < 1 or qr < 1: return "tight"
        if cr <= 2 and qr <= 2: return "healthy"
        return "very high"

    def _fmt_trillions(v: float) -> str:
        return f"${v/1_000_000_000_000:,.2f}T"

    def build_dynamic_summary(stock: "StockInput") -> str:
        s = stock.stockSymbol
        p = stock.parameters
        pe_lbl = _label_pe(p.priceEarningsRatio)
        de_lbl = _label_de(p.debtToEquityRatio)
        roe_lbl = _label_roe(p.returnOnEquity)
        roa_lbl = _label_roa(p.returnOnAssets)
        liq_lbl = _label_liquidity(p.currentRatio, p.quickRatio)
        score = 0
        score += 2 if pe_lbl == "cheap" else (1 if pe_lbl == "fairly valued" else 0)
        score += 2 if roe_lbl == "excellent" else (1 if roe_lbl == "healthy" else 0)
        score += 2 if de_lbl == "low" else (1 if de_lbl == "balanced" else 0)
        score += 1 if liq_lbl == "healthy" else (0 if liq_lbl == "very high" else -1)
        stance = "compelling" if score >= 6 else ("balanced" if score >= 4 else "cautious")

        summary = (
            f"**{s} — At-a-glance**\n"
            f"- **Valuation:** P/E **{p.priceEarningsRatio:.1f}** ({pe_lbl}); EPS **{p.earningsPerShare:.2f}**; "
            f"Dividend yield **{p.dividendYield:.2f}%**.\n"
            f"- **Profitability:** ROE **{p.returnOnEquity*100:.1f}%** ({roe_lbl}), "
            f"ROA **{p.returnOnAssets*100:.1f}%** ({roa_lbl}).\n"
            f"- **Leverage & Liquidity:** D/E **{p.debtToEquityRatio:.2f}** ({de_lbl}); "
            f"Current **{p.currentRatio:.2f}**, Quick **{p.quickRatio:.2f}** ({liq_lbl}).\n"
            f"- **Scale & Book:** Market cap **{_fmt_trillions(p.marketCap)}**; BVPS **{p.bookValuePerShare:.2f}**.\n\n"
            f"**Interpretation:** With valuation {pe_lbl}, profitability {roe_lbl.lower()} ROE and {roa_lbl.lower()} ROA, "
            f"and leverage {de_lbl}, overall the setup looks **{stance}**. "
            f"Income appeal is {'limited' if p.dividendYield < 1 else ('balanced' if p.dividendYield <= 3 else 'high')} "
            f"at **{p.dividendYield:.2f}%**. Consider peer comparisons and growth drivers before acting."
        )
        return summary

    def deterministic_evaluate(stock: "StockInput") -> Dict[str, Any]:
        p = stock.parameters
        feedback = {
            "priceEarningsRatio": rule_comment_pe(p.priceEarningsRatio),
            "earningsPerShare": rule_comment_eps(p.earningsPerShare),
            "dividendYield": rule_comment_divy(p.dividendYield),
            "marketCap": rule_comment_mcap(p.marketCap),
            "debtToEquityRatio": rule_comment_de(p.debtToEquityRatio),
            "returnOnEquity": rule_comment_roe(p.returnOnEquity),
            "returnOnAssets": rule_comment_roa(p.returnOnAssets),
            "currentRatio": rule_comment_current(p.currentRatio),
            "quickRatio": rule_comment_quick(p.quickRatio),
            "bookValuePerShare": rule_comment_bvps(p.bookValuePerShare),
        }
        return {
            "stockSymbol": stock.stockSymbol,
            "feedback": feedback,
            "summary": build_dynamic_summary(stock),
        }

    # Load fixed JSON
    JSON_PATH = r"D:\NGM_Project\NGM\StockTickerSymbols_FILLED.json"


    @st.cache_data
    def load_json_data():
        try:
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            st.error(f"Could not read JSON file: {e}")
            return []

    stocks_data = load_json_data()

    if not stocks_data:
        st.warning("⚠ No stock data loaded. Please check the JSON file.")
    else:
        symbols = [s.get("stockSymbol", f"Stock{i+1}") for i, s in enumerate(stocks_data)]
        selected_symbol = st.selectbox("Choose a stock", symbols, index=0)

        selected_stock_dict = next((s for s in stocks_data if s.get("stockSymbol") == selected_symbol), None)

        if selected_stock_dict and st.button("🔎 Run Analysis", type="primary"):
            try:
                stock = StockInput(**selected_stock_dict)
                with st.spinner("Evaluating…"):
                    result = deterministic_evaluate(stock)
            except Exception:
                result = deterministic_evaluate(StockInput(**selected_stock_dict))

            st.success("✅ Analysis complete")

            st.subheader("📄 Detailed Analysis Report")
            symbol = result.get("stockSymbol", selected_symbol)
            st.write(f"**Stock Symbol:** {symbol}")

            # Compact metrics table with S.No
            p = StockInput(**selected_stock_dict).parameters
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

            # Human-friendly feedback list
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
            if isinstance(feedback, dict) and feedback:
                st.markdown("**Detailed Feedback by Parameter:**")
                items = []
                for key, sentence in feedback.items():
                    label = nice_labels.get(key, key)
                    items.append(f"- **{label}**: {sentence}")
                st.markdown("\n".join(items))
            else:
                st.warning("No feedback available.")

            st.markdown("**Overall Summary (dynamic):**")
            st.write(result.get("summary", "No summary available."))
