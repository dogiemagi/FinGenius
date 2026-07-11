import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import plotly.graph_objects as go
from matplotlib_venn import venn2, venn3
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# PATH AND DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_JSON_PATH = os.path.join(BASE_DIR, "test.json")
STOCK_JSON_PATH = os.path.join(BASE_DIR, "StockTickerSymbols_FILLED.json")

def clean_json_string(s: str) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: PORTFOLIO MODELS & METRICS
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO VISUALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def create_riskometer_gauge(risk_level: str) -> go.Figure:
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
    return fig

def create_sector_score_gauge(sector_score: float) -> go.Figure:
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
    return fig

def create_sector_donut_chart(mix: Dict[str, float]) -> Optional[go.Figure]:
    if not mix: return None
    fig = go.Figure(data=[go.Pie(
        labels=list(mix.keys()), values=list(mix.values()), hole=.45,
        textinfo='label+percent', hoverinfo='label+percent+value'
    )])
    fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def render_fund_overlap_venn(client: Dict, df_mat: pd.DataFrame) -> plt.Figure:
    funds = client.get("funds", [])
    num_funds = len(funds)
    fund_names = [f["fundCode"] for f in funds]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plt.style.use('seaborn-v0_8-whitegrid')

    if num_funds not in [2, 3]:
        ax.text(0.5, 0.5, "Venn diagram is only available for 2 or 3 funds.",
                ha='center', va='center', fontsize=12, wrap=True)
        ax.axis('off')
        return fig

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
                except ValueError:
                    pass
    ax.set_title("Fund Overlap Analysis", fontsize=14, pad=6)
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: STOCK EVALUATOR SCHEMA MODELS & RULE RULES
# ──────────────────────────────────────────────────────────────────────────────
class StockParameters(BaseModel):
    priceEarningsRatio: float
    earningsPerShare: float
    dividendYield: float
    marketCap: float
    debtToEquityRatio: float
    returnOnEquity: float
    returnOnAssets: float
    currentRatio: float
    quickRatio: float
    bookValuePerShare: float

class StockInput(BaseModel):
    stockSymbol: str = Field(..., min_length=1)
    parameters: StockParameters

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

def build_dynamic_summary(stock: StockInput) -> str:
    s = stock.stockSymbol
    p = stock.parameters
    pe_lbl = "cheap" if p.priceEarningsRatio < 15 else ("fairly valued" if p.priceEarningsRatio <= 30 else "expensive")
    de_lbl = "low" if p.debtToEquityRatio < 0.5 else ("balanced" if p.debtToEquityRatio <= 1.5 else "high")
    roe_lbl = "excellent" if p.returnOnEquity * 100 > 15 else ("healthy" if p.returnOnEquity * 100 >= 8 else "weak")
    roa_lbl = "very strong" if p.returnOnAssets * 100 > 10 else ("solid" if p.returnOnAssets * 100 >= 5 else "modest")
    
    if p.currentRatio < 1 or p.quickRatio < 1: liq_lbl = "tight"
    elif p.currentRatio <= 2 and p.quickRatio <= 2: liq_lbl = "healthy"
    else: liq_lbl = "very high"
    
    score = 0
    score += 2 if pe_lbl == "cheap" else (1 if pe_lbl == "fairly valued" else 0)
    score += 2 if roe_lbl == "excellent" else (1 if roe_lbl == "healthy" else 0)
    score += 2 if de_lbl == "low" else (1 if de_lbl == "balanced" else 0)
    score += 1 if liq_lbl == "healthy" else (0 if liq_lbl == "very high" else -1)
    stance = "compelling" if score >= 6 else ("balanced" if score >= 4 else "cautious")

    return (
        f"**{s} — At-a-glance**\n"
        f"- **Valuation:** P/E **{p.priceEarningsRatio:.1f}** ({pe_lbl}); EPS **{p.earningsPerShare:.2f}**; "
        f"Dividend yield **{p.dividendYield:.2f}%**.\n"
        f"- **Profitability:** ROE **{p.returnOnEquity*100:.1f}%** ({roe_lbl}), "
        f"ROA **{p.returnOnAssets*100:.1f}%** ({roa_lbl}).\n"
        f"- **Leverage & Liquidity:** D/E **{p.debtToEquityRatio:.2f}** ({de_lbl}); "
        f"Current **{p.currentRatio:.2f}**, Quick **{p.quickRatio:.2f}** ({liq_lbl}).\n"
        f"- **Scale & Book:** Market cap ${p.marketCap/1_000_000_000_000:,.2f}T; BVPS **{p.bookValuePerShare:.2f}**.\n\n"
        f"**Interpretation:** With valuation {pe_lbl}, profitability {roe_lbl.lower()} ROE and {roa_lbl.lower()} ROA, "
        f"and leverage {de_lbl}, overall the setup looks **{stance}**. "
        f"Income appeal is {'limited' if p.dividendYield < 1 else ('balanced' if p.dividendYield <= 3 else 'high')} "
        f"at **{p.dividendYield:.2f}%**. Consider peer comparisons and growth drivers before acting."
    )

def deterministic_evaluate(stock: StockInput) -> Dict[str, Any]:
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
        "bookValuePerShare": f"BVPS {p.bookValuePerShare:g}: per-share net assets.",
    }
    return {
        "stockSymbol": stock.stockSymbol,
        "feedback": feedback,
        "summary": build_dynamic_summary(stock),
    }