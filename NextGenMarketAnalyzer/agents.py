import json
import warnings
from typing import Dict, Any
from openai import OpenAI

def run_three_agent_debate(metrics: Dict[str, Any], api_key: str) -> Dict[str, str]:
    # Initialize the standard client using the Gemini compatibility endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    model_name = "gemini-2.5-flash"
    metrics_json = json.dumps(metrics, indent=2)

    try:
        # 1. Get Risk Analyst Verdict
        risk_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative financial analyst. Write a compact verdict (80–120 words) covering concentration risks, sector imbalances, and potential drawdowns. Offer 1–2 cautionary steps."
                },
                {
                    "role": "user",
                    "content": f"Portfolio metrics:\n```json\n{metrics_json}\n```\n\nWrite your risk verdict now."
                }
            ],
            temperature=0.35,
            timeout=120
        )
        risk_verdict = risk_response.choices[0].message.content.strip()

        # 2. Get Growth Strategist Verdict
        growth_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a growth-focused strategist. Write a compact verdict (80–120 words) highlighting opportunities, underweighted growth areas, and 2 practical steps to improve upside."
                },
                {
                    "role": "user",
                    "content": f"Portfolio metrics:\n```json\n{metrics_json}\n```\n\nWrite your growth verdict now."
                }
            ],
            temperature=0.35,
            timeout=120
        )
        growth_verdict = growth_response.choices[0].message.content.strip()

        # 3. Get Lead Analyst Final Note
        lead_msg = (
            f"METRICS:\n```json\n{metrics_json}\n```\n\n"
            f"RISK ANALYST NOTE:\n{risk_verdict}\n\n"
            f"GROWTH STRATEGIST NOTE:\n{growth_verdict}\n\n"
            "Write the final advisory note."
        )

        lead_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are the lead analyst. Given two expert notes (risk and growth) and portfolio metrics, write a final advisory note (120–180 words) balancing risk and opportunity, summarizing metrics, and providing 2-3 steps."
                },
                {
                    "role": "user",
                    "content": lead_msg
                }
            ],
            temperature=0.35,
            timeout=120
        )
        lead_verdict = lead_response.choices[0].message.content.strip()

        return {
            "risk_verdict": risk_verdict,
            "growth_verdict": growth_verdict,
            "lead_verdict": lead_verdict,
            "final_note": lead_verdict,
        }

    except Exception as e:
        warnings.warn(f"Gemini multi-agent simulation failure: {e}")
        raise e
