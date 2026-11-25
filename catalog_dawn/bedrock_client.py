"""Amazon Bedrock Converse API client supporting Nova Micro and Nova Lite with latency & cost telemetry."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

NOVA_MICRO_MODEL = "us.amazon.nova-micro-v1:0"
NOVA_LITE_MODEL = "us.amazon.nova-lite-v1:0"

# On-demand US cross-region inference pricing per 1M tokens (USD)
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    NOVA_MICRO_MODEL: {"input": 0.035, "output": 0.140},
    NOVA_LITE_MODEL: {"input": 0.060, "output": 0.240},
}


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model output, stripping optional markdown fences."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def compute_call_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    rates = MODEL_PRICING_PER_1M.get(model_id, MODEL_PRICING_PER_1M[NOVA_LITE_MODEL])
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0
    return round(cost, 8)


def converse_bedrock(
    system_prompt: str,
    user_prompt: str,
    model_id: Optional[str] = None,
    max_tokens: int = 420,
    temperature: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Call Amazon Bedrock Converse API via boto3 (or AWS CLI fallback) and return structured metrics."""
    if os.environ.get("CATALOG_DAWN_OFFLINE_LLM") == "1":
        return None

    chosen_model = model_id or os.environ.get("CATALOG_DAWN_BEDROCK_MODEL", NOVA_LITE_MODEL)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
    t0 = time.perf_counter()

    # Path 1: boto3 (AWS Lambda execution role or local AWS_PROFILE)
    try:
        import boto3  # type: ignore

        if os.environ.get("AWS_EXECUTION_ENV"):
            session = boto3.Session(region_name=region)
        else:
            profile = os.environ.get("CATALOG_DAWN_AWS_PROFILE") or os.environ.get("AWS_PROFILE")
            if profile:
                session = boto3.Session(profile_name=profile, region_name=region)
            else:
                session = boto3.Session(region_name=region)

        client = session.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=chosen_model,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        server_ms = float((resp.get("metrics") or {}).get("latencyMs") or round_trip_ms)
        usage = resp.get("usage") or {}
        in_tok = int(usage.get("inputTokens", 0))
        out_tok = int(usage.get("outputTokens", 0))
        tot_tok = int(usage.get("totalTokens", in_tok + out_tok))
        content_blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(str(b.get("text", "")) for b in content_blocks if isinstance(b, dict)).strip()
        return {
            "text": text,
            "parsed_json": extract_json_object(text),
            "model_id": chosen_model,
            "region": region,
            "latency_ms": round(server_ms, 2),
            "round_trip_ms": round_trip_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "cost_usd": compute_call_cost_usd(chosen_model, in_tok, out_tok),
        }
    except Exception:
        pass

    # Path 2: Local AWS CLI fallback if boto3 session was unavailable
    aws_bin = shutil.which("aws")
    if not aws_bin or os.environ.get("AWS_EXECUTION_ENV"):
        return None

    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = os.environ.get("AWS_PROFILE", "portfolio")
    env["AWS_DEFAULT_REGION"] = region

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        chosen_model,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": max_tokens, "temperature": temperature}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=20.0, check=True)
        resp = json.loads(proc.stdout)
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        server_ms = float((resp.get("metrics") or {}).get("latencyMs") or round_trip_ms)
        usage = resp.get("usage") or {}
        in_tok = int(usage.get("inputTokens", 0))
        out_tok = int(usage.get("outputTokens", 0))
        tot_tok = int(usage.get("totalTokens", in_tok + out_tok))
        content_blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(str(b.get("text", "")) for b in content_blocks if isinstance(b, dict)).strip()
        return {
            "text": text,
            "parsed_json": extract_json_object(text),
            "model_id": chosen_model,
            "region": region,
            "latency_ms": round(server_ms, 2),
            "round_trip_ms": round_trip_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "cost_usd": compute_call_cost_usd(chosen_model, in_tok, out_tok),
        }
    except Exception:
        return None
