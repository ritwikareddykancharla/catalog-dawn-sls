"""HTTP Server & AWS Lambda Function URL Handler for CatalogDawn (`/api/*` behind CloudFront)."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .audit_store import CatalogAuditStore
from .bedrock_client import NOVA_LITE_MODEL, NOVA_MICRO_MODEL
from .entity_resolver import (
    CANONICAL_ASIN_CATALOG,
    SAMPLE_ENTITY_RESOLUTION_CASES,
    VariantSafeEntityResolver,
)
from .offer_extractor import SAMPLE_MERCHANT_PAYLOADS, extract_merchant_offer
from .ontology_normalizer import SAMPLE_NORMALIZATION_RECORDS, normalize_and_reconcile_record
from .self_learning_loop import (
    RETURN_SIGNAL_CLUSTERS,
    compute_epoch_progression,
    synthesize_and_validate_rule,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

AUDIT_STORE = CatalogAuditStore()
ENTITY_RESOLVER = VariantSafeEntityResolver()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _load_benchmarks() -> Dict[str, Any]:
    if BENCHMARK_JSON_PATH.is_file():
        try:
            return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"status": "benchmarks_not_generated_yet"}


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Shared API router for local Python dev server and AWS Lambda `/api/*` CloudFront origin."""
    try:
        if method == "GET" and path == "/api/health":
            chain = AUDIT_STORE.verify_chain_integrity()
            return 200, {
                "status": "ok",
                "service": "catalog-dawn-sls",
                "delivery": os.environ.get("CATALOG_DAWN_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "bedrock_models": [NOVA_MICRO_MODEL, NOVA_LITE_MODEL],
                "python_version": platform.python_version(),
                "canonical_asins_indexed": len(CANONICAL_ASIN_CATALOG),
                "promoted_rules_active": len(AUDIT_STORE.list_promoted_rules()),
                "sqlite_audit_chain_valid": chain["valid"],
                "sqlite_audit_rows": chain["checked_rows"],
                "request_id": request_id,
            }

        if method == "GET" and path == "/api/benchmarks":
            bench = _load_benchmarks()
            return 200, {
                "service": "catalog-dawn-sls",
                "benchmarks": bench,
                "promoted_rules": AUDIT_STORE.list_promoted_rules(),
                "audit_chain": AUDIT_STORE.verify_chain_integrity(),
                "request_id": request_id,
            }

        if method == "GET" and path == "/api/state":
            return 200, {
                "service": "catalog-dawn-sls",
                "canonical_asin_count": len(CANONICAL_ASIN_CATALOG),
                "sample_offers": SAMPLE_MERCHANT_PAYLOADS,
                "sample_resolution_cases": SAMPLE_ENTITY_RESOLUTION_CASES,
                "sample_normalizations": SAMPLE_NORMALIZATION_RECORDS,
                "return_clusters": list(RETURN_SIGNAL_CLUSTERS.values()),
                "epoch_progression": compute_epoch_progression(
                    ["dual_voltage_mismatch", "powder_coat_dishwasher_contradiction", "missing_m10_thread_pitch"]
                ),
                "promoted_rules": AUDIT_STORE.list_promoted_rules(),
                "audit_events": AUDIT_STORE.list_recent_events(limit=20),
                "audit_chain": AUDIT_STORE.verify_chain_integrity(),
                "benchmarks": _load_benchmarks(),
                "request_id": request_id,
            }

        if method == "POST" and path == "/api/offers/ingest":
            raw_payload = str(body.get("raw_payload") or body.get("payload") or "").strip()
            if not raw_payload:
                return 400, {"error": "raw_payload is required"}
            format_hint = str(body.get("format_hint") or "auto").strip()
            merchant_id = str(body.get("merchant_id") or "ext-merch-live").strip()
            ref_med = body.get("reference_median_usd")
            ref_val = float(ref_med) if ref_med is not None and str(ref_med).strip() != "" else None

            res = extract_merchant_offer(
                raw_payload=raw_payload,
                format_hint=format_hint,
                merchant_id=merchant_id,
                reference_median_usd=ref_val,
            )
            receipt = AUDIT_STORE.record_event(
                stage="OFFER_INGEST",
                entity_key=res["gtin_validation"].get("cleaned") or res["offer"]["title"][:42],
                verdict=res["verdict"],
                duration_ms=res["extract_ms"],
                payload={"merchant_id": merchant_id, "guard_flags": res["guard_flags"]},
            )
            res["audit_receipt"] = receipt
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and path == "/api/asin/resolve":
            title = str(body.get("title") or "").strip()
            if not title:
                return 400, {"error": "title is required"}
            brand = str(body.get("brand") or "").strip() or None
            mpn = str(body.get("mpn") or "").strip() or None
            gtin = str(body.get("gtin") or "").strip() or None
            explicit_attrs = body.get("attributes") if isinstance(body.get("attributes"), dict) else None
            use_bedrock = bool(body.get("use_bedrock_tiebreaker", False))

            res = ENTITY_RESOLVER.resolve_offer(
                title=title,
                brand=brand,
                mpn=mpn,
                gtin=gtin,
                explicit_attrs=explicit_attrs,
                use_bedrock_tiebreaker=use_bedrock,
            )
            receipt = AUDIT_STORE.record_event(
                stage="ASIN_RESOLVE",
                entity_key=res.get("matched_asin") or res.get("parent_asin") or title[:42],
                verdict=res["decision"],
                duration_ms=res["resolve_ms"],
                payload={"blocking_key": res["blocking_key"], "veto_count": len(res.get("vetoes_triggered", []))},
            )
            res["audit_receipt"] = receipt
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and path == "/api/catalog/normalize":
            record = body.get("record") if isinstance(body.get("record"), dict) else body
            if not record.get("title") and not record.get("claims"):
                return 400, {"error": "product record with title or claims is required"}
            use_bedrock = bool(body.get("use_bedrock_cascade", True))

            res = normalize_and_reconcile_record(record=record, use_bedrock_cascade=use_bedrock)
            verdict = "CONTRADICTION_RESOLVED" if res["contradiction_detected"] else "NORMALIZED_CLEAN"
            receipt = AUDIT_STORE.record_event(
                stage="ONTOLOGY_NORMALIZE",
                entity_key=f"{res['asin']}:{res['locale']}",
                verdict=verdict,
                duration_ms=res["normalize_ms"],
                payload={
                    "cascade_tier": res["cascade_tier"],
                    "contradictions": len(res["contradictions"]),
                    "rufus_after": res["rufus_readiness_after"],
                },
            )
            res["audit_receipt"] = receipt
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and path == "/api/self-learn/propose":
            cluster_id = str(body.get("cluster_id") or "CLUS-VOLT-110").strip()
            custom_notes = body.get("custom_return_notes")
            notes_list = [str(x) for x in custom_notes] if isinstance(custom_notes, list) else None
            use_bedrock = bool(body.get("use_bedrock", True))

            res = synthesize_and_validate_rule(
                cluster_id=cluster_id,
                custom_return_notes=notes_list,
                use_bedrock=use_bedrock,
            )
            rule = res["candidate_rule"]
            if res["holdout_evaluation"]["passes_promotion_gate"]:
                AUDIT_STORE.save_promoted_rule(rule)

            receipt = AUDIT_STORE.record_event(
                stage="SLS_RULE_PROPOSE",
                entity_key=rule["rule_id"],
                verdict=rule["status"],
                duration_ms=res["total_ms"],
                payload={
                    "cluster_id": cluster_id,
                    "holdout_precision": rule["holdout_precision"],
                    "false_positive_rate": rule["false_positive_rate"],
                },
            )
            res["promoted_rules_active"] = AUDIT_STORE.list_promoted_rules()
            res["audit_receipt"] = receipt
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and path == "/api/reset":
            AUDIT_STORE.reset()
            return 200, {
                "status": "reset_complete",
                "service": "catalog-dawn-sls",
                "request_id": request_id,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalCatalogDawnHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes() if candidate.is_file() else b"<h1>CatalogDawn UI building...</h1>"
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3099) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalCatalogDawnHandler)
    print(f"CatalogDawn Studio running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3099"))
    run_local_server(port=port_num)
