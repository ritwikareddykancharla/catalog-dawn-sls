"""Cross-Marketplace (US / DE / JP) Attribute Normalizer & Multi-Seller Contradiction Resolver."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .bedrock_client import NOVA_LITE_MODEL, NOVA_MICRO_MODEL, converse_bedrock

SOURCE_AUTHORITY_WEIGHTS: Dict[str, float] = {
    "OEM_MANUAL": 1.00,
    "OEM_SPEC_SHEET": 0.98,
    "VERIFIED_RETURN_CLUSTER": 0.92,
    "BRAND_REGISTRY": 0.88,
    "MERCHANT_JSON_LD": 0.65,
    "SELLER_STRUCTURED_ATTR": 0.50,
    "SELLER_PROSE_BULLET": 0.45,
}

CATEGORY_REQUIRED_SCHEMAS: Dict[str, List[str]] = {
    "wall_charger": ["wattage_w", "input_voltage_v", "weight_g", "usb_c_ports"],
    "insulated_tumbler": ["volume_ml", "dishwasher_safe", "weight_g", "material"],
    "power_bank": ["battery_energy_wh", "wattage_w", "faa_carry_on_compliant", "weight_g"],
    "stand_mixer": ["wattage_w", "input_voltage_v", "volume_ml", "dishwasher_safe"],
}


def _parse_locale_float(raw_num: str) -> float:
    s = raw_num.strip()
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) in (1, 2, 3) and len(parts[0]) <= 3:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    return float(s)


def canonicalize_attribute_value(attr_name: str, raw_value: Any, locale: str = "US") -> Tuple[Optional[str], str]:
    """Deterministically normalize messy US / DE / JP attribute strings to canonical Amazon Catalog units."""
    if raw_value is None:
        return None, "empty"
    s = str(raw_value).strip()
    if not s:
        return None, "empty"
    low = s.lower()

    # 1. Wattage -> canonical "<int/float> W"
    if attr_name == "wattage_w":
        m_kw = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:kw|kilowatt|キロワット)", low)
        if m_kw:
            val_w = _parse_locale_float(m_kw.group(1)) * 1000.0
            fmt = f"{int(round(val_w))} W" if abs(val_w - round(val_w)) < 0.05 else f"{val_w:.1f} W"
            return fmt, "kw_to_w"
        m_w = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:w|watt|watts|ワット)\b", low)
        if m_w:
            val_w = _parse_locale_float(m_w.group(1))
            fmt = f"{int(round(val_w))} W" if abs(val_w - round(val_w)) < 0.05 else f"{val_w:.1f} W"
            return fmt, "w_canonical"

    # 2. Volume -> canonical "<int> ml"
    if attr_name == "volume_ml":
        m_l = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:l|liter|litre|リットル)\b", low)
        if m_l and "ml" not in low and "fl" not in low:
            ml = int(round(_parse_locale_float(m_l.group(1)) * 1000.0))
            return f"{ml} ml", "liter_to_ml"
        m_ml = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ml|milliliter|ミリリットル)", low)
        if m_ml:
            ml = int(round(_parse_locale_float(m_ml.group(1))))
            return f"{ml} ml", "ml_canonical"
        m_floz = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:fl\.?\s*oz|fluid\s*ounces?|oz)\b", low)
        if m_floz:
            # 1 US fl oz = 29.5735 ml (16.9 fl oz -> 500 ml; 40 fl oz -> 1183 ml)
            raw_oz = _parse_locale_float(m_floz.group(1))
            ml_exact = raw_oz * 29.5735
            # Snap well-known commercial bottle equivalences cleanly (16.9 fl oz == 500 ml)
            ml = 500 if abs(ml_exact - 500.0) <= 2.0 else int(round(ml_exact))
            return f"{ml} ml", "fl_oz_to_ml"
        m_qt = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:quart|qt)s?\b", low)
        if m_qt:
            ml = int(round(_parse_locale_float(m_qt.group(1)) * 946.353))
            return f"{ml} ml", "quart_to_ml"

    # 3. Weight -> canonical "<int> g"
    if attr_name == "weight_g":
        m_kg = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:kg|kilogram|キログラム)", low)
        if m_kg:
            g = int(round(_parse_locale_float(m_kg.group(1)) * 1000.0))
            return f"{g} g", "kg_to_g"
        m_lb = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:lb|lbs|pound|pounds)\b", low)
        if m_lb:
            g = int(round(_parse_locale_float(m_lb.group(1)) * 453.592))
            return f"{g} g", "lb_to_g"
        m_oz = re.search(r"(\d+(?:[.,]\d+)?)\s*oz\b", low)
        if m_oz and "fl" not in low:
            g = int(round(_parse_locale_float(m_oz.group(1)) * 28.3495))
            return f"{g} g", "oz_to_g"
        m_g = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:g|gram|grams|gramm|グラム)\b", low)
        if m_g:
            g = int(round(_parse_locale_float(m_g.group(1))))
            return f"{g} g", "g_canonical"

    # 4. Dishwasher Safe -> canonical "true" / "false"
    if attr_name == "dishwasher_safe":
        neg_markers = [
            "hand wash only",
            "hand-wash only",
            "hand wash recommended",
            "not dishwasher safe",
            "do not place in dishwasher",
            "dishwasher safe: no",
            "spülmaschinenfest: nein",
            "nur handwäsche",
            "nicht spülmaschinengeeignet",
            "食洗機不可",
            "手洗いのみ",
            "false",
            "no",
            "nein",
        ]
        pos_markers = [
            "dishwasher safe: yes",
            "dishwasher safe",
            "top-rack dishwasher safe",
            "spülmaschinenfest: ja",
            "spülmaschinenfest",
            "食洗機対応",
            "食洗機可",
            "true",
            "yes",
            "ja",
        ]
        for nm in neg_markers:
            if nm in low:
                return "false", "multilingual_boolean_neg"
        for pm in pos_markers:
            if pm in low:
                return "true", "multilingual_boolean_pos"

    # 5. Input Voltage -> canonical "100-240 V", "110-120 V", "220-240 V", "100 V"
    if attr_name == "input_voltage_v":
        if re.search(r"100\s*[-–~/]\s*240\s*v|dual[\s-]*voltage|ユニバーサル電圧|universalspannung", low):
            return "100-240 V", "dual_voltage"
        if re.search(r"\b(?:220|230|240)\s*v\b", low):
            return "220-240 V", "eu_voltage"
        if re.search(r"\b(?:110|115|120)\s*v\b", low):
            return "110-120 V", "us_voltage"
        if re.search(r"\b100\s*v\b", low):
            return "100 V", "jp_voltage"

    # 6. Battery Energy (Wh) from Wh or (mAh + nominal V)
    if attr_name == "battery_energy_wh":
        m_wh = re.search(r"(\d+(?:[.,]\d+)?)\s*wh\b", low)
        if m_wh:
            wh = round(_parse_locale_float(m_wh.group(1)), 1)
            return f"{wh} Wh", "wh_direct"
        m_mah = re.search(r"(\d{4,6})\s*mah\b", low)
        if m_mah:
            mah = float(m_mah.group(1))
            m_nom = re.search(r"(\d+[.,]\d+)\s*v\b", low)
            nom_v = _parse_locale_float(m_nom.group(1)) if m_nom else 3.7
            wh = round((mah * nom_v) / 1000.0, 1)
            return f"{wh} Wh", "mah_to_wh_conversion"

    if attr_name == "faa_carry_on_compliant":
        if low in ("true", "yes", "ja"):
            return "true", "bool_direct"
        if low in ("false", "no", "nein"):
            return "false", "bool_direct"

    if attr_name == "usb_c_ports":
        m_p = re.search(r"(\d+)\s*(?:x\s*)?(?:usb-c|type-c|ポート|anschlüsse)", low)
        if m_p:
            return str(int(m_p.group(1))), "port_count"
        if s.isdigit():
            return str(int(s)), "port_count"

    if attr_name == "material":
        if any(w in low for w in ("18/8", "stainless", "edelstahl", "ステンレス")):
            return "18/8 Stainless Steel", "material_ontology"
        if any(w in low for w in ("plastic", "tritan", "kunststoff")):
            return "BPA-Free Plastic", "material_ontology"

    return s, "passthrough"


def _verify_evidence_span(attr_name: str, val_str: str, full_source_text: str) -> bool:
    """Verify that an LLM-extracted attribute is grounded in source text (blocking hallucinated specs)."""
    low_src = full_source_text.lower()
    low_val = val_str.lower()
    if low_val in low_src:
        return True
    nums = re.findall(r"\d+(?:\.\d+)?", val_str)
    if nums:
        # Allow unit-converted numbers if either the canonical number or a convertible source unit is in text
        if any(n in low_src for n in nums):
            return True
        if attr_name == "volume_ml" and ("fl oz" in low_src or "oz" in low_src or "liter" in low_src or "qt" in low_src):
            return True
        if attr_name == "wattage_w" and ("kw" in low_src or "watt" in low_src or "ワット" in low_src):
            return True
        if attr_name == "weight_g" and ("lb" in low_src or "kg" in low_src or "oz" in low_src or "グラム" in low_src):
            return True
        if attr_name == "battery_energy_wh" and ("mah" in low_src or "wh" in low_src):
            return True
    if attr_name in ("dishwasher_safe", "faa_carry_on_compliant") and low_val in ("true", "false"):
        return any(
            k in low_src
            for k in ("dishwasher", "hand wash", "spülmaschine", "handwäsche", "食洗機", "手洗い", "wh", "mah", "faa")
        )
    if attr_name == "material" and "stainless" in low_val:
        return any(k in low_src for k in ("stainless", "18/8", "edelstahl", "ステンレス"))
    return False


def normalize_and_reconcile_record(
    record: Dict[str, Any],
    use_bedrock_cascade: bool = True,
) -> Dict[str, Any]:
    """Normalize multi-locale claims, detect cross-seller contradictions, and cascade missing specs to Bedrock."""
    t0 = time.perf_counter()
    asin = str(record.get("asin") or "B0UNKNOWN0")
    locale = str(record.get("locale") or "US").upper()
    category = str(record.get("category") or "insulated_tumbler")
    title = str(record.get("title") or "")
    bullets = [str(b) for b in (record.get("bullets") or [])]
    claims: List[Dict[str, Any]] = list(record.get("claims") or [])

    required_attrs = CATEGORY_REQUIRED_SCHEMAS.get(category, ["wattage_w", "weight_g"])
    full_prose = f"{title}\n" + "\n".join(bullets) + "\n" + "\n".join(str(c.get("raw_value", "")) for c in claims)

    # Group claims by attribute and canonicalize each claim
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for cl in claims:
        attr = str(cl.get("attribute") or "").strip()
        src = str(cl.get("source") or "SELLER_STRUCTURED_ATTR").upper()
        raw_val = cl.get("raw_value")
        canon_val, rule_used = canonicalize_attribute_value(attr, raw_val, locale=locale)
        if canon_val is None:
            continue
        weight = SOURCE_AUTHORITY_WEIGHTS.get(src, 0.50)
        grouped.setdefault(attr, []).append(
            {
                "source": src,
                "raw_value": raw_val,
                "canonical_value": canon_val,
                "rule_used": rule_used,
                "authority_weight": weight,
            }
        )

    # Also harvest deterministic values from prose bullets for missing attributes
    for req_attr in required_attrs:
        if req_attr not in grouped:
            canon_from_prose, rule_used = canonicalize_attribute_value(req_attr, full_prose, locale=locale)
            if canon_from_prose is not None and rule_used != "passthrough":
                grouped.setdefault(req_attr, []).append(
                    {
                        "source": "PROSE_DETERMINISTIC_PARSER",
                        "raw_value": "extracted from title/bullets",
                        "canonical_value": canon_from_prose,
                        "rule_used": rule_used,
                        "authority_weight": 0.72,
                    }
                )

    # Derive FAA carry-on compliance deterministically when battery_energy_wh is known
    if category == "power_bank" and "battery_energy_wh" in grouped and "faa_carry_on_compliant" not in grouped:
        wh_str = grouped["battery_energy_wh"][0]["canonical_value"]
        m_wh = re.search(r"(\d+(?:\.\d+)?)", wh_str)
        if m_wh:
            wh_val = float(m_wh.group(1))
            grouped["faa_carry_on_compliant"] = [
                {
                    "source": "ONTOLOGY_SAFETY_RULE",
                    "raw_value": f"{wh_val} Wh <= 100.0 Wh FAA threshold",
                    "canonical_value": "true" if wh_val <= 100.0 else "false",
                    "rule_used": "faa_100wh_limit",
                    "authority_weight": 0.99,
                }
            ]

    resolved_attributes: Dict[str, Any] = {}
    contradictions: List[Dict[str, Any]] = []

    for attr, attr_claims in grouped.items():
        by_val: Dict[str, float] = {}
        for c in attr_claims:
            v = c["canonical_value"]
            by_val[v] = max(by_val.get(v, 0.0), c["authority_weight"]) + 0.05

        ranked_vals = sorted(by_val.items(), key=lambda kv: kv[1], reverse=True)
        winner_val = ranked_vals[0][0]
        resolved_attributes[attr] = {
            "canonical_value": winner_val,
            "winning_source": max(
                [c for c in attr_claims if c["canonical_value"] == winner_val],
                key=lambda x: x["authority_weight"],
            )["source"],
            "evidence_count": len(attr_claims),
        }

        if len(ranked_vals) > 1:
            losing_vals = [v for v, _ in ranked_vals[1:]]
            contradictions.append(
                {
                    "attribute": attr,
                    "resolved_canonical_value": winner_val,
                    "conflicting_values": losing_vals,
                    "claims": attr_claims,
                    "action": "OVERRIDDEN_BY_AUTHORITY_PROVENANCE",
                }
            )

    missing_before_llm = [a for a in required_attrs if a not in resolved_attributes]
    cascade_tier = "TIER_0_DETERMINISTIC"
    bedrock_call: Optional[Dict[str, Any]] = None
    rejected_hallucinations: List[Dict[str, str]] = []

    if use_bedrock_cascade and (missing_before_llm or contradictions):
        # Tier 1: Nova Micro for missing attribute extraction; Tier 2: Nova Lite when contradictions exist
        chosen_model = NOVA_LITE_MODEL if contradictions else NOVA_MICRO_MODEL
        cascade_tier = "TIER_2_NOVA_LITE_RECONCILER" if contradictions else "TIER_1_NOVA_MICRO_EXTRACTOR"
        sys_prompt = (
            "You are an Amazon Catalog System Services normalization model. "
            "Extract ONLY facts explicitly supported by the product text or OEM spec claims. "
            "Never guess missing specs. Return strict JSON: "
            "{\"extracted_attributes\": {\"<attr>\": \"<value>\"}, \"reconciliation_note\": \"<1 concise sentence>\"}."
        )
        usr_prompt = json.dumps(
            {
                "asin": asin,
                "locale": locale,
                "category": category,
                "title": title,
                "bullets": bullets,
                "missing_attributes": missing_before_llm,
                "detected_contradictions": contradictions,
            },
            ensure_ascii=False,
        )
        bedrock_call = converse_bedrock(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            model_id=chosen_model,
            max_tokens=280,
            temperature=0.0,
        )
        if bedrock_call and isinstance(bedrock_call.get("parsed_json"), dict):
            ext = bedrock_call["parsed_json"].get("extracted_attributes") or {}
            if isinstance(ext, dict):
                for k, raw_v in ext.items():
                    if k in missing_before_llm and raw_v is not None and str(raw_v).strip():
                        canon_v, r_used = canonicalize_attribute_value(k, raw_v, locale=locale)
                        if canon_v and _verify_evidence_span(k, canon_v, full_prose):
                            resolved_attributes[k] = {
                                "canonical_value": canon_v,
                                "winning_source": f"BEDROCK_GROUNDED ({chosen_model})",
                                "evidence_count": 1,
                                "rule_used": r_used,
                            }
                        elif canon_v:
                            rejected_hallucinations.append(
                                {"attribute": k, "proposed_value": str(raw_v), "reason": "UNGROUNDED_IN_SOURCE_SPAN"}
                            )

    raw_seller_filled = len({str(c.get("attribute")) for c in claims if str(c.get("source", "")).startswith("SELLER")})
    before_readiness = int(round(100.0 * min(len(required_attrs), raw_seller_filled) / max(1, len(required_attrs))))
    if contradictions:
        before_readiness = max(15, before_readiness - 35)

    after_filled = sum(1 for a in required_attrs if a in resolved_attributes)
    after_readiness = int(round(100.0 * after_filled / max(1, len(required_attrs))))

    return {
        "asin": asin,
        "locale": locale,
        "category": category,
        "title": title,
        "cascade_tier": cascade_tier,
        "contradiction_detected": len(contradictions) > 0,
        "contradictions": contradictions,
        "resolved_attributes": resolved_attributes,
        "required_schema": required_attrs,
        "missing_attributes_remaining": [a for a in required_attrs if a not in resolved_attributes],
        "rejected_ungrounded_attributes": rejected_hallucinations,
        "rufus_readiness_before": before_readiness,
        "rufus_readiness_after": after_readiness,
        "bedrock_telemetry": {
            "invoked": bedrock_call is not None,
            "model_id": bedrock_call["model_id"] if bedrock_call else None,
            "latency_ms": bedrock_call["latency_ms"] if bedrock_call else 0.0,
            "total_tokens": bedrock_call["total_tokens"] if bedrock_call else 0,
            "cost_usd": bedrock_call["cost_usd"] if bedrock_call else 0.0,
            "reconciliation_note": (bedrock_call.get("parsed_json") or {}).get("reconciliation_note")
            if bedrock_call
            else "Resolved deterministically via multi-locale unit & provenance rules.",
        },
        "normalize_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


import json  # placed cleanly at module level for json.dumps in prompt

SAMPLE_NORMALIZATION_RECORDS: List[Dict[str, Any]] = [
    {
        "sample_id": "NORM-CONTRADICT-US-01",
        "label": "Multi-Seller Contradiction (US): Seller Claims 'Dishwasher Safe: Yes' vs OEM + 3 Returns 'Hand Wash Only'",
        "asin": "B0B3N5M78Q",
        "locale": "US",
        "category": "insulated_tumbler",
        "title": "Stanley Quencher H2.0 FlowState 16.9 fl oz Vacuum Insulated Tumbler Rose Quartz",
        "bullets": [
            "Constructed from recycled 18/8 stainless steel with double-wall vacuum insulation",
            "Item weight: 1.1 lb empty; powder-coated matte finish",
        ],
        "claims": [
            {"attribute": "dishwasher_safe", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "Dishwasher Safe: Yes"},
            {"attribute": "dishwasher_safe", "source": "OEM_MANUAL", "raw_value": "Hand Wash Only - powder coat may dull in commercial dishwashers"},
            {"attribute": "dishwasher_safe", "source": "VERIFIED_RETURN_CLUSTER", "raw_value": "Hand wash only (3 buyer returns reported peeling finish after dishwasher)"},
            {"attribute": "volume_ml", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "16.9 fl oz"},
        ],
    },
    {
        "sample_id": "NORM-LOCALE-DE-02",
        "label": "German Marketplace (DE): '0,065 kW', '0,18 kg', Comma Decimals & Voltage Contradiction",
        "asin": "B0C4YTB56H",
        "locale": "DE",
        "category": "wall_charger",
        "title": "Anker Prime 0,065 kW GaN USB-C Schnellladegerät 3 Anschlüsse Schwarz",
        "bullets": [
            "Kompaktes Reisenetzteil mit 3x USB-C Anschlüssen für MacBook und Smartphone",
            "Eingangsspannung laut Typenschild: 100-240 V Universalspannung, Gewicht: 0,18 kg",
        ],
        "claims": [
            {"attribute": "wattage_w", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "0,065 kW"},
            {"attribute": "input_voltage_v", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "230 V"},
            {"attribute": "input_voltage_v", "source": "OEM_SPEC_SHEET", "raw_value": "100-240 V~ 50/60Hz"},
            {"attribute": "weight_g", "source": "MERCHANT_JSON_LD", "raw_value": "0,18 kg"},
        ],
    },
    {
        "sample_id": "NORM-LOCALE-JP-03",
        "label": "Japanese Marketplace (JP): '65ワット', '20000mAh @ 3.7V' -> Wh & FAA Carry-On Derivation",
        "asin": "B09ANKERJP1",
        "locale": "JP",
        "category": "power_bank",
        "title": "Anker PowerCore モバイルバッテリー 20000mAh 65ワット PD急速充電 ブラック",
        "bullets": [
            "大容量 20000mAh (3.7V リチウムイオン電池内蔵)、最大出力 65ワット USB-C PD対応",
            "本体重量 340グラム、機内持ち込み規定 (100Wh以下) 適合モデル",
        ],
        "claims": [
            {"attribute": "wattage_w", "source": "MERCHANT_JSON_LD", "raw_value": "65ワット"},
            {"attribute": "battery_energy_wh", "source": "OEM_SPEC_SHEET", "raw_value": "20000 mAh 3.7V"},
            {"attribute": "weight_g", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "340グラム"},
        ],
    },
]
