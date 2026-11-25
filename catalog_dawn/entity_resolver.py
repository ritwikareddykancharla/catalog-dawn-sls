"""Variant-Safe ASIN Entity Matcher: Blocking Keys + Character-Shingle MinHash + Hard Variant Vetoes."""
from __future__ import annotations

import re
import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

from .bedrock_client import NOVA_MICRO_MODEL, converse_bedrock
from .offer_extractor import compute_gs1_check_digit, extract_variant_axes, validate_gtin_check_digit


def _make_valid_gtin14(body13: str) -> str:
    """Build a valid 14-digit GTIN from a 13-digit body string."""
    b = re.sub(r"\D", "", body13).zfill(13)[:13]
    return f"{b}{compute_gs1_check_digit(b)}"


CANONICAL_ASIN_CATALOG: List[Dict[str, Any]] = [
    # Family 1: Samsung T7 Shield Portable SSD (1TB / 2TB, Blue / Black)
    {
        "asin": "B0BHZQGN26",
        "parent_asin": "B0BHZPARENT1",
        "title": "Samsung T7 Shield 1TB USB 3.2 Gen 2 Portable SSD Blue (MU-PE1T0R)",
        "brand": "Samsung",
        "product_type": "portable_ssd",
        "model_family": "t7 shield",
        "mpn": "MU-PE1T0R",
        "gtin14": "00887276646077",
        "pack_count": 1,
        "storage_gb": 1000,
        "voltage_v": None,
        "color": "blue",
        "size": None,
        "median_price_usd": 99.99,
    },
    {
        "asin": "B0BHZT5S12",
        "parent_asin": "B0BHZPARENT1",
        "title": "Samsung T7 Shield 2TB USB 3.2 Gen 2 Portable SSD Blue (MU-PE2T0R)",
        "brand": "Samsung",
        "product_type": "portable_ssd",
        "model_family": "t7 shield",
        "mpn": "MU-PE2T0R",
        "gtin14": "00887276646084",
        "pack_count": 1,
        "storage_gb": 2000,
        "voltage_v": None,
        "color": "blue",
        "size": None,
        "median_price_usd": 169.99,
    },
    {
        "asin": "B0BHZ99BLK",
        "parent_asin": "B0BHZPARENT1",
        "title": "Samsung T7 Shield 2TB USB 3.2 Gen 2 Portable SSD Black (MU-PE2T0S)",
        "brand": "Samsung",
        "product_type": "portable_ssd",
        "model_family": "t7 shield",
        "mpn": "MU-PE2T0S",
        "gtin14": _make_valid_gtin14("0088727664609"),
        "pack_count": 1,
        "storage_gb": 2000,
        "voltage_v": None,
        "color": "black",
        "size": None,
        "median_price_usd": 169.99,
    },
    # Family 2: Anker Prime 67W USB-C GaN Wall Charger (1-Pack vs 2-Pack, Black vs White)
    {
        "asin": "B0C4YTB56H",
        "parent_asin": "B0C4YPARENT2",
        "title": "Anker Prime 67W USB-C GaN Wall Charger 3-Port Compact Fast Charger Black (1-Pack)",
        "brand": "Anker",
        "product_type": "wall_charger",
        "model_family": "prime 67w gan",
        "mpn": "A2669-1PK",
        "gtin14": "00194644127480",
        "pack_count": 1,
        "storage_gb": None,
        "voltage_v": "100-240V",
        "color": "black",
        "size": None,
        "median_price_usd": 54.99,
    },
    {
        "asin": "B0C4YTB52P",
        "parent_asin": "B0C4YPARENT2",
        "title": "Anker Prime 67W USB-C GaN Wall Charger 3-Port Compact Fast Charger Black (2-Pack)",
        "brand": "Anker",
        "product_type": "wall_charger",
        "model_family": "prime 67w gan",
        "mpn": "A2669-2PK",
        "gtin14": "00194644127497",
        "pack_count": 2,
        "storage_gb": None,
        "voltage_v": "100-240V",
        "color": "black",
        "size": None,
        "median_price_usd": 99.99,
    },
    # Family 3: Brita Standard Water Pitcher Replacement Filters (1-Pack, 3-Pack, 6-Pack, 12-Pack)
    {
        "asin": "B00006IV0P",
        "parent_asin": "B0000BRITA00",
        "title": "Brita Standard Water Pitcher and Dispenser Replacement Filters BPA-Free White (3-Pack)",
        "brand": "Brita",
        "product_type": "water_filter",
        "model_family": "standard pitcher filter",
        "mpn": "OB03-3PK",
        "gtin14": _make_valid_gtin14("0006025836214"),
        "pack_count": 3,
        "storage_gb": None,
        "voltage_v": None,
        "color": "white",
        "size": None,
        "median_price_usd": 18.99,
    },
    {
        "asin": "B00004SU18",
        "parent_asin": "B0000BRITA00",
        "title": "Brita Standard Water Pitcher and Dispenser Replacement Filters BPA-Free White (6-Pack)",
        "brand": "Brita",
        "product_type": "water_filter",
        "model_family": "standard pitcher filter",
        "mpn": "OB03-6PK",
        "gtin14": "00060258362169",
        "pack_count": 6,
        "storage_gb": None,
        "voltage_v": None,
        "color": "white",
        "size": None,
        "median_price_usd": 34.99,
    },
    {
        "asin": "B09X1BRITA12",
        "parent_asin": "B0000BRITA00",
        "title": "Brita Standard Water Pitcher and Dispenser Replacement Filters BPA-Free White (12-Pack)",
        "brand": "Brita",
        "product_type": "water_filter",
        "model_family": "standard pitcher filter",
        "mpn": "OB03-12PK",
        "gtin14": _make_valid_gtin14("0006025836218"),
        "pack_count": 12,
        "storage_gb": None,
        "voltage_v": None,
        "color": "white",
        "size": None,
        "median_price_usd": 62.99,
    },
    # Family 4: Stanley Quencher H2.0 Tumbler (30oz vs 40oz, Rose Quartz vs Black)
    {
        "asin": "B0B3N5M78Q",
        "parent_asin": "B0B3NSTANLEY",
        "title": "Stanley Quencher H2.0 FlowState Stainless Steel Vacuum Insulated Tumbler 40 oz Rose Quartz (1-Pack)",
        "brand": "Stanley",
        "product_type": "insulated_tumbler",
        "model_family": "quencher h2.0 flowstate",
        "mpn": "10-10824-040",
        "gtin14": _make_valid_gtin14("0004160435284"),
        "pack_count": 1,
        "storage_gb": None,
        "voltage_v": None,
        "color": "rose_quartz",
        "size": "40OZ",
        "median_price_usd": 45.00,
    },
    {
        "asin": "B0B3N5M30Q",
        "parent_asin": "B0B3NSTANLEY",
        "title": "Stanley Quencher H2.0 FlowState Stainless Steel Vacuum Insulated Tumbler 30 oz Rose Quartz (1-Pack)",
        "brand": "Stanley",
        "product_type": "insulated_tumbler",
        "model_family": "quencher h2.0 flowstate",
        "mpn": "10-10827-030",
        "gtin14": _make_valid_gtin14("0004160435285"),
        "pack_count": 1,
        "storage_gb": None,
        "voltage_v": None,
        "color": "rose_quartz",
        "size": "30OZ",
        "median_price_usd": 35.00,
    },
    # Family 5: KitchenAid Artisan 5-Quart Stand Mixer (110-120V US vs 220-240V EU)
    {
        "asin": "B08X6P2K19",
        "parent_asin": "B08X6KITCH00",
        "title": "KitchenAid Artisan Series 5-Quart Tilt-Head Stand Mixer Empire Red 120V US",
        "brand": "KitchenAid",
        "product_type": "stand_mixer",
        "model_family": "artisan 5qt tilt-head",
        "mpn": "KSM150PSER",
        "gtin14": _make_valid_gtin14("0088381162951"),
        "pack_count": 1,
        "storage_gb": None,
        "voltage_v": "110-120V",
        "color": "red",
        "size": "5QT",
        "median_price_usd": 449.99,
    },
    {
        "asin": "B08X6P2EU9",
        "parent_asin": "B08X6KITCH00",
        "title": "KitchenAid Artisan Series 5-Quart Tilt-Head Stand Mixer Empire Red 230V EU",
        "brand": "KitchenAid",
        "product_type": "stand_mixer",
        "model_family": "artisan 5qt tilt-head",
        "mpn": "5KSM150PSER",
        "gtin14": _make_valid_gtin14("0088381162952"),
        "pack_count": 1,
        "storage_gb": None,
        "voltage_v": "220-240V",
        "color": "red",
        "size": "5QT",
        "median_price_usd": 479.99,
    },
    # Family 6: Apple MacBook Pro 14-inch M3 Pro (512GB vs 1TB, Space Black vs Silver)
    {
        "asin": "B09JQL8KP9",
        "parent_asin": "B09JQMACBOOK",
        "title": "Apple MacBook Pro 14-inch M3 Pro Laptop 18GB Unified Memory 512GB SSD Space Black",
        "brand": "Apple",
        "product_type": "laptop",
        "model_family": "macbook pro 14 m3 pro",
        "mpn": "MRX33LL/A",
        "gtin14": "00194252056134",
        "pack_count": 1,
        "storage_gb": 512,
        "voltage_v": "100-240V",
        "color": "black",
        "size": None,
        "median_price_usd": 1799.00,
    },
    {
        "asin": "B09JQL1TB9",
        "parent_asin": "B09JQMACBOOK",
        "title": "Apple MacBook Pro 14-inch M3 Pro Laptop 18GB Unified Memory 1TB SSD Space Black",
        "brand": "Apple",
        "product_type": "laptop",
        "model_family": "macbook pro 14 m3 pro",
        "mpn": "MRX43LL/A",
        "gtin14": _make_valid_gtin14("0019425205614"),
        "pack_count": 1,
        "storage_gb": 1000,
        "voltage_v": "100-240V",
        "color": "black",
        "size": None,
        "median_price_usd": 2199.00,
    },
    # Family 7: Bosch Stainless Metric Mirror Thread Adapter (M10x1.25 vs M10x1.5)
    {
        "asin": "B07M10P125",
        "parent_asin": "B07M10THREAD",
        "title": "Bosch CNC Stainless Mirror Mount Adapter Stud Black M10x1.25 Fine Pitch (2-Pack)",
        "brand": "Bosch",
        "product_type": "thread_adapter",
        "model_family": "cnc mirror mount stud",
        "mpn": "BSH-M10-125-2P",
        "gtin14": _make_valid_gtin14("0031651409911"),
        "pack_count": 2,
        "storage_gb": None,
        "voltage_v": None,
        "color": "black",
        "size": "M10X1.25",
        "median_price_usd": 14.99,
    },
    {
        "asin": "B07M10P150",
        "parent_asin": "B07M10THREAD",
        "title": "Bosch CNC Stainless Mirror Mount Adapter Stud Black M10x1.5 Standard Pitch (2-Pack)",
        "brand": "Bosch",
        "product_type": "thread_adapter",
        "model_family": "cnc mirror mount stud",
        "mpn": "BSH-M10-150-2P",
        "gtin14": _make_valid_gtin14("0031651409912"),
        "pack_count": 2,
        "storage_gb": None,
        "voltage_v": None,
        "color": "black",
        "size": "M10X1.5",
        "median_price_usd": 14.99,
    },
    # Family 8: DeWalt 20V MAX XR Lithium-Ion Battery Pack (1-Pack vs 2-Pack)
    {
        "asin": "B01DEW20V1",
        "parent_asin": "B01DEWPARENT",
        "title": "DeWalt 20V MAX XR 5.0Ah Lithium-Ion Power Tool Battery Pack DCB205 (1-Pack)",
        "brand": "DeWalt",
        "product_type": "tool_battery",
        "model_family": "20v max xr 5.0ah dcb205",
        "mpn": "DCB205",
        "gtin14": _make_valid_gtin14("0088591136111"),
        "pack_count": 1,
        "storage_gb": None,
        "voltage_v": "20V-DC",
        "color": None,
        "size": None,
        "median_price_usd": 89.00,
    },
    {
        "asin": "B01DEW20V2",
        "parent_asin": "B01DEWPARENT",
        "title": "DeWalt 20V MAX XR 5.0Ah Lithium-Ion Power Tool Battery Pack DCB205-2 (2-Pack)",
        "brand": "DeWalt",
        "product_type": "tool_battery",
        "model_family": "20v max xr 5.0ah dcb205",
        "mpn": "DCB205-2",
        "gtin14": _make_valid_gtin14("0088591136112"),
        "pack_count": 2,
        "storage_gb": None,
        "voltage_v": "20V-DC",
        "color": None,
        "size": None,
        "median_price_usd": 159.00,
    },
]

# Deterministic universal hash parameters for 64-permutation character 3-gram MinHash
_PRIME_MOD = 4294967311
_MINHASH_COEFFS: List[Tuple[int, int]] = [
    ((i * 2654435761 + 1013904223) % (_PRIME_MOD - 1) + 1, (i * 1664525 + 374761393) % _PRIME_MOD)
    for i in range(1, 65)
]

STOPWORDS = {"the", "and", "for", "with", "new", "genuine", "authentic", "official", "by", "in", "of", "to"}


def infer_product_type(title: str) -> str:
    low = title.lower()
    if any(k in low for k in ("sleeve", "neoprene case", "carrying pouch", "screen protector", "protective case for")):
        return "accessory_case"
    if any(k in low for k in ("portable ssd", "external ssd", "solid state drive", "t7 shield")):
        return "portable_ssd"
    if any(k in low for k in ("wall charger", "gan charger", "usb-c charger", "fast charger", "netzteil", "充電器")):
        return "wall_charger"
    if any(k in low for k in ("pitcher", "water filter", "replacement filter", "wasserfilter")):
        return "water_filter"
    if any(k in low for k in ("tumbler", "quencher", "insulated cup", "travel mug")):
        return "insulated_tumbler"
    if any(k in low for k in ("stand mixer", "tilt-head mixer", "küchenmaschine")):
        return "stand_mixer"
    if any(k in low for k in ("macbook", "laptop", "notebook")):
        return "laptop"
    if any(k in low for k in ("mirror mount", "thread adapter", "adapter stud")):
        return "thread_adapter"
    if any(k in low for k in ("tool battery", "lithium-ion", "dcb205")):
        return "tool_battery"
    return "general_merchandise"


def _normalize_tokens(text: str) -> Set[str]:
    raw_toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw_toks if len(t) >= 2 and t not in STOPWORDS}


def compute_minhash_signature(text: str) -> List[int]:
    """Compute 64-permutation MinHash signature over character 3-grams."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()
    if len(cleaned) < 3:
        cleaned = cleaned.ljust(3, "_")
    shingles = {zlib.crc32(cleaned[i : i + 3].encode("utf-8")) & 0xFFFFFFFF for i in range(len(cleaned) - 2)}
    sig: List[int] = []
    for a, b in _MINHASH_COEFFS:
        sig.append(min((a * sh + b) % _PRIME_MOD for sh in shingles))
    return sig


def minhash_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def token_jaccard(toks_a: Set[str], toks_b: Set[str]) -> float:
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def evaluate_variant_vetoes(offer_axes: Dict[str, Any], asin_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return hard-constraint variant vetoes when pack_count, storage_gb, voltage_v, color, or size conflict."""
    vetoes: List[Dict[str, Any]] = []

    # 1. Pack Count is ALWAYS enforced (default 1 vs multi-pack)
    o_pack = int(offer_axes.get("pack_count") or 1)
    a_pack = int(asin_entry.get("pack_count") or 1)
    if o_pack != a_pack:
        vetoes.append(
            {
                "axis": "pack_count",
                "code": "HARD_VETO_PACK_COUNT",
                "offer_value": o_pack,
                "asin_value": a_pack,
                "detail": f"Offer pack_count={o_pack} conflicts with ASIN pack_count={a_pack}",
            }
        )

    # 2. Storage Capacity (GB)
    o_st = offer_axes.get("storage_gb")
    a_st = asin_entry.get("storage_gb")
    if o_st is not None and a_st is not None and int(o_st) != int(a_st):
        vetoes.append(
            {
                "axis": "storage_gb",
                "code": "HARD_VETO_STORAGE_GB",
                "offer_value": int(o_st),
                "asin_value": int(a_st),
                "detail": f"Offer storage={o_st}GB conflicts with ASIN storage={a_st}GB",
            }
        )

    # 3. Voltage Class
    o_volt = offer_axes.get("voltage_v")
    a_volt = asin_entry.get("voltage_v")
    if o_volt and a_volt and str(o_volt).upper() != str(a_volt).upper():
        vetoes.append(
            {
                "axis": "voltage_v",
                "code": "HARD_VETO_VOLTAGE",
                "offer_value": o_volt,
                "asin_value": a_volt,
                "detail": f"Offer voltage={o_volt} conflicts with ASIN voltage={a_volt}",
            }
        )

    # 4. Color
    o_col = offer_axes.get("color")
    a_col = asin_entry.get("color")
    if o_col and a_col and str(o_col).lower() != str(a_col).lower():
        vetoes.append(
            {
                "axis": "color",
                "code": "HARD_VETO_COLOR",
                "offer_value": o_col,
                "asin_value": a_col,
                "detail": f"Offer color='{o_col}' conflicts with ASIN color='{a_col}'",
            }
        )

    # 5. Size / Thread Pitch / Volume Variant
    o_sz = offer_axes.get("size")
    a_sz = asin_entry.get("size")
    if o_sz and a_sz and str(o_sz).upper() != str(a_sz).upper():
        vetoes.append(
            {
                "axis": "size",
                "code": "HARD_VETO_SIZE",
                "offer_value": o_sz,
                "asin_value": a_sz,
                "detail": f"Offer size='{o_sz}' conflicts with ASIN size='{a_sz}'",
            }
        )

    return vetoes


class VariantSafeEntityResolver:
    """Blocks candidates by (brand, product_type), scores MinHash + token overlap, and enforces hard variant vetoes."""

    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None) -> None:
        self.catalog = list(catalog or CANONICAL_ASIN_CATALOG)
        self.by_asin: Dict[str, Dict[str, Any]] = {}
        self.by_gtin14: Dict[str, Dict[str, Any]] = {}
        self.by_block: Dict[str, List[Dict[str, Any]]] = {}
        self._signatures: Dict[str, List[int]] = {}
        self._tokens: Dict[str, Set[str]] = {}

        for row in self.catalog:
            asin = row["asin"]
            self.by_asin[asin] = row
            if row.get("gtin14"):
                self.by_gtin14[str(row["gtin14"])] = row
            block_key = f"{row['brand'].strip().lower()}:{row['product_type']}"
            self.by_block.setdefault(block_key, []).append(row)
            self._signatures[asin] = compute_minhash_signature(f"{row['brand']} {row['title']} {row.get('mpn', '')}")
            self._tokens[asin] = _normalize_tokens(f"{row['brand']} {row['title']} {row.get('mpn', '')}")

    def resolve_naive_text_baseline(
        self,
        title: str,
        brand: Optional[str] = None,
        gtin: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Naive text-similarity baseline WITHOUT structured variant vetoes (used for benchmark comparison)."""
        q_str = f"{brand or ''} {title}".strip()
        q_toks = _normalize_tokens(q_str)
        q_sig = compute_minhash_signature(q_str)

        best_asin: Optional[str] = None
        best_score = 0.0
        for row in self.catalog:
            asin = row["asin"]
            t_sim = token_jaccard(q_toks, self._tokens[asin])
            m_sim = minhash_jaccard(q_sig, self._signatures[asin])
            score = round(0.60 * t_sim + 0.40 * m_sim, 4)
            if score > best_score:
                best_score = score
                best_asin = asin

        if best_score >= 0.46 and best_asin:
            return {
                "decision": "MATCH_EXISTING_ASIN",
                "matched_asin": best_asin,
                "similarity": best_score,
            }
        return {"decision": "NO_MATCH", "matched_asin": None, "similarity": best_score}

    def resolve_offer(
        self,
        title: str,
        brand: Optional[str] = None,
        mpn: Optional[str] = None,
        gtin: Optional[str] = None,
        explicit_attrs: Optional[Dict[str, Any]] = None,
        use_bedrock_tiebreaker: bool = False,
    ) -> Dict[str, Any]:
        """Resolve an external merchant offer to an existing ASIN, a new child variant ASIN, or a conflict hold."""
        t0 = time.perf_counter()
        if not title or not title.strip():
            raise ValueError("Offer title is required for ASIN resolution")

        inferred_type = infer_product_type(title)
        offer_axes = extract_variant_axes(title, explicit_attrs)
        brand_clean = (brand or "").strip()
        if not brand_clean:
            for row in self.catalog:
                if re.search(rf"\b{re.escape(row['brand'])}\b", title, re.IGNORECASE):
                    brand_clean = row["brand"]
                    break
        brand_key = (brand_clean or "generic").lower()
        block_key = f"{brand_key}:{inferred_type}"

        gtin_info = validate_gtin_check_digit(gtin)
        gtin14 = gtin_info["gtin14"] if gtin_info["valid"] else ""

        # Guard 0: Accessory masquerading as a primary hardware GTIN
        if gtin14 and gtin14 in self.by_gtin14:
            gtin_target = self.by_gtin14[gtin14]
            if inferred_type == "accessory_case" and gtin_target["product_type"] != "accessory_case":
                return {
                    "decision": "HOLD_VARIANT_CONFLICT",
                    "matched_asin": None,
                    "parent_asin": None,
                    "conflicting_asin": gtin_target["asin"],
                    "blocking_key": block_key,
                    "offer_variants": offer_axes,
                    "vetoes_triggered": [
                        {
                            "axis": "product_type",
                            "code": "HARD_VETO_PRODUCT_TYPE",
                            "offer_value": inferred_type,
                            "asin_value": gtin_target["product_type"],
                            "detail": f"Accessory offer claims primary hardware GTIN of {gtin_target['asin']}",
                        }
                    ],
                    "naive_baseline": self.resolve_naive_text_baseline(title, brand_clean, gtin),
                    "candidates_evaluated": 1,
                    "bedrock_tiebreaker": None,
                    "resolve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }

            gtin_vetoes = evaluate_variant_vetoes(offer_axes, gtin_target)
            if gtin_vetoes:
                return {
                    "decision": "HOLD_VARIANT_CONFLICT",
                    "matched_asin": None,
                    "parent_asin": gtin_target["parent_asin"],
                    "conflicting_asin": gtin_target["asin"],
                    "blocking_key": block_key,
                    "offer_variants": offer_axes,
                    "vetoes_triggered": gtin_vetoes,
                    "naive_baseline": self.resolve_naive_text_baseline(title, brand_clean, gtin),
                    "candidates_evaluated": 1,
                    "bedrock_tiebreaker": None,
                    "resolve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }

        # Blocking lookup: same (brand, product_type), falling back to same product_type if brand missing
        candidates = list(self.by_block.get(block_key, []))
        if not candidates:
            candidates = [r for r in self.catalog if r["product_type"] == inferred_type]

        q_str = f"{brand_clean} {title} {mpn or ''}".strip()
        q_sig = compute_minhash_signature(q_str)
        q_toks = _normalize_tokens(q_str)
        mpn_norm = re.sub(r"[\s\-]", "", (mpn or "").upper())

        compatible_scored: List[Dict[str, Any]] = []
        vetoed_siblings: List[Dict[str, Any]] = []

        for cand in candidates:
            asin = cand["asin"]
            t_sim = token_jaccard(q_toks, self._tokens[asin])
            m_sim = minhash_jaccard(q_sig, self._signatures[asin])
            cand_mpn = re.sub(r"[\s\-]", "", str(cand.get("mpn") or "").upper())
            mpn_boost = 0.22 if (mpn_norm and cand_mpn and mpn_norm == cand_mpn) else 0.0
            gtin_boost = 0.30 if (gtin14 and gtin14 == cand.get("gtin14")) else 0.0
            combined_score = min(1.0, round(0.55 * t_sim + 0.45 * m_sim + mpn_boost + gtin_boost, 4))

            vetoes = evaluate_variant_vetoes(offer_axes, cand)
            entry_eval = {
                "asin": asin,
                "parent_asin": cand["parent_asin"],
                "title": cand["title"],
                "token_jaccard": round(t_sim, 4),
                "minhash_similarity": round(m_sim, 4),
                "combined_score": combined_score,
                "vetoes": vetoes,
            }
            if vetoes:
                vetoed_siblings.append(entry_eval)
            else:
                compatible_scored.append(entry_eval)

        compatible_scored.sort(key=lambda x: x["combined_score"], reverse=True)
        vetoed_siblings.sort(key=lambda x: x["combined_score"], reverse=True)

        bedrock_meta: Optional[Dict[str, Any]] = None
        if compatible_scored:
            top = compatible_scored[0]
            if top["combined_score"] >= 0.46:
                if use_bedrock_tiebreaker and 0.46 <= top["combined_score"] < 0.78:
                    bedrock_meta = converse_bedrock(
                        system_prompt=(
                            "You are an Amazon Project Dawn ASIN entity resolution judge. "
                            "Reply ONLY with JSON: {\"same_product\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}."
                        ),
                        user_prompt=(
                            f"Merchant Offer: {title} (variants={offer_axes})\n"
                            f"Candidate ASIN {top['asin']}: {top['title']}"
                        ),
                        model_id=NOVA_MICRO_MODEL,
                        max_tokens=160,
                    )
                return {
                    "decision": "MATCH_EXISTING_ASIN",
                    "matched_asin": top["asin"],
                    "matched_title": top["title"],
                    "parent_asin": top["parent_asin"],
                    "similarity": top["combined_score"],
                    "blocking_key": block_key,
                    "offer_variants": offer_axes,
                    "vetoes_triggered": vetoed_siblings[0]["vetoes"] if vetoed_siblings else [],
                    "vetoed_sibling_candidates": vetoed_siblings[:3],
                    "naive_baseline": self.resolve_naive_text_baseline(title, brand_clean, gtin),
                    "candidates_evaluated": len(candidates),
                    "bedrock_tiebreaker": bedrock_meta,
                    "resolve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }

        # No compatible child ASIN matched, check if a strong sibling family was vetoed -> create new child variant ASIN!
        if vetoed_siblings and vetoed_siblings[0]["combined_score"] >= 0.40:
            best_sib = vetoed_siblings[0]
            return {
                "decision": "CREATE_CHILD_VARIANT_ASIN",
                "matched_asin": None,
                "parent_asin": best_sib["parent_asin"],
                "closest_sibling_asin": best_sib["asin"],
                "closest_sibling_title": best_sib["title"],
                "similarity": best_sib["combined_score"],
                "blocking_key": block_key,
                "offer_variants": offer_axes,
                "vetoes_triggered": best_sib["vetoes"],
                "vetoed_sibling_candidates": vetoed_siblings[:3],
                "naive_baseline": self.resolve_naive_text_baseline(title, brand_clean, gtin),
                "candidates_evaluated": len(candidates),
                "bedrock_tiebreaker": None,
                "resolve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        return {
            "decision": "CREATE_NEW_PARENT_ASIN",
            "matched_asin": None,
            "parent_asin": None,
            "similarity": compatible_scored[0]["combined_score"] if compatible_scored else 0.0,
            "blocking_key": block_key,
            "offer_variants": offer_axes,
            "vetoes_triggered": [],
            "vetoed_sibling_candidates": vetoed_siblings[:2],
            "naive_baseline": self.resolve_naive_text_baseline(title, brand_clean, gtin),
            "candidates_evaluated": len(candidates),
            "bedrock_tiebreaker": None,
            "resolve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


SAMPLE_ENTITY_RESOLUTION_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "ER-TRAP-PACK-01",
        "label": "Adversarial Pack-Size Trap: 1-Pack Anker 67W vs 2-Pack ASIN (96% Token Overlap)",
        "title": "Anker Prime 67W USB-C GaN Wall Charger 3-Port Compact Fast Charger Black (Single 1-Pack)",
        "brand": "Anker",
        "mpn": "",
        "gtin": "",
        "expected_decision": "MATCH_EXISTING_ASIN",
        "expected_asin": "B0C4YTB56H",
    },
    {
        "case_id": "ER-NEW-VARIANT-02",
        "label": "New Variant Discovery: Anker Prime 67W Charger 4-Pack Bundle -> Child ASIN",
        "title": "Anker Prime 67W USB-C GaN Wall Charger 3-Port Compact Fast Charger Black (Pack of 4)",
        "brand": "Anker",
        "mpn": "A2669-4PK",
        "gtin": "",
        "expected_decision": "CREATE_CHILD_VARIANT_ASIN",
        "expected_parent_asin": "B0C4YPARENT2",
    },
    {
        "case_id": "ER-CONFLICT-GTIN-03",
        "label": "GTIN Barcode Reuse Conflict: 1-Pack Brita Filter Using 6-Pack GTIN",
        "title": "Brita Standard Water Pitcher and Dispenser Replacement Filter BPA-Free White (1-Pack)",
        "brand": "Brita",
        "mpn": "OB03-1PK",
        "gtin": "00060258362169",
        "expected_decision": "HOLD_VARIANT_CONFLICT",
    },
    {
        "case_id": "ER-TRAP-STORAGE-04",
        "label": "Capacity Variant Separation: Samsung T7 Shield 4TB Blue -> New Child Under Parent",
        "title": "Samsung T7 Shield 4TB USB 3.2 Gen 2 Portable SSD Blue (MU-PE4T0R)",
        "brand": "Samsung",
        "mpn": "MU-PE4T0R",
        "gtin": "",
        "expected_decision": "CREATE_CHILD_VARIANT_ASIN",
        "expected_parent_asin": "B0BHZPARENT1",
    },
]
