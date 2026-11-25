"""Raw Merchant HTML / JSON-LD / CSV Offer Extractor & GTIN / Unit-Price Integrity Guard."""
from __future__ import annotations

import csv
import io
import json
import re
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

FX_TO_USD: Dict[str, float] = {
    "USD": 1.00,
    "$": 1.00,
    "EUR": 1.08,
    "€": 1.08,
    "GBP": 1.27,
    "£": 1.27,
    "JPY": 0.0067,
    "¥": 0.0067,
    "CAD": 0.73,
}

# Reference ASIN/GTIN median prices (canonical 14-digit GTIN -> expected unit/bundle price & item type)
REFERENCE_GTIN_CATALOG: Dict[str, Dict[str, Any]] = {
    "00194252056134": {
        "asin": "B09JQL8KP9",
        "title": "Apple MacBook Pro 14-inch M3 Pro 18GB RAM 512GB SSD Space Black",
        "median_price_usd": 1799.00,
        "expected_pack_count": 1,
        "primary_device": True,
    },
    "00887276646084": {
        "asin": "B0BHZT5S12",
        "title": "Samsung T7 Shield 2TB USB 3.2 Gen 2 Portable SSD Blue (MU-PE2T0R)",
        "median_price_usd": 169.99,
        "expected_pack_count": 1,
        "primary_device": True,
    },
    "00887276646077": {
        "asin": "B0BHZQGN26",
        "title": "Samsung T7 Shield 1TB USB 3.2 Gen 2 Portable SSD Blue (MU-PE1T0R)",
        "median_price_usd": 99.99,
        "expected_pack_count": 1,
        "primary_device": True,
    },
    "00194644127480": {
        "asin": "B0C4YTB56H",
        "title": "Anker Prime 67W USB-C GaN Wall Charger 3-Port Black (1-Pack)",
        "median_price_usd": 54.99,
        "expected_pack_count": 1,
        "primary_device": True,
    },
    "00194644127497": {
        "asin": "B0C4YTB52P",
        "title": "Anker Prime 67W USB-C GaN Wall Charger 3-Port Black (2-Pack)",
        "median_price_usd": 99.99,
        "expected_pack_count": 2,
        "primary_device": True,
    },
    "00060258362169": {
        "asin": "B00004SU18",
        "title": "Brita Standard Water Pitcher Replacement Filters White (6-Pack)",
        "median_price_usd": 34.99,
        "expected_pack_count": 6,
        "primary_device": False,
    },
    "00060258362145": {
        "asin": "B00006IV0P",
        "title": "Brita Standard Water Pitcher Replacement Filters White (3-Pack)",
        "median_price_usd": 18.99,
        "expected_pack_count": 3,
        "primary_device": False,
    },
    "00041604352843": {
        "asin": "B0B3N5M78Q",
        "title": "Stanley Quencher H2.0 FlowState Tumbler 40 oz Rose Quartz (1-Pack)",
        "median_price_usd": 45.00,
        "expected_pack_count": 1,
        "primary_device": False,
    },
    "00883811629512": {
        "asin": "B08X6P2K19",
        "title": "KitchenAid Artisan Series 5-Quart Tilt-Head Stand Mixer Empire Red 120V",
        "median_price_usd": 449.99,
        "expected_pack_count": 1,
        "primary_device": True,
    },
}

ACCESSORY_HIJACK_RE = re.compile(
    r"\b(?:silicone\s+sleeve|neoprene\s+sleeve|protective\s+case\s+for|screen\s+protector\s+for|"
    r"carrying\s+pouch\s+for|replacement\s+cord\s+for|skin\s+sticker\s+for|dust\s+cover\s+for)\b",
    re.IGNORECASE,
)

COLOR_MAP: Dict[str, str] = {
    "black": "black",
    "space black": "black",
    "obsidian": "black",
    "midnight": "black",
    "schwarz": "black",
    "ブラック": "black",
    "white": "white",
    "arctic white": "white",
    "weiß": "white",
    "ホワイト": "white",
    "blue": "blue",
    "navy": "blue",
    "blau": "blue",
    "silver": "silver",
    "silber": "silver",
    "rose quartz": "rose_quartz",
    "pink": "rose_quartz",
    "empire red": "red",
    "red": "red",
    "beige": "beige",
    "sandstone": "beige",
    "green": "green",
}

KNOWN_BRANDS = [
    "Apple",
    "Samsung",
    "Anker",
    "Brita",
    "Stanley",
    "KitchenAid",
    "DeWalt",
    "Sony",
    "Bosch",
    "Yeti",
    "Logitech",
    "Makita",
]

WORD_PACK_MAP = {
    "single": 1,
    "twin": 2,
    "double": 2,
    "duo": 2,
    "triple": 3,
    "quad": 4,
}


def compute_gs1_check_digit(body_digits: str) -> int:
    """Compute GS1 Modulo-10 check digit for GTIN-8/12/13/14 body digits (excluding check digit)."""
    total = 0
    for idx, ch in enumerate(reversed(body_digits)):
        weight = 3 if (idx % 2 == 0) else 1
        total += int(ch) * weight
    return (10 - (total % 10)) % 10


def validate_gtin_check_digit(raw_gtin: Optional[str]) -> Dict[str, Any]:
    """Validate UPC-A (12), EAN-13/GTIN-13 (13), EAN-8 (8), or GTIN-14 (14) Modulo-10 check digit."""
    if raw_gtin is None or not str(raw_gtin).strip():
        return {
            "present": False,
            "valid": False,
            "format": "MISSING",
            "cleaned": "",
            "gtin14": "",
            "reason": "No GTIN/UPC/EAN supplied in merchant feed",
        }

    cleaned = re.sub(r"[\s\-_]", "", str(raw_gtin).strip())
    if not cleaned.isdigit():
        return {
            "present": True,
            "valid": False,
            "format": "NON_NUMERIC",
            "cleaned": cleaned,
            "gtin14": "",
            "reason": f"Non-numeric characters in barcode '{raw_gtin}'",
        }

    length = len(cleaned)
    fmt_map = {8: "EAN-8", 12: "UPC-A", 13: "GTIN-13", 14: "GTIN-14"}
    if length not in fmt_map:
        return {
            "present": True,
            "valid": False,
            "format": f"INVALID_LENGTH_{length}",
            "cleaned": cleaned,
            "gtin14": "",
            "reason": f"Expected 8, 12, 13, or 14 digits; got {length}",
        }

    if int(cleaned) == 0:
        return {
            "present": True,
            "valid": False,
            "format": fmt_map[length],
            "cleaned": cleaned,
            "gtin14": "",
            "reason": "All-zero placeholder barcode rejected",
        }

    expected = compute_gs1_check_digit(cleaned[:-1])
    observed = int(cleaned[-1])
    is_valid = expected == observed
    gtin14 = cleaned.zfill(14) if is_valid else ""
    return {
        "present": True,
        "valid": is_valid,
        "format": fmt_map[length],
        "cleaned": cleaned,
        "gtin14": gtin14,
        "expected_check_digit": expected,
        "observed_check_digit": observed,
        "reason": "Modulo-10 GS1 check digit verified"
        if is_valid
        else f"Modulo-10 check digit mismatch (observed {observed}, expected {expected})",
    }


def extract_variant_axes(text: str, explicit_attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract canonical variant constraint axes (pack_count, storage_gb, voltage_v, color, size, wattage_w)."""
    attrs = explicit_attrs or {}
    combined = f"{text or ''} {' '.join(f'{k}:{v}' for k, v in attrs.items() if v is not None)}"
    low = combined.lower()

    # 1. Pack Count
    pack_count: Optional[int] = None
    if attrs.get("pack_count") is not None:
        try:
            pack_count = int(attrs["pack_count"])
        except (TypeError, ValueError):
            pass

    if pack_count is None:
        patterns = [
            r"\b(\d{1,3})\s*[-–]?\s*(?:pack|pk|count|ct|stück|er[\s-]*pack)\b",
            r"\b(?:pack|set|box|bundle|case)\s+of\s+(\d{1,3})\b",
            r"(\d{1,3})\s*(?:個入|個セット|本組)",
        ]
        for pat in patterns:
            m = re.search(pat, low, flags=re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 200:
                    pack_count = val
                    break

    if pack_count is None:
        m_word = re.search(r"\b(single|twin|double|duo|triple|quad)\s*(?:pack|set)?\b", low)
        if m_word:
            pack_count = WORD_PACK_MAP.get(m_word.group(1).lower(), 1)

    if pack_count is None:
        pack_count = 1

    # 2. Storage Capacity (GB) - distinguish SSD/drive storage from small RAM mentions when both appear
    storage_gb: Optional[int] = None
    if attrs.get("storage_gb") is not None:
        try:
            storage_gb = int(attrs["storage_gb"])
        except (TypeError, ValueError):
            pass

    if storage_gb is None:
        tb_matches = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\s*tb\b", low)]
        gb_matches = [
            int(x)
            for x in re.findall(r"\b(\d{2,4})\s*gb(?!\s*(?:ram|unified\s+memory|ddr))", low)
        ]
        if tb_matches:
            storage_gb = int(round(max(tb_matches) * 1000))
        elif gb_matches:
            # Prefer capacities >= 64 GB as primary storage
            ssd_candidates = [g for g in gb_matches if g >= 64]
            storage_gb = max(ssd_candidates) if ssd_candidates else max(gb_matches)

    # 3. Voltage Class
    voltage_v: Optional[str] = None
    if attrs.get("voltage_v"):
        voltage_v = str(attrs["voltage_v"]).strip()
    else:
        if re.search(r"\b100\s*[-–/]\s*240\s*v\b|\bdual[\s-]*voltage\b|\bworldwide\s+voltage\b", low):
            voltage_v = "100-240V"
        elif re.search(r"\b(?:220|230|240)\s*v(?:olt)?\b", low):
            voltage_v = "220-240V"
        elif re.search(r"\b(?:110|115|120)\s*v(?:olt)?\b", low):
            voltage_v = "110-120V"
        elif re.search(r"\b100\s*v(?:olt)?\b", low):
            voltage_v = "100V"
        elif re.search(r"\b(18|20|40)\s*v\s*(?:max|xr|li-ion|cordless)?\b", low):
            m_dc = re.search(r"\b(18|20|40)\s*v\b", low)
            if m_dc:
                voltage_v = f"{m_dc.group(1)}V-DC"

    # 4. Color
    color: Optional[str] = None
    if attrs.get("color"):
        c_raw = str(attrs["color"]).strip().lower()
        color = COLOR_MAP.get(c_raw, c_raw)
    else:
        # Match multi-word colors first ("space black", "rose quartz", "empire red")
        for phrase in sorted(COLOR_MAP.keys(), key=len, reverse=True):
            if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", low):
                color = COLOR_MAP[phrase]
                break

    # 5. Size / Thread Pitch / Fluid Capacity Variant
    size: Optional[str] = None
    if attrs.get("size"):
        size = str(attrs["size"]).strip().upper()
    else:
        m_thread = re.search(r"\b(m\d{1,2}\s*x\s*\d+(?:\.\d+)?)\b", low)
        m_oz = re.search(r"\b(14|16|20|24|30|32|40|64)\s*(?:fl\s*)?oz\b", low)
        m_qt = re.search(r"\b(\d+(?:\.\d+)?)\s*[-–]?\s*(?:quart|qt)\b", low)
        if m_thread:
            size = re.sub(r"\s+", "", m_thread.group(1).upper())
        elif m_oz:
            size = f"{m_oz.group(1)}OZ"
        elif m_qt:
            size = f"{m_qt.group(1)}QT"

    # 6. Wattage
    wattage_w: Optional[int] = None
    m_kw = re.search(r"\b(\d+[.,]\d+)\s*kw\b", low)
    m_w = re.search(r"\b(\d{2,4})\s*(?:w|watt|watts|ワット)\b", low)
    if m_kw:
        wattage_w = int(round(float(m_kw.group(1).replace(",", ".")) * 1000))
    elif m_w:
        wattage_w = int(m_w.group(1))

    return {
        "pack_count": pack_count,
        "storage_gb": storage_gb,
        "voltage_v": voltage_v,
        "color": color,
        "size": size,
        "wattage_w": wattage_w,
    }


class _MerchantHTMLSnippetParser(HTMLParser):
    """Lightweight HTML + JSON-LD + Microdata parser for messy external merchant pages."""

    def __init__(self) -> None:
        super().__init__()
        self.in_ld_json = False
        self.in_h1 = False
        self.in_title = False
        self.in_li = False
        self.in_price = False
        self._buf: List[str] = []
        self.ld_json_blocks: List[str] = []
        self.meta: Dict[str, str] = {}
        self.h1_texts: List[str] = []
        self.title_texts: List[str] = []
        self.bullets: List[str] = []
        self.price_texts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        ad = {k.lower(): (v or "") for k, v in attrs}
        t = tag.lower()
        if t == "script" and "ld+json" in ad.get("type", "").lower():
            self.in_ld_json = True
            self._buf = []
        elif t == "meta":
            key = (ad.get("property") or ad.get("name") or ad.get("itemprop") or "").lower().strip()
            content = ad.get("content", "").strip()
            if key and content:
                self.meta[key] = content
        elif t == "h1":
            self.in_h1 = True
            self._buf = []
        elif t == "title":
            self.in_title = True
            self._buf = []
        elif t == "li":
            self.in_li = True
            self._buf = []
        elif ad.get("itemprop", "").lower() == "price" or "price" in ad.get("class", "").lower():
            if ad.get("content"):
                self.price_texts.append(ad["content"])
            else:
                self.in_price = True
                self._buf = []

    def handle_data(self, data: str) -> None:
        if self.in_ld_json or self.in_h1 or self.in_title or self.in_li or self.in_price:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        joined = "".join(self._buf).strip()
        if t == "script" and self.in_ld_json:
            self.in_ld_json = False
            if joined:
                self.ld_json_blocks.append(joined)
        elif t == "h1" and self.in_h1:
            self.in_h1 = False
            if joined:
                self.h1_texts.append(joined)
        elif t == "title" and self.in_title:
            self.in_title = False
            if joined:
                self.title_texts.append(joined)
        elif t == "li" and self.in_li:
            self.in_li = False
            if joined:
                self.bullets.append(joined)
        elif self.in_price:
            self.in_price = False
            if joined:
                self.price_texts.append(joined)


def _parse_money_amount(raw: Any, default_currency: str = "USD") -> tuple[float, str]:
    if raw is None:
        return 0.0, default_currency
    if isinstance(raw, (int, float)):
        return float(raw), default_currency
    s = str(raw).strip()
    if not s:
        return 0.0, default_currency

    currency = default_currency
    for sym, code in [("€", "EUR"), ("£", "GBP"), ("¥", "JPY"), ("$", "USD")]:
        if sym in s:
            currency = code
            break
    for code in ("USD", "EUR", "GBP", "JPY", "CAD"):
        if re.search(rf"\b{code}\b", s, re.IGNORECASE):
            currency = code
            break

    num_str = re.sub(r"[^\d.,]", "", s)
    if not num_str:
        return 0.0, currency
    if "," in num_str and "." in num_str:
        if num_str.rfind(",") > num_str.rfind("."):
            num_str = num_str.replace(".", "").replace(",", ".")
        else:
            num_str = num_str.replace(",", "")
    elif "," in num_str:
        parts = num_str.split(",")
        if len(parts[-1]) == 2:
            num_str = num_str.replace(",", ".")
        else:
            num_str = num_str.replace(",", "")
    try:
        return float(num_str), currency
    except ValueError:
        return 0.0, currency


def _infer_brand(title: str, explicit_brand: Optional[str] = None) -> str:
    if explicit_brand and str(explicit_brand).strip():
        return str(explicit_brand).strip()
    for b in KNOWN_BRANDS:
        if re.search(rf"\b{re.escape(b)}\b", title, re.IGNORECASE):
            return b
    first_tok = re.split(r"[\s\-–:|]+", title.strip())[0] if title.strip() else "Generic"
    return first_tok or "Generic"


def _infer_mpn(text: str, explicit_mpn: Optional[str] = None) -> str:
    if explicit_mpn and str(explicit_mpn).strip():
        return str(explicit_mpn).strip().upper()
    m = re.search(r"\b([A-Z]{2,4}[-]?[A-Z0-9]{4,10})\b", text)
    if m:
        cand = m.group(1).upper()
        if not cand.isdigit() and cand not in ("HTML", "JSON", "USBC", "PACK", "FREE", "SHIPPING"):
            return cand
    return ""


def extract_merchant_offer(
    raw_payload: str,
    format_hint: str = "auto",
    merchant_id: str = "ext-merch-01",
    reference_median_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract a canonical commercial offer from raw HTML+JSON-LD, JSON, or CSV and run integrity guards."""
    t0 = time.perf_counter()
    raw = (raw_payload or "").strip()
    if not raw:
        raise ValueError("raw_payload must not be empty")

    detected_format = format_hint.lower()
    if detected_format == "auto":
        if raw.startswith("<") or "<html" in raw.lower() or "<script" in raw.lower() or "<div" in raw.lower():
            detected_format = "html"
        elif raw.startswith("{"):
            detected_format = "json"
        else:
            detected_format = "csv"

    title = ""
    brand = ""
    mpn = ""
    raw_gtin = ""
    condition = "New"
    raw_price: Any = 0.0
    raw_shipping: Any = 0.0
    currency = "USD"
    bullets: List[str] = []
    explicit_attrs: Dict[str, Any] = {}
    provenance_sources: List[str] = []

    if detected_format == "html":
        parser = _MerchantHTMLSnippetParser()
        parser.feed(raw)
        bullets.extend(parser.bullets)

        for block in parser.ld_json_blocks:
            try:
                ld = json.loads(block)
                items = ld.get("@graph", [ld]) if isinstance(ld, dict) else (ld if isinstance(ld, list) else [])
                for node in items:
                    if not isinstance(node, dict):
                        continue
                    ntype = str(node.get("@type", "")).lower()
                    if "product" in ntype or "title" in node or "name" in node:
                        title = title or str(node.get("name") or node.get("title") or "")
                        b_obj = node.get("brand")
                        if isinstance(b_obj, dict):
                            brand = brand or str(b_obj.get("name") or "")
                        elif isinstance(b_obj, str):
                            brand = brand or b_obj
                        mpn = mpn or str(node.get("mpn") or node.get("sku") or "")
                        raw_gtin = raw_gtin or str(
                            node.get("gtin13")
                            or node.get("gtin12")
                            or node.get("gtin14")
                            or node.get("gtin")
                            or node.get("upc")
                            or ""
                        )
                        if node.get("color"):
                            explicit_attrs["color"] = node["color"]
                        offers = node.get("offers")
                        if isinstance(offers, list) and offers:
                            offers = offers[0]
                        if isinstance(offers, dict):
                            raw_price = offers.get("price") or raw_price
                            currency = str(offers.get("priceCurrency") or currency)
                            cond_str = str(offers.get("itemCondition") or "")
                            if "refurb" in cond_str.lower():
                                condition = "Refurbished"
                            elif "used" in cond_str.lower():
                                condition = "Used"
                            ship_obj = offers.get("shippingDetails")
                            if isinstance(ship_obj, dict):
                                rate = ship_obj.get("shippingRate") or {}
                                if isinstance(rate, dict) and rate.get("value") is not None:
                                    raw_shipping = rate.get("value")
                        provenance_sources.append("json_ld_product")
            except Exception:
                pass

        if not title:
            title = (
                parser.h1_texts[0]
                if parser.h1_texts
                else (parser.meta.get("og:title") or (parser.title_texts[0] if parser.title_texts else ""))
            )
            if title:
                provenance_sources.append("html_dom_heading")

        if not raw_gtin:
            raw_gtin = (
                parser.meta.get("product:gtin")
                or parser.meta.get("gtin13")
                or parser.meta.get("upc")
                or ""
            )
            if not raw_gtin:
                m_bar = re.search(r"(?:UPC|EAN|GTIN(?:-13|-12)?)[\s:#]*([0-9\-]{8,17})", raw, re.IGNORECASE)
                if m_bar:
                    raw_gtin = m_bar.group(1)
                    provenance_sources.append("html_spec_barcode")

        if not raw_price:
            raw_price = (
                parser.meta.get("product:price:amount")
                or (parser.price_texts[0] if parser.price_texts else "")
            )
            currency = parser.meta.get("product:price:currency") or currency
            if raw_price:
                provenance_sources.append("html_price_tag")

    elif detected_format == "json":
        obj = json.loads(raw)
        title = str(obj.get("title") or obj.get("name") or "")
        brand = str(obj.get("brand") or "")
        mpn = str(obj.get("mpn") or obj.get("model") or "")
        raw_gtin = str(obj.get("gtin") or obj.get("upc") or obj.get("ean") or "")
        raw_price = obj.get("price", 0.0)
        raw_shipping = obj.get("shipping", 0.0)
        currency = str(obj.get("currency") or "USD")
        condition = str(obj.get("condition") or "New")
        bullets = [str(x) for x in (obj.get("bullets") or [])]
        explicit_attrs = dict(obj.get("attributes") or {})
        merchant_id = str(obj.get("merchant_id") or merchant_id)
        provenance_sources.append("merchant_json_feed")

    else:
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        if not rows:
            raise ValueError("CSV feed contained zero data rows")
        row = {str(k).strip().lower(): str(v or "").strip() for k, v in rows[0].items() if k}
        title = row.get("title") or row.get("product_name") or row.get("name") or ""
        brand = row.get("brand") or ""
        mpn = row.get("mpn") or row.get("sku") or ""
        raw_gtin = row.get("gtin") or row.get("upc") or row.get("ean") or ""
        raw_price = row.get("price") or "0"
        raw_shipping = row.get("shipping") or "0"
        currency = row.get("currency") or "USD"
        condition = row.get("condition") or "New"
        merchant_id = row.get("merchant_id") or merchant_id
        provenance_sources.append("merchant_csv_row")

    full_text = f"{title} {' '.join(bullets)}"
    brand = _infer_brand(title, brand)
    mpn = _infer_mpn(full_text, mpn)
    gtin_check = validate_gtin_check_digit(raw_gtin)
    variants = extract_variant_axes(full_text, explicit_attrs)

    price_val, detected_curr = _parse_money_amount(raw_price, currency)
    ship_val, _ = _parse_money_amount(raw_shipping, detected_curr)
    fx = FX_TO_USD.get(detected_curr.upper(), 1.0)
    item_price_usd = round(price_val * fx, 2)
    shipping_usd = round(ship_val * fx, 2)
    landed_price_usd = round(item_price_usd + shipping_usd, 2)
    pack_count = max(1, int(variants["pack_count"] or 1))
    unit_price_usd = round(landed_price_usd / pack_count, 2)

    # Integrity Guard checks
    ref_entry = REFERENCE_GTIN_CATALOG.get(gtin_check["gtin14"]) if gtin_check["gtin14"] else None
    ref_median = reference_median_usd or (ref_entry["median_price_usd"] if ref_entry else None)
    guard_flags: List[str] = []
    verdict = "INGEST_CLEAN"

    if gtin_check["present"] and not gtin_check["valid"]:
        verdict = "QUARANTINE_GTIN_INVALID"
        guard_flags.append(f"GTIN_MOD10_FAIL: {gtin_check['reason']}")
    elif not gtin_check["present"]:
        verdict = "ENRICH_MISSING_GTIN"
        guard_flags.append("MISSING_GTIN: Route to Variant-Safe Title/MPN Matcher")

    if landed_price_usd <= 0.0:
        verdict = "QUARANTINE_PRICE_OUTLIER"
        guard_flags.append("ZERO_OR_NEGATIVE_PRICE: Commercial offer requires positive landed price")
    elif ref_median is not None and ref_median > 0:
        ratio = landed_price_usd / ref_median
        if ref_entry and ref_entry.get("primary_device") and ACCESSORY_HIJACK_RE.search(full_text):
            verdict = "QUARANTINE_PRICE_OUTLIER"
            guard_flags.append(
                f"ACCESSORY_UPC_HIJACK: Accessory title claims primary device GTIN ({ref_entry['asin']}, median ${ref_median:.2f})"
            )
        elif ratio < 0.22:
            verdict = "QUARANTINE_PRICE_OUTLIER"
            guard_flags.append(
                f"LOW_PRICE_ANOMALY: Landed ${landed_price_usd:.2f} is {ratio * 100:.1f}% of GTIN median ${ref_median:.2f}"
            )
        elif ratio > 3.80:
            verdict = "QUARANTINE_PRICE_OUTLIER"
            guard_flags.append(
                f"HIGH_PRICE_ANOMALY: Landed ${landed_price_usd:.2f} is {ratio:.1f}x GTIN median ${ref_median:.2f} (decimal/pack feed error)"
            )
        elif ref_entry and int(ref_entry.get("expected_pack_count", 1)) != pack_count and verdict == "INGEST_CLEAN":
            verdict = "QUARANTINE_PACK_GTIN_MISMATCH"
            guard_flags.append(
                f"PACK_GTIN_CONFLICT: Offer specifies {pack_count}-pack but GTIN belongs to {ref_entry['expected_pack_count']}-pack {ref_entry['asin']}"
            )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "merchant_id": merchant_id,
        "detected_format": detected_format,
        "verdict": verdict,
        "guard_flags": guard_flags,
        "offer": {
            "title": title,
            "brand": brand,
            "mpn": mpn,
            "condition": condition,
            "raw_currency": detected_curr.upper(),
            "raw_price": price_val,
            "item_price_usd": item_price_usd,
            "shipping_usd": shipping_usd,
            "landed_price_usd": landed_price_usd,
            "unit_price_usd": unit_price_usd,
            "reference_median_usd": ref_median,
            "bullets": bullets,
            **variants,
        },
        "gtin_validation": gtin_check,
        "provenance_sources": provenance_sources,
        "extract_ms": elapsed_ms,
    }


SAMPLE_MERCHANT_PAYLOADS: List[Dict[str, Any]] = [
    {
        "sample_id": "OFFER-CLEAN-HTML-01",
        "label": "Clean External HTML + JSON-LD (Samsung T7 Shield 2TB Blue)",
        "merchant_id": "bh-photo-feed",
        "format_hint": "html",
        "payload": (
            '<div class="product-page">\n'
            '  <h1>Samsung T7 Shield 2TB USB 3.2 Gen 2 Portable SSD - Blue (MU-PE2T0R)</h1>\n'
            '  <script type="application/ld+json">\n'
            '  {\n'
            '    "@context": "https://schema.org",\n'
            '    "@type": "Product",\n'
            '    "name": "Samsung T7 Shield 2TB USB 3.2 Gen 2 Portable SSD Blue",\n'
            '    "brand": {"@type": "Brand", "name": "Samsung"},\n'
            '    "mpn": "MU-PE2T0R",\n'
            '    "gtin13": "0887276646084",\n'
            '    "offers": {"@type": "Offer", "price": "164.99", "priceCurrency": "USD", "itemCondition": "NewCondition"}\n'
            '  }\n'
            '  </script>\n'
            '  <ul><li>IP65 water and dust resistant rugged 2TB external solid state drive</li><li>Up to 1,050 MB/s read speeds</li></ul>\n'
            '</div>'
        ),
    },
    {
        "sample_id": "OFFER-HIJACK-UPC-02",
        "label": "Accessory Price Hijack ($6.99 Sleeve Claiming $1,799 MacBook Pro UPC)",
        "merchant_id": "rogue-marketplace-seller-77",
        "format_hint": "html",
        "payload": (
            '<div class="listing">\n'
            '  <h1>Soft Neoprene Protective Case Sleeve for Apple MacBook Pro 14-inch Space Black</h1>\n'
            '  <meta property="product:gtin" content="0194252056134" />\n'
            '  <meta property="product:price:amount" content="6.99" />\n'
            '  <meta property="product:price:currency" content="USD" />\n'
            '  <ul><li>Form-fitting zipper sleeve only (laptop not included)</li></ul>\n'
            '</div>'
        ),
    },
    {
        "sample_id": "OFFER-BAD-GTIN-03",
        "label": "Corrupted Check-Digit Feed (Anker 67W GaN Charger Invalid GTIN-13)",
        "merchant_id": "eu-distributor-csv",
        "format_hint": "csv",
        "payload": (
            "merchant_id,title,brand,mpn,gtin,price,shipping,currency,condition\n"
            "eu-distributor-csv,Anker Prime 67W USB-C GaN Wall Charger Black (1-Pack),Anker,A2669,0194644127489,49.50,0.00,EUR,New\n"
        ),
    },
    {
        "sample_id": "OFFER-PACK-HIJACK-04",
        "label": "Single Filter Claiming 6-Pack Brita GTIN ($7.49 vs $34.99 6-Pack)",
        "merchant_id": "discount-kitchen-supply",
        "format_hint": "json",
        "payload": json.dumps(
            {
                "merchant_id": "discount-kitchen-supply",
                "title": "Brita Standard Water Pitcher Replacement Filter White - Single (1-Pack)",
                "brand": "Brita",
                "gtin": "060258362169",
                "price": 7.49,
                "shipping": 0.00,
                "currency": "USD",
                "condition": "New",
            },
            indent=2,
        ),
    },
]
