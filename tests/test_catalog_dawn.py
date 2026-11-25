"""Comprehensive unit tests (12 tests) for CatalogDawn modules and API router."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

# Run deterministic unit tests fast without requiring network calls
os.environ["CATALOG_DAWN_OFFLINE_LLM"] = "1"

from catalog_dawn.audit_store import CatalogAuditStore
from catalog_dawn.entity_resolver import VariantSafeEntityResolver
from catalog_dawn.offer_extractor import (
    SAMPLE_MERCHANT_PAYLOADS,
    extract_merchant_offer,
    validate_gtin_check_digit,
)
from catalog_dawn.ontology_normalizer import (
    canonicalize_attribute_value,
    normalize_and_reconcile_record,
)
from catalog_dawn.self_learning_loop import (
    evaluate_rule_on_holdout,
    synthesize_and_validate_rule,
)
from catalog_dawn.server import lambda_handler, route_api_request


class TestCatalogDawnSuite(unittest.TestCase):
    def test_01_gtin_modulo10_validation(self) -> None:
        valid_ean13 = validate_gtin_check_digit("0887276646084")
        self.assertTrue(valid_ean13["valid"])
        self.assertEqual(valid_ean13["format"], "GTIN-13")
        self.assertEqual(valid_ean13["gtin14"], "00887276646084")

        valid_upca = validate_gtin_check_digit("060258362169")
        self.assertTrue(valid_upca["valid"])
        self.assertEqual(valid_upca["format"], "UPC-A")
        self.assertEqual(valid_upca["gtin14"], "00060258362169")

        invalid_ean13 = validate_gtin_check_digit("0194644127489")
        self.assertFalse(invalid_ean13["valid"])
        self.assertEqual(invalid_ean13["expected_check_digit"], 0)
        self.assertEqual(invalid_ean13["observed_check_digit"], 9)

    def test_02_html_jsonld_clean_offer_extraction(self) -> None:
        sample = SAMPLE_MERCHANT_PAYLOADS[0]
        res = extract_merchant_offer(sample["payload"], format_hint="html", merchant_id="test-bh")
        self.assertEqual(res["verdict"], "INGEST_CLEAN")
        self.assertEqual(res["offer"]["brand"], "Samsung")
        self.assertEqual(res["offer"]["storage_gb"], 2000)
        self.assertEqual(res["offer"]["color"], "blue")
        self.assertAlmostEqual(res["offer"]["landed_price_usd"], 164.99, places=2)

    def test_03_accessory_price_hijack_quarantine(self) -> None:
        sample = SAMPLE_MERCHANT_PAYLOADS[1]
        res = extract_merchant_offer(sample["payload"], format_hint="html", merchant_id="rogue-seller")
        self.assertEqual(res["verdict"], "QUARANTINE_PRICE_OUTLIER")
        self.assertTrue(any("ACCESSORY_UPC_HIJACK" in f or "LOW_PRICE_ANOMALY" in f for f in res["guard_flags"]))

    def test_04_csv_invalid_gtin_quarantine(self) -> None:
        sample = SAMPLE_MERCHANT_PAYLOADS[2]
        res = extract_merchant_offer(sample["payload"], format_hint="csv")
        self.assertEqual(res["verdict"], "QUARANTINE_GTIN_INVALID")
        self.assertEqual(res["offer"]["raw_currency"], "EUR")
        self.assertGreater(res["offer"]["landed_price_usd"], 50.0)

    def test_05_variant_safe_pack_count_never_conflates(self) -> None:
        resolver = VariantSafeEntityResolver()
        # 4-Pack Anker Prime 67W has 95%+ token overlap with 1-Pack and 2-Pack ASINs, must NOT merge onto either
        out = resolver.resolve_offer(
            title="Anker Prime 67W USB-C GaN Wall Charger 3-Port Compact Fast Charger Black (Pack of 4)",
            brand="Anker",
            mpn="A2669-4PK",
        )
        self.assertEqual(out["decision"], "CREATE_CHILD_VARIANT_ASIN")
        self.assertIsNone(out["matched_asin"])
        self.assertEqual(out["parent_asin"], "B0C4YPARENT2")
        self.assertTrue(any(v["code"] == "HARD_VETO_PACK_COUNT" for v in out["vetoes_triggered"]))
        # Meanwhile naive baseline falsely conflates onto an existing 1-pack or 2-pack ASIN!
        self.assertEqual(out["naive_baseline"]["decision"], "MATCH_EXISTING_ASIN")

    def test_06_variant_safe_storage_and_voltage_vetoes(self) -> None:
        resolver = VariantSafeEntityResolver()
        # 1TB SSD offer must match 1TB B0BHZQGN26, never 2TB B0BHZT5S12
        ssd_1tb = resolver.resolve_offer(
            title="Samsung T7 Shield Portable SSD 1TB USB 3.2 Blue External Solid State Drive",
            brand="Samsung",
        )
        self.assertEqual(ssd_1tb["decision"], "MATCH_EXISTING_ASIN")
        self.assertEqual(ssd_1tb["matched_asin"], "B0BHZQGN26")

        # 230V EU KitchenAid Stand Mixer must match EU ASIN B08X6P2EU9, never 120V US B08X6P2K19
        mixer_eu = resolver.resolve_offer(
            title="KitchenAid Artisan Series 5-Quart Tilt-Head Stand Mixer Empire Red 230V",
            brand="KitchenAid",
        )
        self.assertEqual(mixer_eu["decision"], "MATCH_EXISTING_ASIN")
        self.assertEqual(mixer_eu["matched_asin"], "B08X6P2EU9")

    def test_07_gtin_barcode_reuse_conflict_held(self) -> None:
        resolver = VariantSafeEntityResolver()
        # Offer claims 6-pack Brita GTIN 00060258362169 while selling 1-Pack
        conflict = resolver.resolve_offer(
            title="Brita Standard Water Pitcher Replacement Filter White Single (1-Pack)",
            brand="Brita",
            gtin="060258362169",
        )
        self.assertEqual(conflict["decision"], "HOLD_VARIANT_CONFLICT")
        self.assertEqual(conflict["conflicting_asin"], "B00004SU18")

    def test_08_cross_locale_unit_canonicalization(self) -> None:
        self.assertEqual(canonicalize_attribute_value("wattage_w", "0,065 kW", "DE")[0], "65 W")
        self.assertEqual(canonicalize_attribute_value("wattage_w", "65ワット", "JP")[0], "65 W")
        self.assertEqual(canonicalize_attribute_value("volume_ml", "16.9 fl oz", "US")[0], "500 ml")
        self.assertEqual(canonicalize_attribute_value("volume_ml", "500ミリリットル", "JP")[0], "500 ml")
        self.assertEqual(canonicalize_attribute_value("weight_g", "0,5 kg", "DE")[0], "500 g")
        self.assertEqual(canonicalize_attribute_value("battery_energy_wh", "20000 mAh 3.7V", "JP")[0], "74.0 Wh")

    def test_09_multi_seller_contradiction_resolution(self) -> None:
        rec = {
            "asin": "B0B3N5M78Q",
            "locale": "US",
            "category": "insulated_tumbler",
            "title": "Stanley Quencher H2.0 16.9 fl oz Stainless Tumbler",
            "bullets": ["18/8 stainless steel construction; weight 500 g"],
            "claims": [
                {"attribute": "dishwasher_safe", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "Dishwasher Safe: Yes"},
                {"attribute": "dishwasher_safe", "source": "OEM_MANUAL", "raw_value": "Hand Wash Only"},
                {"attribute": "dishwasher_safe", "source": "VERIFIED_RETURN_CLUSTER", "raw_value": "Hand wash only"},
                {"attribute": "volume_ml", "source": "SELLER_STRUCTURED_ATTR", "raw_value": "16.9 fl oz"},
            ],
        }
        out = normalize_and_reconcile_record(rec, use_bedrock_cascade=False)
        self.assertTrue(out["contradiction_detected"])
        self.assertEqual(out["resolved_attributes"]["dishwasher_safe"]["canonical_value"], "false")
        self.assertEqual(out["resolved_attributes"]["volume_ml"]["canonical_value"], "500 ml")
        self.assertGreater(out["rufus_readiness_after"], out["rufus_readiness_before"])

    def test_10_self_learning_holdout_promotes_high_precision_and_rejects_noisy(self) -> None:
        good = synthesize_and_validate_rule("CLUS-VOLT-110", use_bedrock=False)
        self.assertTrue(good["holdout_evaluation"]["passes_promotion_gate"])
        self.assertEqual(good["candidate_rule"]["status"], "PROMOTED_ACTIVE")
        self.assertGreaterEqual(good["holdout_evaluation"]["holdout_precision"], 0.95)
        self.assertEqual(good["holdout_evaluation"]["false_positives"], 0)

        noisy = evaluate_rule_on_holdout("overbroad_heavy_keyword_rule")
        self.assertFalse(noisy["passes_promotion_gate"])
        self.assertEqual(noisy["verdict"], "REJECTED_GUARDRAIL_FPR")
        self.assertGreater(noisy["false_positives"], 5)

    def test_11_sqlite_wal_hash_chain_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_audit.db"
            store = CatalogAuditStore(db_path=db_path)
            store.record_event("OFFER_INGEST", "0887276646084", "INGEST_CLEAN", 0.42, {"price": 164.99})
            ev2 = store.record_event("ASIN_RESOLVE", "B0BHZT5S12", "MATCH_EXISTING_ASIN", 0.31, {"sim": 0.91})
            self.assertTrue(store.verify_chain_integrity()["valid"])

            # Mutate a historical row directly in SQLite and verify hash-chain catches tampering
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE catalog_audit_events SET verdict='TAMPERED_VERDICT' WHERE event_id=?",
                (ev2["event_id"],),
            )
            conn.commit()
            conn.close()

            verify_after = store.verify_chain_integrity()
            self.assertFalse(verify_after["valid"])
            self.assertEqual(verify_after["broken_at"], ev2["event_id"])

    def test_12_lambda_handler_and_api_routes_end_to_end(self) -> None:
        st, health = route_api_request("GET", "/api/health", {}, {})
        self.assertEqual(st, 200)
        self.assertEqual(health["service"], "catalog-dawn-sls")

        event = {
            "rawPath": "/api/asin/resolve",
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(
                {
                    "title": "Bosch CNC Stainless Mirror Mount Adapter Stud Black M10x1.25 Fine Pitch (2-Pack)",
                    "brand": "Bosch",
                }
            ),
        }
        resp = lambda_handler(event, None)
        self.assertEqual(resp["statusCode"], 200)
        payload = json.loads(resp["body"])
        self.assertEqual(payload["decision"], "MATCH_EXISTING_ASIN")
        self.assertEqual(payload["matched_asin"], "B07M10P125")


if __name__ == "__main__":
    unittest.main()
