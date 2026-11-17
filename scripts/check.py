"""Check original-data integrity, preparation evidence, calculations and optional submission."""
from __future__ import annotations

import argparse
import io
import json
import unittest
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from profile_data import build_profile
from rdm_common import EXPECTED, ROOT, TARGET, load_config, read_csv, sha256, validate_probabilities


def check_predictions(path: Path) -> None:
    columns, predictions = read_csv(path)
    if columns != ["equipment_id", "maintenance_date", "prob"]:
        raise ValueError("Submission columns must be equipment_id, maintenance_date, prob")
    _, original = read_csv(ROOT / "data" / "drone_test.csv")
    key = lambda r: (r["equipment_id"], r["maintenance_date"])
    if len(predictions) != len(original) or Counter(map(key, predictions)) != Counter(map(key, original)):
        raise ValueError("Submission must match all 1,241 source records, including every equipment/date key")
    for row in predictions:
        if date.fromisoformat(row["maintenance_date"]).isoformat() != row["maintenance_date"]:
            raise ValueError("Submission dates must use YYYY-MM-DD")
    validate_probabilities([row["prob"] for row in predictions])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Validate a future test_predictions_2025.csv")
    args = parser.parse_args()
    checks = []

    def run(name, operation):
        try:
            detail = operation()
            checks.append({"name": name, "passed": True, "detail": detail or "OK"})
            print(f"PASS {name}" + (f": {detail}" if detail else ""))
        except Exception as error:
            checks.append({"name": name, "passed": False, "detail": f"{type(error).__name__}: {error}"})
            print(f"FAIL {name}: {error}")

    def config_check():
        config = load_config()
        if config["report_month"] != "2026-01":
            raise ValueError("README reporting month is 2026-01; record a scope change before altering it")
        return f"month={config['report_month']}, capacity={config['capacity']}, threshold={config['risk_threshold']}"

    def integrity_check():
        manifest = json.loads((ROOT / "reports" / "data_manifest.json").read_text(encoding="utf-8"))
        for filename in EXPECTED:
            relative = "data/" + filename
            if sha256(ROOT / relative) != manifest[relative]:
                raise ValueError(f"Original data changed: {relative}. Investigate; do not reset the baseline")
        return "SHA-256 matched for all 3 original CSVs"

    def contract_check():
        keys = []
        for filename, contract in EXPECTED.items():
            columns, rows = read_csv(ROOT / "data" / filename)
            if tuple(columns) != tuple(contract["columns"]) or len(rows) != contract["rows"]:
                raise ValueError(f"README/schema mismatch: {filename}")
            for row in rows:
                day = date.fromisoformat(row["maintenance_date"]).isoformat()
                if not contract["start"] <= day <= contract["end"] or not row["equipment_id"].strip():
                    raise ValueError(f"Invalid date or equipment ID: {filename}")
                if TARGET in columns and row[TARGET] not in ("0", "1"):
                    raise ValueError("Training targets must be present and binary")
                keys.append((row["equipment_id"], day))
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate equipment/date keys need a documented resolution")
        return "train=2346, test=1241, month=106; held-out targets absent"

    def profile_check():
        saved = json.loads((ROOT / "reports" / "data_profile.json").read_text(encoding="utf-8"))
        if saved != build_profile():
            raise ValueError("Profile is stale. Run python scripts/profile_data.py")
        return "Saved profile matches a fresh calculation"

    def documentation_check():
        required = ["AGENTS.md", ".agents/skills/rdm-analysis/SKILL.md",
                    ".agents/skills/rdm-analysis/references/analysis-protocol.md",
                    "reports/preparation_log.md", "reports/data_profile.md"]
        missing = [path for path in required if not (ROOT / path).is_file()]
        if missing:
            raise ValueError("Missing preparation files: " + ", ".join(missing))
        return "Project rules, skill, protocol and preparation log exist"

    def calculation_check():
        pattern = "test_*.py" if (ROOT / "reports/analysis_results.json").exists() else "test_rdm_common.py"
        suite = unittest.defaultTestLoader.discover(str(ROOT / "scripts"), pattern=pattern)
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output).run(suite)
        if result.testsRun == 0 or not result.wasSuccessful():
            raise ValueError(output.getvalue())
        return f"{result.testsRun} calculation, temporal, submission and UI tests passed"

    run("configuration", config_check)
    run("original_integrity", integrity_check)
    run("data_contract", contract_check)
    run("profile_freshness", profile_check)
    run("preparation_documents", documentation_check)
    run("automated_tests", calculation_check)
    if args.predictions:
        run("test_submission", lambda: check_predictions(args.predictions.resolve()))
    summary = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "passed": all(item["passed"] for item in checks), "checks": checks,
               "scope": "Integrity, reproducibility, temporal and UI behavior checks. Educational validation metrics do not prove operational safety."}
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "check_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Checks: {sum(item['passed'] for item in checks)}/{len(checks)} passed")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
