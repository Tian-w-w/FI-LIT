from __future__ import annotations

import hashlib
import json

from fi_lit.offline import check_assets


def test_asset_check_accepts_existing_file_with_matching_digest(tmp_path) -> None:
    asset = tmp_path / "asset.txt"
    asset.write_text("offline", encoding="utf-8")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "kind": "fixture",
                        "path": str(asset),
                        "sha256": hashlib.sha256(b"offline").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = check_assets(manifest)
    assert result["passed"] is True
    assert result["assets"][0]["sha256_matches"] is True


def test_asset_check_reports_missing_path(tmp_path) -> None:
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [{"kind": "model", "path": str(tmp_path / "missing")}],
            }
        ),
        encoding="utf-8",
    )
    assert check_assets(manifest)["passed"] is False

