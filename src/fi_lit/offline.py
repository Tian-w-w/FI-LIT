"""Validate a private, local inventory of assets needed by an offline server."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union


class AssetManifestError(ValueError):
    """Raised for invalid asset inventory files."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_assets(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Check file presence and optional checksums without network access."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise AssetManifestError("Asset manifest must use schema_version 1.")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise AssetManifestError("Asset manifest 'assets' must be a list.")
    results: List[Dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, Mapping) or not isinstance(asset.get("path"), str):
            raise AssetManifestError("Each asset requires a string path.")
        asset_path = Path(asset["path"])
        item: Dict[str, Any] = {
            "kind": asset.get("kind", "unknown"),
            "path": str(asset_path),
            "exists": asset_path.exists(),
        }
        expected = asset.get("sha256", "")
        if expected and asset_path.is_file():
            item["sha256_matches"] = _sha256_file(asset_path).lower() == str(expected).lower()
        elif expected:
            item["sha256_matches"] = False
        results.append(item)
    passed = all(item["exists"] and item.get("sha256_matches", True) for item in results)
    return {"passed": passed, "assets": results}

