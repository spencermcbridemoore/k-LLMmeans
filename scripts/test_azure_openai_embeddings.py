"""
Smoke-test Azure OpenAI: list deployments and call embeddings once.
Loads repo-root .env via python-dotenv. Does not print secrets.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv
from openai import AzureOpenAI

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"), encoding="utf-8")

KEY = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
API_VERSION = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01").strip()
# Use a dedicated embedding deployment name (Azure "Deployment name" for text-embedding-3-small).
EMBEDDING_DEPLOYMENT = (os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "").strip()
# Fallback: some .env files only set AZURE_OPENAI_DEPLOYMENT (often a chat model — wrong for embeddings).
FALLBACK_DEPLOYMENT = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()


def normalize_azure_endpoint(raw: str) -> str:
    """Strip trailing slashes and common path suffixes (404 if left on)."""
    u = raw.rstrip("/")
    for suffix in ("/openai", "/openai/v1", "/v1"):
        if u.lower().endswith(suffix.lower()):
            u = u[: -len(suffix)].rstrip("/")
    return u


def _placeholder(s: str) -> bool:
    if not s:
        return True
    low = s.lower()
    return "your_" in low or "here" in low and "your" in low or s == "..."


def list_deployments_rest(base: str, api_ver: str) -> list[dict]:
    url = f"{base}/openai/deployments?api-version={api_ver}"
    req = urllib.request.Request(
        url,
        headers={"api-key": KEY},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("data") or body.get("value") or []


def main() -> int:
    if _placeholder(KEY) or _placeholder(ENDPOINT):
        print(
            "FAIL: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT missing or still placeholder in .env",
            file=sys.stderr,
        )
        return 1

    base = normalize_azure_endpoint(ENDPOINT)
    if base != ENDPOINT:
        print(f"Normalized endpoint (removed path suffix): {base!r}")

    deployments: list[dict] = []
    list_errors: list[str] = []
    for dep_ver in ("2023-05-15", "2024-10-21", "2024-02-01"):
        print(f"Listing deployments (REST, api-version={dep_ver})...")
        try:
            deployments = list_deployments_rest(base, dep_ver)
            if deployments:
                break
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:500]
            list_errors.append(f"HTTP {e.code} ({dep_ver}): {err}")
        except Exception as e:
            list_errors.append(f"{type(e).__name__} ({dep_ver}): {e}")

    if not deployments and list_errors:
        print("WARN: Could not list deployments:", file=sys.stderr)
        for msg in list_errors:
            print(f"  {msg}", file=sys.stderr)

    if not deployments:
        print("WARN: No deployments returned (empty list).")
    else:
        for d in deployments:
            name = d.get("id") or d.get("name", "?")
            model = (d.get("model") or d.get("properties", {}).get("model", "?"))
            print(f"  deployment={name!r} model={model!r}")

    # Pick embedding deployment (must be an embeddings deployment in Azure, not gpt-4o etc.)
    target = None
    if EMBEDDING_DEPLOYMENT and not _placeholder(EMBEDDING_DEPLOYMENT):
        target = EMBEDDING_DEPLOYMENT
        print(f"Using AZURE_OPENAI_EMBEDDING_DEPLOYMENT={target!r}")
    elif FALLBACK_DEPLOYMENT and not _placeholder(FALLBACK_DEPLOYMENT):
        target = FALLBACK_DEPLOYMENT
        print(f"Using AZURE_OPENAI_DEPLOYMENT={target!r} (set AZURE_OPENAI_EMBEDDING_DEPLOYMENT for clarity)")
    else:
        for d in deployments:
            model = str(d.get("model") or d.get("properties", {}).get("model", "")).lower()
            name = d.get("id") or d.get("name")
            if "embedding" in model or "text-embedding" in model:
                target = name
                print(f"Auto-selected embedding deployment={target!r} (model hint={model!r})")
                break
        if not target and deployments:
            target = deployments[0].get("id") or deployments[0].get("name")
            print(f"WARN: No obvious embedding deployment; trying first: {target!r}")

    if not target:
        print(
            "FAIL: No embedding deployment to test. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT "
            "to your Azure deployment name for text-embedding-3-small (or similar).",
            file=sys.stderr,
        )
        return 1

    print(f"Calling embeddings.create (api_version={API_VERSION!r})...")
    try:
        client = AzureOpenAI(
            api_key=KEY,
            api_version=API_VERSION,
            azure_endpoint=base,
        )
        r = client.embeddings.create(model=target, input="ping")
        vec = r.data[0].embedding
        print(f"OK: embedding len={len(vec)} (deployment={target!r})")
    except Exception as e:
        print(f"FAIL: embeddings.create: {type(e).__name__}: {e}", file=sys.stderr)
        if "does not work with the specified model" in str(e).lower():
            print(
                "HINT: That deployment is probably a chat model. In Azure Portal → your OpenAI "
                "resource → Model deployments, copy the deployment name for text-embedding-3-small "
                "into AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
