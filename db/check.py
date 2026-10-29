"""Verify the local setup before we build anything on top of it.

    python -m db.check

Checks, in order: the pinned library versions, that .env is present and
parseable, that Supabase answers, and that every table the product needs
exists. Each failure prints what to do about it rather than a traceback.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

REQUIRED_TABLES = ["datasets", "runs", "sequences", "predictions"]
GONE_TABLES = ["run_artifacts", "sequence_metadata"]


def main() -> int:
    ok = True

    # --- versions ----------------------------------------------------------
    import numpy
    import torch

    print(f"torch {torch.__version__}  numpy {numpy.__version__}")
    if not torch.__version__.startswith("2.4.1"):
        print("  ! expected torch 2.4.1 -- Render pins this; checkpoints "
              "trained on another version may not load there")
        ok = False
    if not numpy.__version__.startswith("1.26"):
        print("  ! expected numpy 1.26.x")
        ok = False

    # --- credentials -------------------------------------------------------
    load_dotenv()
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not url or not key:
        print("! .env is missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        return 1
    if url.rstrip("/").endswith("/rest/v1"):
        print("! SUPABASE_URL must be the bare project URL, with no /rest/v1")
        return 1
    if not key.startswith("eyJ"):
        print("! SUPABASE_SERVICE_KEY does not look like a legacy JWT. Use the "
              "Legacy API keys tab, not the new sb_secret_ format.")
        return 1
    print(f"env  {url}")

    # --- connectivity + schema --------------------------------------------
    from supabase import create_client

    client = create_client(url, key)

    for table in REQUIRED_TABLES:
        try:
            res = client.table(table).select("*", count="exact").limit(1).execute()
            print(f"  {table:<12} ok  ({res.count} rows)")
        except Exception as exc:                      # noqa: BLE001
            print(f"  {table:<12} MISSING -- {type(exc).__name__}")
            ok = False

    for table in GONE_TABLES:
        try:
            client.table(table).select("id").limit(1).execute()
            print(f"  {table:<12} still present -- 002 did not finish")
            ok = False
        except Exception:                             # noqa: BLE001
            pass

    print("\nready" if ok else "\nsomething above needs fixing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())