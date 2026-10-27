#!/usr/bin/env bash
# Provisioning log for IntelliClinic AI build host.
set -u
LOG="D:/Hermes Codes/intelliclinic-ai-provision.log"
PY="python3.12"

echo "=== provision start $(date -u +%FT%TZ) ===" > "$LOG"

echo "--- pip install fastapi stack ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check \
  fastapi "uvicorn[standard]" pydantic pydantic-settings email-validator \
  httpx python-multipart jinja2 >> "$LOG" 2>&1
echo "fastapi_exit=$?" >> "$LOG"

echo "--- chromadb ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check chromadb >> "$LOG" 2>&1
echo "chromadb_exit=$?" >> "$LOG"

echo "--- sentence-transformers ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check sentence-transformers >> "$LOG" 2>&1
echo "sentence_transformers_exit=$?" >> "$LOG"

echo "--- sklearn (real intent model) ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check scikit-learn >> "$LOG" 2>&1
echo "sklearn_exit=$?" >> "$LOG"

echo "--- pytest ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check pytest pytest-asyncio >> "$LOG" 2>&1
echo "pytest_exit=$?" >> "$LOG"

echo "--- torch repair attempt (cpu wheel) ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check --force-reinstall \
  --index-url https://download.pytorch.org/whl/cpu torch >> "$LOG" 2>&1
echo "torch_exit=$?" >> "$LOG"

echo "--- transformers ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check transformers >> "$LOG" 2>&1
echo "transformers_exit=$?" >> "$LOG"

echo "--- verify ---" >> "$LOG"
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import importlib
for m in ["fastapi","pydantic","chromadb","sentence_transformers","sklearn","torch","transformers"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {str(e)[:200]}")
PYEOF

echo "=== provision end $(date -u +%FT%TZ) ===" >> "$LOG"
