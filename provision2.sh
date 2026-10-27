#!/usr/bin/env bash
# Retry the packages that failed on ENOSPC, with cache+tmp pinned to D: so a
# full system volume cannot abort the install.
set -u
LOG="D:/Hermes Codes/intelliclinic-ai-provision2.log"
PY="python3.12"
CACHE="D:/Hermes Codes/.pip-cache"
TMPD="D:/Hermes Codes/.pip-tmp"
mkdir -p "$CACHE" "$TMPD"
export PIP_CACHE_DIR="$CACHE"
export TMPDIR="$TMPD"
export TEMP="$TMPD"
export TMP="$TMPD"

echo "=== retry start $(date -u +%FT%TZ) ===" > "$LOG"
df -h / >> "$LOG" 2>&1

run() {
  local label="$1"; shift
  echo "--- $label ---" >> "$LOG"
  "$PY" -m pip install --quiet --disable-pip-version-check --no-warn-script-location "$@" >> "$LOG" 2>&1
  echo "${label}_exit=$?" >> "$LOG"
}

run sklearn scikit-learn
run sentence_transformers sentence-transformers
run transformers transformers

echo "--- torch (cpu index) ---" >> "$LOG"
"$PY" -m pip install --quiet --disable-pip-version-check --no-warn-script-location \
  --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch >> "$LOG" 2>&1
echo "torch_exit=$?" >> "$LOG"

echo "--- verify ---" >> "$LOG"
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import importlib, sys
for m in ["fastapi","pydantic","chromadb","sentence_transformers","sklearn","torch","transformers"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {str(e)[:160]}")
PYEOF
df -h / >> "$LOG" 2>&1
echo "=== retry end $(date -u +%FT%TZ) ===" >> "$LOG"
