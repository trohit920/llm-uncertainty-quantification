#!/usr/bin/env bash
#
# End-to-end verification of the Track 2 submission.
#
# Runs the same path a reviewer would: environment check, unit tests, artifact
# regeneration from recorded runs, and a completeness check against the
# assignment's required deliverables. Exits non-zero if any stage fails.
#
# Usage:
#   ./verify_submission.sh              # fast: tests + regenerate from records (~2 min)
#   ./verify_submission.sh --full       # also re-runs generation from scratch (~40 min)
#   ./verify_submission.sh --quick-gen  # re-runs generation on 15 examples (~6 min)

set -uo pipefail

PYTHON="${PYTHON:-../.venv/bin/python}"
MODE="fast"
[[ "${1:-}" == "--full" ]] && MODE="full"
[[ "${1:-}" == "--quick-gen" ]] && MODE="quick"

FAILURES=0
STAGE=0

rule() { printf '\n%s\n%s\n' "$(printf '=%.0s' {1..72})" "$1"; }

check() {
    local label="$1"; shift
    STAGE=$((STAGE + 1))
    printf '  [%d] %-52s' "$STAGE" "$label"
    if "$@" >/tmp/verify_stage.log 2>&1; then
        printf 'PASS\n'
    else
        printf 'FAIL\n'
        sed 's/^/      /' /tmp/verify_stage.log | tail -12
        FAILURES=$((FAILURES + 1))
    fi
}

exists() { [[ -s "$1" ]]; }

count_at_least() {
    local pattern="$1" minimum="$2"
    local found
    found=$(find . -path "$pattern" -type f 2>/dev/null | wc -l)
    [[ "$found" -ge "$minimum" ]]
}

records_at_least() {
    local file="$1" minimum="$2"
    [[ -s "$file" ]] && [[ "$(wc -l < "$file")" -ge "$minimum" ]]
}

# --------------------------------------------------------------------------
rule "1. ENVIRONMENT"
# --------------------------------------------------------------------------
check "Python interpreter present"        test -x "$PYTHON"
check "torch imports"                     "$PYTHON" -c "import torch"
check "transformers imports"              "$PYTHON" -c "import transformers"
check "datasets imports"                  "$PYTHON" -c "import datasets"
check "CUDA available (optional)"         "$PYTHON" -c "import torch; assert torch.cuda.is_available()"
printf '      note: CUDA failure above is non-fatal; CPU works but is slow\n'

# --------------------------------------------------------------------------
rule "2. UNIT TESTS"
# --------------------------------------------------------------------------
check "pytest suite"                      "$PYTHON" -m pytest -q

# --------------------------------------------------------------------------
rule "3. GENERATION"
# --------------------------------------------------------------------------
if [[ "$MODE" == "full" ]]; then
    check "full run (200 NQ + 100 GSM8K)" "$PYTHON" scripts/run_experiment.py
elif [[ "$MODE" == "quick" ]]; then
    check "quick run (15 per task)"       "$PYTHON" scripts/run_experiment.py --quick
else
    printf '      skipped (use --full or --quick-gen to re-run generation)\n'
fi
check "NQ-Open records present"           records_at_least results/records_nq_open.jsonl 15
check "GSM8K records present"             records_at_least results/records_gsm8k.jsonl 15

# --------------------------------------------------------------------------
rule "4. ANALYSIS AND ARTIFACTS"
# --------------------------------------------------------------------------
check "full artifact pipeline (run_all)"   "$PYTHON" scripts/run_all.py --skip-generation
check "applications demo"                 "$PYTHON" scripts/demo_applications.py
check "self-consistency demo"             "$PYTHON" scripts/demo_self_consistency.py

# --------------------------------------------------------------------------
rule "5. REQUIRED DELIVERABLES"
# --------------------------------------------------------------------------
check "README.md"                         exists README.md
check "requirements.txt"                  exists requirements.txt
check "report.md (analysis report)"       exists report.md
check "study_guide.pdf"                   exists study_guide.pdf
check "executed notebook"                 exists notebooks/uncertainty_analysis.ipynb
check "at least 15 figures"               count_at_least "./figures/*.png" 15
check "metrics JSON per task"             count_at_least "./results/metrics_*.json" 2

# --------------------------------------------------------------------------
rule "6. SUBMISSION HYGIENE"
# --------------------------------------------------------------------------
check "no API keys or credentials"        bash -c '! grep -rIlE "(api[_-]?key *=|sk-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{30})" --include="*.py" --include="*.md" --include="*.txt" . 2>/dev/null | grep -q .'
check "no .env file"                      bash -c '! test -f .env'
check "no venv committed into tree"       bash -c '! test -d .venv'
check "no CLAUDE.md in submission"        bash -c '! test -f CLAUDE.md'

# --------------------------------------------------------------------------
rule "RESULT"
# --------------------------------------------------------------------------
if [[ "$FAILURES" -eq 0 ]]; then
    printf '  ALL %d CHECKS PASSED - ready to package\n\n' "$STAGE"
    exit 0
fi
printf '  %d of %d checks FAILED - do not submit yet\n\n' "$FAILURES" "$STAGE"
exit 1
