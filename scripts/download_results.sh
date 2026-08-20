#!/usr/bin/env bash
# =============================================================================
# BackdoorDM — Download pre-trained backdoored models from HuggingFace Hub.
#
# Hosted at: https://huggingface.co/Weilin0/BackdoorDM
# The weights mirror the `./results` layout used by the codebase, so downloaded
# files can be used directly by evaluation/defense/analysis without any change.
#
# Usage:
#   bash scripts/download_results.sh                      # interactive: pick methods/versions
#   bash scripts/download_results.sh --all                # download everything (SD15 + SD20)
#   bash scripts/download_results.sh --method eviledit    # one method, both versions
#   bash scripts/download_results.sh --method eviledit --version sd15
#   bash scripts/download_results.sh --method badt2i_object,badt2i_pixel --version sd15,sd20
#   bash scripts/download_results.sh --method rickrolling_TPA --version sd15 --target ./weights
#
# Options:
#   -m, --method METHOD[,METHOD...]   attack methods to download (comma-separated)
#   -v, --version VER[,VER...]        model versions: sd15, sd20 (default: both)
#   -a, --all                         download all released methods & versions
#   -t, --target DIR                  target directory (default: ./results)
#   -f, --force                       re-download even if files already exist
#   -h, --help                        show this help
#
# Requires: huggingface_hub (pip install huggingface_hub) OR huggingface-cli.
# Note: bash >= 3.2 compatible (no associative arrays).
# =============================================================================
set -euo pipefail

REPO_ID="Weilin0/BackdoorDM"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${ROOT_DIR}/results"
FORCE=0

# --- Available methods (method name -> result dir prefix) ---------------------
# Keep in sync with evaluation/configs/bdmodel_path.py
# Parallel arrays: METHOD_NAMES[i] <-> METHOD_PREFIX[i]
METHOD_NAMES=(badt2i_object badt2i_pixel badt2i_style bibaddiff eviledit paas_db paas_ti rickrolling_TAA rickrolling_TPA)
METHOD_PREFIX=(badt2i_object badt2i_pixel badt2i_style bibaddiff eviledit paas_db paas_ti rickrolling_TAA rickrolling_TPA)
VERSIONS=(sd15 sd20)

# Methods that have NO sd20 release (only sd15).
NO_SD20=(bibaddiff)

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}"; }

contains() {  # contains <element> <list...>
  local e="$1"; shift
  for x in "$@"; do [[ "$x" == "$e" ]] && return 0; done
  return 1
}

method_prefix() {  # echo the result-dir prefix for a method
  local m="$1" i
  for i in "${!METHOD_NAMES[@]}"; do
    [[ "${METHOD_NAMES[$i]}" == "$m" ]] && { echo "${METHOD_PREFIX[$i]}"; return 0; }
  done
  echo ""
}

# --- Parse args ---------------------------------------------------------------
SELECTED_METHODS=()
SELECTED_VERSIONS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--method)  IFS=',' read -ra m <<< "$2"; SELECTED_METHODS+=("${m[@]}"); shift 2;;
    -v|--version) IFS=',' read -ra v <<< "$2"; SELECTED_VERSIONS+=("${v[@]}"); shift 2;;
    -a|--all)     SELECTED_METHODS=("${METHOD_NAMES[@]}"); shift;;
    -t|--target)  TARGET="$2"; shift 2;;
    -f|--force)   FORCE=1; shift;;
    -h|--help)    usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

# --- Validate ------------------------------------------------------------
for m in "${SELECTED_METHODS[@]}"; do
  contains "$m" "${METHOD_NAMES[@]}" || { echo "Error: unknown method '$m'. Valid: ${METHOD_NAMES[*]}"; exit 1; }
done
for v in "${SELECTED_VERSIONS[@]}"; do
  contains "$v" "${VERSIONS[@]}" || { echo "Error: unknown version '$v'. Valid: ${VERSIONS[*]}"; exit 1; }
done

# --- Interactive selection (when neither --method nor --all given) ------------
if [[ ${#SELECTED_METHODS[@]} -eq 0 ]]; then
  echo "BackdoorDM — select the backdoored models to download:"
  echo "  repo: ${REPO_ID}"
  echo ""
  for i in "${!METHOD_NAMES[@]}"; do
    echo "  [$((i+1))] ${METHOD_NAMES[$i]}"
  done
  echo "  [0] all"
  echo ""
  read -rp "Enter numbers (e.g. 1 3 5, or 0 for all): " -a picks
  for p in "${picks[@]}"; do
    if [[ "$p" == "0" ]]; then
      SELECTED_METHODS=("${METHOD_NAMES[@]}")
      break
    elif [[ "$p" =~ ^[0-9]+$ ]] && (( p >= 1 && p <= ${#METHOD_NAMES[@]} )); then
      SELECTED_METHODS+=("${METHOD_NAMES[$((p-1))]}")
    else
      echo "Invalid selection: '$p' (ignored)"
    fi
  done
fi

if [[ ${#SELECTED_VERSIONS[@]} -eq 0 ]]; then
  SELECTED_VERSIONS=("${VERSIONS[@]}")
fi

echo ""
echo "BackdoorDM download"
echo "  target:   ${TARGET}"
echo "  methods:  ${SELECTED_METHODS[*]}"
echo "  versions: ${SELECTED_VERSIONS[*]}"
echo ""

# --- Tooling check ------------------------------------------------------------
if python3 -c "import huggingface_hub" 2>/dev/null; then
  PY_DL=1
elif command -v huggingface-cli >/dev/null 2>&1; then
  PY_DL=0
else
  echo "Error: neither 'huggingface_hub' (pip install huggingface_hub) nor 'huggingface-cli' is available."
  exit 1
fi

mkdir -p "$TARGET"

total=0
for m in "${SELECTED_METHODS[@]}"; do
  prefix="$(method_prefix "$m")"
  for v in "${SELECTED_VERSIONS[@]}"; do
    if contains "$v" sd20 && contains "$m" "${NO_SD20[@]}"; then
      echo "[skip] ${m}_${v}: no sd20 release yet (only sd15)."
      continue
    fi
    dir="${prefix}_${v}"
    subpath="results/${dir}"
    total=$((total+1))

    echo "[download] ${dir}  (${REPO_ID}/${subpath})"
    if [[ "$PY_DL" == "1" ]]; then
      if [[ "$FORCE" == "1" ]]; then
        python3 - "$subpath" "$TARGET" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
subpath, target = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id="Weilin0/BackdoorDM",
    allow_patterns=[f"{subpath}/**"],
    local_dir=target,
    force_download=True,
)
PYEOF
      else
        python3 - "$subpath" "$TARGET" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
subpath, target = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id="Weilin0/BackdoorDM",
    allow_patterns=[f"{subpath}/**"],
    local_dir=target,
)
PYEOF
      fi
    else
      hf_args=(download "$REPO_ID" --local-dir "$TARGET" --include "$subpath/**")
      [[ "$FORCE" == "1" ]] && hf_args+=(--force)
      huggingface-cli "${hf_args[@]}"
    fi
  done
done

echo ""
echo "Done. Downloaded ${total} model(s) into '${TARGET}'."
