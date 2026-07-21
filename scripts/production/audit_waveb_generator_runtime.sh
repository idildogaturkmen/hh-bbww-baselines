#!/usr/bin/env bash

set -Eeuo pipefail

REPO="/uscms_data/d3/$USER/repos/hh-bbww-baselines"
STORE="/uscms_data/d3/$USER/hh4b_delphes"

OUTPUT_DIR="$REPO/outputs/agent_runs/unified_background_5m_waveb_generator_runtime_20260721"
OUTPUT_LOG="$OUTPUT_DIR/generator_runtime.txt"
MG5_LIST="$OUTPUT_DIR/mg5_candidates.txt"

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"

mkdir -p "$OUTPUT_DIR"

test -s "$LCG_SETUP"

set +u
source "$LCG_SETUP"
set -u

unset SOURCE 2>/dev/null || true
hash -r

{
  echo "============================================================"
  echo "Wave-B generator runtime audit"
  echo "time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "============================================================"

  echo
  echo "=== Repository ==="

  LOCAL_HEAD="$(git -C "$REPO" rev-parse HEAD)"
  REMOTE_HEAD="$(
    git -C "$REPO" rev-parse origin/delphes-hh4b-production
  )"

  echo "local_head=$LOCAL_HEAD"
  echo "remote_head=$REMOTE_HEAD"

  test "$LOCAL_HEAD" = "$REMOTE_HEAD"
  test -z "$(git -C "$REPO" status --porcelain)"

  echo
  echo "=== Python and compilers ==="

  echo "python=$(command -v python3)"
  python3 --version

  echo "gcc=$(command -v gcc)"
  gcc --version |
  head -n 1

  echo "gfortran=$(command -v gfortran)"
  gfortran --version |
  head -n 1

  echo
  echo "=== Search for global mg5_aMC executables ==="

  {
    find \
      /cvmfs/sft.cern.ch/lcg/releases/MCGenerators/madgraph5amc \
      -type f \
      -path '*/bin/mg5_aMC' \
      -perm -u+x \
      2>/dev/null

    find \
      "/uscms_data/d3/$USER/software" \
      -maxdepth 10 \
      -type f \
      -path '*/bin/mg5_aMC' \
      -perm -u+x \
      2>/dev/null
  } |
  sort -u |
  tee "$MG5_LIST"

  MG5_COUNT="$(
    grep -c . "$MG5_LIST" 2>/dev/null ||
    true
  )"

  echo "mg5_candidate_count=$MG5_COUNT"

  if test "$MG5_COUNT" -eq 0; then
    echo "ERROR: no global mg5_aMC executable was found"
    exit 20
  fi

  echo
  echo "=== MG5 candidate version probes ==="

  VALID_FILE="$OUTPUT_DIR/mg5_353_candidates.txt"
  : > "$VALID_FILE"

  while IFS= read -r MG5_BIN
  do
    test -n "$MG5_BIN" || continue

    echo
    echo "candidate=$MG5_BIN"

    PROBE_LOG="$OUTPUT_DIR/$(echo "$MG5_BIN" | sha256sum | awk '{print $1}').log"

    timeout 60 \
      bash -c "printf 'quit\n' | \"\$1\"" \
      _ \
      "$MG5_BIN" \
      > "$PROBE_LOG" \
      2>&1 \
    || true

    sed -n '1,45p' "$PROBE_LOG"

    if grep -Eq \
      '(^|[^0-9])3[.]5[.]3([^0-9]|$)' \
      "$PROBE_LOG"
    then
      echo "$MG5_BIN" >> "$VALID_FILE"
      echo "version_353=true"
    else
      echo "version_353=false"
    fi
  done < "$MG5_LIST"

  VALID_COUNT="$(
    grep -c . "$VALID_FILE" 2>/dev/null ||
    true
  )"

  echo
  echo "mg5_353_candidate_count=$VALID_COUNT"

  if test "$VALID_COUNT" -eq 0; then
    echo "ERROR: no working MG5_aMC 3.5.3 executable was found"
    exit 21
  fi

  SELECTED_MG5="$(
    grep \
      'x86_64-el9-gcc13-opt' \
      "$VALID_FILE" |
    head -n 1
  )"

  if test -z "$SELECTED_MG5"; then
    SELECTED_MG5="$(
      head -n 1 "$VALID_FILE"
    )"
  fi

  test -x "$SELECTED_MG5"

  echo "selected_mg5=$SELECTED_MG5"

  echo
  echo "=== LHAPDF runtime ==="

  if command -v lhapdf-config >/dev/null 2>&1; then
    echo "lhapdf_config=$(command -v lhapdf-config)"
    echo "lhapdf_version=$(lhapdf-config --version)"
    echo "lhapdf_datadir=$(lhapdf-config --datadir)"

    echo
    echo "Installed NNPDF candidates:"

    find "$(lhapdf-config --datadir)" \
      -maxdepth 1 \
      -type d \
      -name 'NNPDF*' \
      -printf '%f\n' |
    sort |
    head -n 100
  else
    echo "lhapdf_config=not_found"
  fi

  echo
  echo "=== Pythia runtime ==="

  if command -v pythia8-config >/dev/null 2>&1; then
    echo "pythia8_config=$(command -v pythia8-config)"
    echo "pythia8_version=$(pythia8-config --version)"
  else
    echo "pythia8_config=not_found"
  fi

  echo
  echo "=== Existing run-card beam and PDF settings ==="

  for RUN_CARD in \
    "$STORE/mg5/TTbar_smoke/Cards/run_card.dat" \
    "$STORE/mg5/Zbbbb_presel_smoke/Cards/run_card.dat" \
    "$STORE/mg5/HH4b_smoke_vbf/Cards/run_card.dat"
  do
    echo
    echo "FILE=$RUN_CARD"

    test -s "$RUN_CARD"

    grep -E \
      '=[[:space:]]*(ebeam1|ebeam2|lpp1|lpp2|pdlabel|lhaid|ickkw|event_norm|dynamical_scale_choice|scalefact|alpsfact|cut_decays|ptj|ptb|etaj|etab|drbb|drbj|ihtmin|ihtmax)' \
      "$RUN_CARD" \
    || true
  done

  echo
  echo "=== Existing process versions ==="

  for PROCESS_DIR in \
    "$STORE/mg5/TTbar_smoke" \
    "$STORE/mg5/TTbb_smoke" \
    "$STORE/mg5/ZH4b_smoke" \
    "$STORE/mg5/ZZ4b_smoke"
  do
    VERSION_FILE="$PROCESS_DIR/MGMEVersion.txt"

    printf '%s=' "$PROCESS_DIR"

    if test -s "$VERSION_FILE"; then
      tr -d '\r\n' < "$VERSION_FILE"
      echo
    else
      echo "missing"
    fi
  done

  echo
  echo "=== Storage ==="

  df -h "$STORE"
  quota -s 2>/dev/null || true

  echo
  echo "selected_mg5=$SELECTED_MG5"
  echo "lcg_setup=$LCG_SETUP"
  echo "WAVEB_GENERATOR_RUNTIME_AUDIT_PASS"
} |
tee "$OUTPUT_LOG"

echo
echo "runtime_log=$OUTPUT_LOG"
