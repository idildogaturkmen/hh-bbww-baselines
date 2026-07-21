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
  echo "=== Deterministic MG5 selection ==="

  EXPECTED_MG5="/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/madgraph5amc/3.5.3.atlas7-2c347/x86_64-el9-gcc13-opt/bin/mg5_aMC"

  VIEW_MG5="$(
    command -v mg5_aMC 2>/dev/null ||
    true
  )"

  if test -n "$VIEW_MG5" \
    && test -x "$VIEW_MG5"
  then
    SELECTED_MG5="$(
      readlink -f "$VIEW_MG5"
    )"

    SELECTION_SOURCE="LCG_106_PATH"
  elif test -x "$EXPECTED_MG5"
  then
    SELECTED_MG5="$EXPECTED_MG5"
    SELECTION_SOURCE="EXPECTED_LCG106_BUILD"
  else
    echo "ERROR: expected MG5 3.5.3 executable is unavailable"
    echo "expected_mg5=$EXPECTED_MG5"
    exit 20
  fi

  printf '%s\n' "$SELECTED_MG5" \
    > "$MG5_LIST"

  echo "mg5_candidate_count=1"
  echo "selection_source=$SELECTION_SOURCE"
  echo "selected_mg5=$SELECTED_MG5"

  test -x "$SELECTED_MG5"

  echo
  echo "=== Selected MG5 version probe ==="

  PROBE_LOG="$OUTPUT_DIR/mg5_selected_probe.log"
  PROBE_RC=0

  env \
    -u PYTHONHOME \
    -u PYTHONPATH \
    -u PYTHONSTARTUP \
    -u PYTHONUSERBASE \
    timeout 60 \
    bash -c '
      printf "quit\n" |
      "$1"
    ' \
    _ \
    "$SELECTED_MG5" \
    > "$PROBE_LOG" \
    2>&1 \
  || PROBE_RC=$?

  echo "mg5_probe_exit_code=$PROBE_RC"

  sed -n '1,60p' "$PROBE_LOG"

  if test "$PROBE_RC" -ne 0; then
    echo "ERROR: selected MG5 executable did not start cleanly"
    exit 21
  fi

  if ! grep -Eq \
    '(^|[^0-9])3[.]5[.]3([^0-9]|$)' \
    "$PROBE_LOG"
  then
    echo "ERROR: selected executable did not report MG5 3.5.3"
    exit 22
  fi

  echo "version_353=true"

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
    sed -n '1,100p'
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
