#!/usr/bin/env bash

# Source this file, do not execute it:
#   source scripts/delphes/setup_lpc_delphes_env.sh

_LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"

if [ ! -r "$_LCG_VIEW" ]; then
  echo "ERROR: Cannot read LCG setup file: $_LCG_VIEW"
  return 1 2>/dev/null || exit 1
fi

source "$_LCG_VIEW"

export HH4B_REPO="/uscms_data/d3/${USER}/repos/hh-bbww-baselines"
export HH4B_STORE="/uscms_data/d3/${USER}/hh4b_delphes"
export HH4B_SOFTWARE="/uscms_data/d3/${USER}/software"
export DELPHES_DIR="${HH4B_SOFTWARE}/Delphes"

mkdir -p "$HH4B_STORE" "$HH4B_SOFTWARE"

echo "HH4B_REPO=${HH4B_REPO}"
echo "HH4B_STORE=${HH4B_STORE}"
echo "HH4B_SOFTWARE=${HH4B_SOFTWARE}"
echo "DELPHES_DIR=${DELPHES_DIR}"
echo "ROOT=$(root-config --version)"
echo "Python=$(python3 --version)"
