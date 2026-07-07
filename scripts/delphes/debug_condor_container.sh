#!/usr/bin/env bash
set -euo pipefail
set -x

echo "HOSTNAME=$(hostname)"
echo "USER=${USER:-unknown}"
echo "PWD=$PWD"
echo "HOME=$HOME"

echo "---- key paths ----"
for path in / /srv /cvmfs /cvmfs/sft.cern.ch /cvmfs/cms.cern.ch /uscms_data /uscms /eos; do
  if [ -e "$path" ]; then
    ls -ld "$path"
  else
    echo "MISSING: $path"
  fi
done

echo "---- environment snippets ----"
env | sort | grep -E "CONDOR|APPTAINER|SINGULARITY|X509|USER|HOME|PWD" || true

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
echo "---- LCG_106 setup ----"
if [ ! -r "$LCG_SETUP" ]; then
  echo "ERROR: cannot read $LCG_SETUP"
  exit 2
fi
echo "OK: readable $LCG_SETUP"

# shellcheck disable=SC1090
set +u
source "$LCG_SETUP"
set -u

echo "---- required commands after LCG_106 ----"
for cmd in python3 root-config pythia8-config HepMC3-config mg5_aMC; do
  if command -v "$cmd"; then
    echo "AVAILABLE: $cmd -> $(command -v "$cmd")"
    set +e
    case "$cmd" in
      python3) python3 --version ;;
      root-config) root-config --version ;;
      pythia8-config) pythia8-config --version ;;
      HepMC3-config) HepMC3-config --version ;;
      mg5_aMC) mg5_aMC --version 2>&1 | head -20 ;;
    esac
    version_status=$?
    set -e
    echo "VERSION_STATUS: $cmd -> $version_status"
  else
    echo "MISSING: $cmd"
  fi
done

echo "---- DelphesHepMC3 search ----"
set +e
DELPHES_MATCHES="$(timeout 120s find /cvmfs/sft.cern.ch -name DelphesHepMC3 2>/dev/null | head -20)"
find_status=$?
set -e
if [ -n "$DELPHES_MATCHES" ]; then
  printf '%s\n' "$DELPHES_MATCHES"
else
  echo "MISSING: DelphesHepMC3 not found under /cvmfs/sft.cern.ch"
fi
echo "FIND_STATUS: DelphesHepMC3 -> $find_status"

echo "DONE worker tool debug"
