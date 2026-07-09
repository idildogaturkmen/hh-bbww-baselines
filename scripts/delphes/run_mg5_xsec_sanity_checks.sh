#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_mg5_xsec_sanity_checks_v2.sh" "$@"
