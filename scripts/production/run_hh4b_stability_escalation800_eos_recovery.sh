#!/usr/bin/env bash
set -euo pipefail

# Recovery transport for accepted cluster 30020809. The schedd cannot read the
# login-node EOS FUSE path, so download the already-frozen payload inside the
# execute sandbox and then delegate to the unchanged production runner.

EOS_XROOTD_ROOT="root://cmseos.fnal.gov//store/user/iturkmen/hh4b_cut_baseline/selection_stability_escalation800_production_package_v1_20260809/archives"
FROZEN_RUNNER_NAME="run_escalation800_category_fold_job.sh"
FROZEN_RUNNER_SHA256="b5a0f96950f1b2cba99e1ed173b33a70630f10f2e1c516090920dea7137c15e5"

download_payload() {
    local payload_basename="$1"
    local expected_sha256="$2"
    local destination_root="$3"

    [[ "$payload_basename" =~ ^escalation800_payload_replica_[0-9]{4}__(exact3tag|ge4tag)\.tar\.gz$ ]]
    [[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]
    [[ -d "$destination_root" ]]
    command -v xrdcp >/dev/null 2>&1

    local destination="$destination_root/$payload_basename"
    local partial="$destination.partial"
    [[ ! -e "$destination" ]]
    [[ ! -e "$partial" ]]

    xrdcp --force --nopbar "$EOS_XROOTD_ROOT/$payload_basename" "$partial"

    local observed_sha256
    observed_sha256="$(sha256sum "$partial" | awk '{print $1}')"
    [[ "$observed_sha256" == "$expected_sha256" ]]
    mv "$partial" "$destination"
}

if [[ "${1:-}" == "--download-only-smoke" ]]; then
    [[ "$#" -eq 4 ]]
    download_payload "$2" "$3" "$4"
    exit 0
fi

[[ "$#" -eq 8 ]]
payload_basename="$6"
expected_payload_sha256="$7"

download_payload "$payload_basename" "$expected_payload_sha256" "$PWD"

frozen_runner="$PWD/$FROZEN_RUNNER_NAME"
[[ -f "$frozen_runner" ]]
observed_runner_sha256="$(sha256sum "$frozen_runner" | awk '{print $1}')"
[[ "$observed_runner_sha256" == "$FROZEN_RUNNER_SHA256" ]]

exec /usr/bin/bash "$frozen_runner" "$@"
