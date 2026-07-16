#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 SOURCE OUTPUT [ARGUMENT_LOG]" >&2
  exit 2
fi

GENERATOR_SOURCE="$1"
GENERATOR_OUTPUT="$2"
ARGUMENT_LOG="${3:-}"
CONFIG_MODE="${HH4B_CONFIG_MODE:-auto}"

case "$CONFIG_MODE" in
  auto|pkg-config|config-tools)
    ;;
  *)
    echo "ERROR: HH4B_CONFIG_MODE must be auto, pkg-config, or config-tools" >&2
    exit 2
    ;;
esac

test -f "$GENERATOR_SOURCE" || {
  echo "ERROR: missing generator source: $GENERATOR_SOURCE" >&2
  exit 3
}

case "$GENERATOR_SOURCE" in
  *.cc|*.cpp|*.cxx)
    ;;
  *)
    echo "ERROR: unexpected generator source suffix: $GENERATOR_SOURCE" >&2
    exit 3
    ;;
esac

GXX=$(command -v g++) || {
  echo "ERROR: g++ is unavailable" >&2
  exit 4
}

split_flags() {
  local raw_flags="$1"
  local array_name="$2"
  local -n output_array="$array_name"

  output_array=()

  if [[ "$raw_flags" == *$'\n'* ]]; then
    echo "ERROR: configuration command returned multiple lines" >&2
    exit 5
  fi

  if [[ -n "$raw_flags" ]]; then
    read -r -a output_array <<< "$raw_flags"
  fi
}

query_flags() {
  local array_name="$1"
  shift
  local raw_flags

  if raw_flags=$("$@"); then
    split_flags "$raw_flags" "$array_name"
  else
    local status=$?
    echo "ERROR: configuration command failed ($status): $*" >&2
    exit 5
  fi
}

find_pkg_config_module() {
  local array_name="$1"
  shift
  local -n selected_module="$array_name"
  local candidate

  selected_module=""

  command -v pkg-config >/dev/null 2>&1 || return 1

  for candidate in "$@"; do
    if pkg-config --exists "$candidate"; then
      selected_module="$candidate"
      return 0
    fi
  done

  return 1
}

PYTHIA_MODULE=""
HEPMC3_MODULE=""

if [[ "$CONFIG_MODE" != "config-tools" ]]; then
  find_pkg_config_module PYTHIA_MODULE pythia8 Pythia8 || true
  find_pkg_config_module HEPMC3_MODULE HepMC3 hepmc3 || true
fi

declare -a PYTHIA_CXXFLAGS=()
declare -a PYTHIA_LIBS=()
declare -a HEPMC3_CXXFLAGS=()
declare -a HEPMC3_LIBS=()

if [[ -n "$PYTHIA_MODULE" ]]; then
  PYTHIA_METHOD="pkg-config:$PYTHIA_MODULE"
  query_flags PYTHIA_CXXFLAGS pkg-config --cflags "$PYTHIA_MODULE"
  query_flags PYTHIA_LIBS pkg-config --libs "$PYTHIA_MODULE"
elif [[ "$CONFIG_MODE" == "pkg-config" ]]; then
  echo "ERROR: no Pythia 8 pkg-config module is available" >&2
  exit 5
else
  command -v pythia8-config >/dev/null || {
    echo "ERROR: neither Pythia 8 pkg-config nor pythia8-config is available" >&2
    exit 5
  }
  PYTHIA_METHOD="pythia8-config"
  query_flags PYTHIA_CXXFLAGS pythia8-config --cxxflags
  query_flags PYTHIA_LIBS pythia8-config --libs
fi

if [[ -n "$HEPMC3_MODULE" ]]; then
  HEPMC3_METHOD="pkg-config:$HEPMC3_MODULE"
  query_flags HEPMC3_CXXFLAGS pkg-config --cflags "$HEPMC3_MODULE"
  query_flags HEPMC3_LIBS pkg-config --libs "$HEPMC3_MODULE"
elif [[ "$CONFIG_MODE" == "pkg-config" ]]; then
  echo "ERROR: no HepMC3 pkg-config module is available" >&2
  exit 5
else
  command -v HepMC3-config >/dev/null || {
    echo "ERROR: neither HepMC3 pkg-config nor HepMC3-config is available" >&2
    exit 5
  }
  HEPMC3_METHOD="HepMC3-config"
  query_flags HEPMC3_CXXFLAGS HepMC3-config --cxxflags
  query_flags HEPMC3_LIBS HepMC3-config --libs
fi

COMPILE_ARGS=(
  "${PYTHIA_CXXFLAGS[@]}"
  "${HEPMC3_CXXFLAGS[@]}"
  -O2
  -std=c++17
  "$GENERATOR_SOURCE"
  -o
  "$GENERATOR_OUTPUT"
  "${PYTHIA_LIBS[@]}"
  "${HEPMC3_LIBS[@]}"
)

for argument in "$GXX" "${COMPILE_ARGS[@]}"; do
  if [[ "$argument" == *".sh"* ]]; then
    echo "ERROR: refusing unexpected .sh compiler argument: $argument" >&2
    exit 6
  fi
done

log_arguments() {
  local index=0
  local argument

  printf 'pythia_configuration=%s\n' "$PYTHIA_METHOD"
  printf 'hepmc3_configuration=%s\n' "$HEPMC3_METHOD"
  printf 'compile_executable=%q\n' "$GXX"

  for argument in "${COMPILE_ARGS[@]}"; do
    printf 'compile_arg[%03d]=%q\n' "$index" "$argument"
    index=$((index + 1))
  done
}

log_arguments

if [[ -n "$ARGUMENT_LOG" ]]; then
  mkdir -p "$(dirname "$ARGUMENT_LOG")"
  log_arguments > "$ARGUMENT_LOG"
fi

"$GXX" "${COMPILE_ARGS[@]}"

test -x "$GENERATOR_OUTPUT" || {
  echo "ERROR: compiler did not create an executable: $GENERATOR_OUTPUT" >&2
  exit 7
}

LDD_OUTPUT=$(ldd "$GENERATOR_OUTPUT")
printf '%s\n' "$LDD_OUTPUT"

if [[ "$LDD_OUTPUT" == *"not found"* ]]; then
  echo "ERROR: compiled generator has an unresolved runtime library" >&2
  exit 8
fi

echo "COMPILE_SUCCESS=$GENERATOR_OUTPUT"
