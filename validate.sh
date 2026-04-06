#!/usr/bin/env bash
set -uo pipefail

# Configuration
DOCKER_BUILD_TIMEOUT=600
PING_URL="${1:-http://localhost:7860}"
REPO_DIR="${2:-.}"

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

portable_mktemp() {
  local prefix="$1"
  if command -v mktemp &>/dev/null; then
    mktemp -t "$prefix.XXXXXX"
  else
    local hash; hash=$(date +%s%N | shasum | head -c 6)
    echo "/tmp/$prefix.$hash"
  fi
}

run_with_timeout() {
  local timeout="$1"
  shift
  if command -v timeout &>/dev/null; then
    timeout "$timeout" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$timeout" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$timeout" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid"
    local status=$?
    kill "$watcher" 2>/dev/null
    return $status
  fi
}

CLEANUP_FILES=()
cleanup() {
  for f in "${CLEANUP_FILES[@]}"; do
    [ -f "$f" ] && rm -f "$f"
  done
}
trap cleanup EXIT

printf "\n${BOLD}=== OpenEnv Pre-Submission Validator ===${NC}\n\n"

# Step 1: Ping HF Space
log "${BOLD}Step 1/3: Pinging Environment${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
