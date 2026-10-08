#!/usr/bin/env bash
# Dispatcher host -> SSH execution host. No full training logs return to dispatcher.
set -euo pipefail
run_id="${1:?run_id}" attempt="${2:?attempt}" sha="${3:?sha}" stamp="${4:?timestamp}"
[[ "$run_id" =~ ^[0-9]+$ && "$attempt" =~ ^[0-9]+$ && "$sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$stamp" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_run-[0-9]+_attempt-[0-9]+$ ]]
repo='depeng1994/torchtitan-npu'
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git -C "$work" init -q
git -C "$work" remote add origin "https://github.com/$repo.git"
git -C "$work" fetch -q --depth=1 origin "$sha"
actual=$(git -C "$work" rev-parse FETCH_HEAD)
[[ "$actual" == "$sha" ]]
git -C "$work" archive --format=tar "$sha" | gzip -1 > "$work/source.tar.gz"
test -s "$work/source.tar.gz"
dest="/mnt/share/ci_tests/${stamp}"
echo "Source SHA: $sha"
echo "Internal SSH package bytes: $(stat -c %s "$work/source.tar.gz")"
ssh -o BatchMode=yes -o ConnectTimeout=15 a3-4-docker-relay "mkdir '$dest'"
scp -q "$work/source.tar.gz" "a3-4-docker-relay:$dest/source.tar.gz"
# Remote command writes all rich outputs to the execution host. Only small summary
# and 20 final lines travel back through SSH and GitHub.
ssh -o BatchMode=yes -o ServerAliveInterval=30 a3-4-docker-relay   bash -s -- "$dest" "$sha" <<'REMOTE'
set -uo pipefail
dest="$1"
sha="$2"
cd "$dest" || exit 125
mkdir repo
tar -xzf source.tar.gz -C repo || exit 125
printf '%s\n' "$sha" > source_sha.txt
cd repo || exit 125
source /mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh >"$dest/env_setup.log" 2>&1
export ASCEND_SET_ENV_PATH=/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh
export HF_ASSETS_PATH=/mnt/share/models/DeepSeek-V4-Flash-bf16
export CKPT_INIT_LOAD_PATH=/mnt/share/dsv4_ckpt_8rank
export CKPT_SAVE_LOAD_PATH="$dest/checkpoints"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCHINDUCTOR_NPU_BACKEND=ascendc
export STEPS=5
# Fast functional Actions smoke: no Inductor; the standalone example still
# defaults to Inductor, selectable with COMPILE_ENABLE=1 outside this job.
export COMPILE_ENABLE=0
export PYTHONUNBUFFERED=1
echo "8P CI mode: eager (COMPILE_ENABLE=$COMPILE_ENABLE); STEPS=$STEPS"
echo "$(date -Iseconds)" > "$dest/start_time.txt"
timeout --signal=TERM --kill-after=30s 6900s python3 -u -m tests.integration_tests.run_multinode_tests "$dest/output" >"$dest/run.log" 2>&1
rc=$?
printf '%s\n' "$rc" >"$dest/exit_code.txt"
echo "$(date -Iseconds)" > "$dest/end_time.txt"
echo "RESULT_DIR: $dest"
echo "RESULT_SHA: $sha"
echo "RESULT_EXIT_CODE: $rc"
echo "RESULT_FULL_LOG_BYTES: $(stat -c %s "$dest/run.log")"
# Normalize on the execution host; a terminated Python runner may leave
# run.log empty, but rank-0 structured logs are still available.
python3 .github/scripts/a3-8p-log-tail.py "$dest" >"$dest/tail_20.log" 2>"$dest/tail_error.log" ||     printf 'Log-tail extraction failed; check execution-host tail_error.log\n' >"$dest/tail_20.log"
echo "RESULT_LAST_20_LINES_BEGIN"
cat "$dest/tail_20.log"
echo "RESULT_LAST_20_LINES_END"
exit "$rc"
REMOTE
