#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

DEFAULT_SKILLS=("gitcode-pr" "gitcode-pipeline")
REPO_URL="${DEFAULT_SKILLS_REPO_URL:-https://gitcode.com/cann-agent/skills.git}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
CLONE_TIMEOUT="${CLONE_TIMEOUT:-30}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SKILLS_DIR}/../.." && pwd)"
REMOTE_DIR="${SKILLS_DIR}/_remote"
GITIGNORE="${PROJECT_ROOT}/.gitignore"

mkdir -p "${REMOTE_DIR}"

if command -v mktemp >/dev/null 2>&1; then
  TEMP_DIR="$(mktemp -d)"
else
  TEMP_DIR="/tmp/torchtitan_npu_skills_install_$$"
  mkdir -p "${TEMP_DIR}"
fi
trap 'rm -rf "${TEMP_DIR}"' EXIT

check_connectivity() {
  if command -v curl >/dev/null 2>&1; then
    curl --connect-timeout "${CONNECT_TIMEOUT}" --max-time "${CONNECT_TIMEOUT}" \
      -fsS -o /dev/null "https://gitcode.com" >/dev/null 2>&1
    return $?
  fi

  git ls-remote "${REPO_URL}" HEAD >/dev/null 2>&1
}

append_gitignore_once() {
  local entry="$1"
  if [[ -f "${GITIGNORE}" ]] && ! grep -qxF "${entry}" "${GITIGNORE}"; then
    printf '%s\n' "${entry}" >> "${GITIGNORE}"
  fi
}

install_skill() {
  local skill="$1"
  local source_dir="${TEMP_DIR}/skills/skills/${skill}"
  local target_dir="${REMOTE_DIR}/${skill}"
  local link_path="${SKILLS_DIR}/${skill}"

  if [[ ! -d "${source_dir}" ]]; then
    echo "Warning: remote skill not found: ${skill}" >&2
    return
  fi

  if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
    echo "Error: ${link_path} exists and is not a symlink. Refusing to overwrite." >&2
    exit 1
  fi

  rm -rf "${target_dir}"
  cp -R "${source_dir}" "${target_dir}"
  ln -sfn "_remote/${skill}" "${link_path}"
  append_gitignore_once ".agents/skills/${skill}"
  echo "Installed skill: ${skill}"
}

if ! check_connectivity; then
  echo "Error: cannot access gitcode.com. Please check network connectivity." >&2
  exit 1
fi

echo "Cloning skills repository..."
if command -v timeout >/dev/null 2>&1; then
  timeout "${CLONE_TIMEOUT}" git clone --depth 1 "${REPO_URL}" "${TEMP_DIR}/skills"
else
  GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME="${CLONE_TIMEOUT}" \
    git clone --depth 1 "${REPO_URL}" "${TEMP_DIR}/skills"
fi

if [[ ! -d "${TEMP_DIR}/skills/skills" ]]; then
  echo "Error: skills directory not found in ${REPO_URL}" >&2
  exit 1
fi

append_gitignore_once ".agents/skills/_remote/"

for skill in "${DEFAULT_SKILLS[@]}"; do
  install_skill "${skill}"
done

echo "Default GitCode skills installed successfully."
