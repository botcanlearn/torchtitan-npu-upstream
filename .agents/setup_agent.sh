#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SKILLS_DIR="${PROJECT_ROOT}/.agents/skills"

if [[ ! -d "${SKILLS_DIR}" ]]; then
  echo "未找到 skills 目录: ${SKILLS_DIR}" >&2
  echo "请在项目根目录执行该脚本，或确认 .agents/skills 存在。" >&2
  exit 1
fi

parse_frontmatter_field() {
  local file="$1"
  local key="$2"

  awk -v key="${key}" '
    BEGIN { in_frontmatter=0; separator_count=0; capture_block=0 }
    /^---[[:space:]]*$/ {
      separator_count++
      if (separator_count == 1) {
        in_frontmatter=1
        next
      }
      if (separator_count == 2) {
        exit
      }
    }
    capture_block {
      if ($0 ~ /^[[:space:]]+/) {
        sub(/^[[:space:]]+/, "", $0)
        if ($0 != "") {
          print $0
          exit
        }
        next
      }
      exit
    }
    in_frontmatter {
      if ($0 ~ ("^" key ":[[:space:]]*")) {
        sub("^" key ":[[:space:]]*", "", $0)
        if ($0 ~ /^[>|][[:space:]]*$/) {
          capture_block=1
          next
        }
        print $0
        exit
      }
    }
  ' "${file}"
}

trim_wrapped_quotes() {
  local value="$1"
  if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
    value="${value:1:${#value}-2}"
  fi
  echo "${value}"
}

show_available_skills() {
  # 在 quiet 模式下不打印 skill 列表，避免每次启动都刷屏
  if [[ "${QUIET:-0}" -eq 1 ]]; then
    return
  fi

  local skill_files=()
  local file=""

  shopt -s nullglob
  skill_files=("${SKILLS_DIR}"/*/SKILL.md)
  shopt -u nullglob

  echo
  echo "可用 skills:"

  if [[ ${#skill_files[@]} -eq 0 ]]; then
    echo "- (未发现可用 skill)"
    return
  fi

  for file in "${skill_files[@]}"; do
    local skill_name
    local skill_desc

    skill_name="$(parse_frontmatter_field "${file}" "name")"
    skill_desc="$(parse_frontmatter_field "${file}" "description")"

    skill_name="$(trim_wrapped_quotes "${skill_name}")"
    skill_desc="$(trim_wrapped_quotes "${skill_desc}")"

    if [[ -z "${skill_name}" ]]; then
      skill_name="$(basename "$(dirname "${file}")")"
    fi
    if [[ -z "${skill_desc}" ]]; then
      skill_desc="(无简介)"
    fi

    echo "- ${skill_name}: ${skill_desc}"
  done
}

apply_agent_choice() {
  local agent="$1"
  case "${agent}" in
    claude|claude-code|.claude)
      AGENT_LABEL="claude code"
      AGENT_DIR_NAME=".claude"
      ;;
    opencode|.opencode)
      AGENT_LABEL="opencode"
      AGENT_DIR_NAME=".opencode"
      ;;
    codex|.codex)
      AGENT_LABEL="codex"
      AGENT_DIR_NAME=".codex"
      ;;
    *)
      echo "未知 agent: ${agent}（仅支持 claude / opencode / codex）" >&2
      return 1
      ;;
  esac
}

usage() {
  cat <<EOF
用法: $(basename "$0") --agent claude|opencode|codex [--quiet]
  --agent  必填，指定要初始化的 agent 目录
  --quiet  仅打印异常信息，适合在 hook/plugin 里调用

注意：本脚本已交由 .claude/settings.json (SessionStart hook)、
.opencode/plugins/bootstrap.ts 和 .codex/hooks.json 自动调用，无需手动执行。
EOF
}

parse_args() {
  AGENT_DIR_NAME=""
  AGENT_LABEL=""
  QUIET=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --agent)
        [[ $# -ge 2 ]] || { echo "--agent 缺少参数" >&2; exit 1; }
        apply_agent_choice "$2" || exit 1
        shift 2
        ;;
      --agent=*)
        apply_agent_choice "${1#--agent=}" || exit 1
        shift
        ;;
      --quiet|-q)
        QUIET=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "未知参数: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [[ -z "${AGENT_DIR_NAME}" ]]; then
    echo "缺少必填参数 --agent" >&2
    usage >&2
    exit 1
  fi
}

log() {
  [[ "${QUIET:-0}" -eq 1 ]] || echo "$@"
}

ensure_symlink() {
  local link_path="$1"
  local relative_target="$2"

  if [[ -L "${link_path}" ]]; then
    local current_target
    current_target="$(readlink "${link_path}")"
    if [[ "${current_target}" == "${relative_target}" ]]; then
      log "已存在软链接: ${link_path} -> ${current_target}"
      return
    fi
  fi

  if [[ -e "${link_path}" || -L "${link_path}" ]]; then
    local backup_path="${link_path}.bak.$(date +%Y%m%d%H%M%S)"
    mv "${link_path}" "${backup_path}"
    log "检测到已有 ${link_path}，已备份到 ${backup_path}"
  fi

  ln -s "${relative_target}" "${link_path}"
  log "已创建软链接: ${link_path} -> ${relative_target}"
}

setup_agent_links() {
  # claude code 与 opencode 各自需要的入口名不同：
  #   - claude code: .claude/CLAUDE.md   <- ../.agents/AGENTS.md
  #   - opencode  : .opencode/AGENTS.md  <- ../.agents/AGENTS.md
  #   - codex     : AGENTS.md             <- .agents/AGENTS.md
  # 其余 skills / rules 命名一致，codex 侧保留 .codex/skills 软链以兼容旧版扫描路径。
  local agent_dir="${PROJECT_ROOT}/${AGENT_DIR_NAME}"
  mkdir -p "${agent_dir}"

  local context_filename
  local context_link_path
  local context_target
  case "${AGENT_DIR_NAME}" in
    .claude)
      context_filename="CLAUDE.md"
      context_link_path="${agent_dir}/${context_filename}"
      context_target="../.agents/AGENTS.md"
      ;;
    .opencode)
      context_filename="AGENTS.md"
      context_link_path="${agent_dir}/${context_filename}"
      context_target="../.agents/AGENTS.md"
      ;;
    .codex)
      context_filename="AGENTS.md"
      context_link_path="${PROJECT_ROOT}/${context_filename}"
      context_target=".agents/AGENTS.md"
      ;;
    *)
      echo "未知 AGENT_DIR_NAME: ${AGENT_DIR_NAME}" >&2
      exit 1
      ;;
  esac

  ensure_symlink "${agent_dir}/skills"             "../.agents/skills"
  ensure_symlink "${agent_dir}/rules"              "../.agents/rules"
  ensure_symlink "${context_link_path}"            "${context_target}"
}

main() {
  parse_args "$@"

  log "项目根目录: ${PROJECT_ROOT}"
  log "skills 目录: ${SKILLS_DIR}"
  log ""

  setup_agent_links
  show_available_skills

  log ""
  log "初始化完成。你现在可以在 ${AGENT_LABEL} 中使用这些 skills。"
}

_LOG="${PROJECT_ROOT}/.agents/.setup_agent.log"
if ! _err=$(main "$@" 2>&1); then
  echo "${_err}" > "${_LOG}"
  echo "[setup_agent] 初始化失败，详见 ${_LOG}" >&2
  exit 1
fi
if [[ -n "${_err}" ]]; then
  printf '%s\n' "${_err}"
fi
