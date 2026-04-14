#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: zdev-container.sh <command> [args...]

Commands
  shell [shell-args]    Launch or resume an interactive zdev shell.
  claude [args...]      Launch or resume the Claude Code CLI in-container.
  codex  [args...]      Launch or resume the OpenAI CLI in-container.
  start <name>          Resume a stopped container by suffix (shell|claude|codex).
  stop <name>           Stop a running container by suffix.
  rm <name>             Remove a stopped container by suffix.

Environment
  ZDEV_IMAGE            Image tag to use (default ghcr.io/wherewithai/zdev:cuda-12.4).
  ZDEV_CONTAINER_PREFIX Container name prefix (default zdev-<repo-name>).
  ZDEV_GPU              When set to "1", adds --gpus all.
  ZDEV_WORKSPACE        Workspace path to mount (default current directory).
HELP
}

ensure_docker() {
    # Guard early if Docker CLI is unavailable.
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is required to run this helper" >&2
        exit 1
    fi
}

repo_name() {
    # Use the workspace (or cwd) to derive a stable repo-specific suffix.
    basename "${ZDEV_WORKSPACE:-$PWD}"
}

container_name() {
    local suffix="$1"
    local prefix="${ZDEV_CONTAINER_PREFIX:-zdev-$(repo_name)}"
    # Compose the container name, e.g. zdev-myrepo-shell.
    printf '%s-%s' "${prefix}" "${suffix}"
}

volume_args() {
    local repo
    repo="$(repo_name)"
    local workspace="${ZDEV_WORKSPACE:-$PWD}"
    local host_hf="${ZDEV_HF_CACHE:-${HOME}/.cache/huggingface}"

    local volumes=(
        # Workspace mount and shared caches keep container start lightweight.
        "-v" "${workspace}:/workspaces/${repo}"
        "-v" "pixi_pkgs_cache:/caches/pixi"
        "-v" "pip_cache:/caches/pip"
        "-v" "uv_cache:/caches/uv"
        "-v" "jupyter_cache:/caches/jupyter_cache"
        "-v" "zdev_egress_logs:/var/log/zdev/egress"
    )

    if [ -d "${host_hf}" ]; then
        volumes+=("-v" "${host_hf}:/caches/huggingface")
    else
        echo "warning: huggingface cache directory '${host_hf}' not found; skipping mount" >&2
    fi

    printf '%s\n' "${volumes[@]}"
}

docker_run_or_start() {
    local suffix="$1"
    shift
    local container
    container="$(container_name "${suffix}")"
    local image="${ZDEV_IMAGE:-ghcr.io/wherewithai/zdev:cuda-12.4}"
    local gpu_args=()
    if [ "${ZDEV_GPU:-0}" = "1" ]; then
        gpu_args=(--gpus all)
    fi

    if docker ps -a --format '{{.Names}}' | grep -Fx "${container}" >/dev/null 2>&1; then
        docker start -ai "${container}"
    else
        # shellcheck disable=SC2046
        # Launch a new container and reuse named volumes so caches persist.
        docker run -it \
            --name "${container}" \
            "${gpu_args[@]}" \
            $(volume_args) \
            "${image}" "$@"
    fi
}

docker_start() {
    local suffix="$1"
    local container
    container="$(container_name "${suffix}")"
    docker start -ai "${container}"
}

docker_stop() {
    local suffix="$1"
    local container
    container="$(container_name "${suffix}")"
    docker stop "${container}"
}

docker_rm() {
    local suffix="$1"
    local container
    container="$(container_name "${suffix}")"
    docker rm "${container}"
}

main() {
    if [ $# -eq 0 ]; then
        usage
        exit 2
    fi

    ensure_docker

    local cmd="$1"
    shift

    case "${cmd}" in
        shell)
            docker_run_or_start "shell" zdev shell "$@"
            ;;
        claude)
            docker_run_or_start "claude" claude "$@"
            ;;
        codex)
            docker_run_or_start "codex" openai "$@"
            ;;
        start)
            [ $# -ge 1 ] || { echo "start requires a suffix (shell|claude|codex)" >&2; exit 2; }
            docker_start "$1"
            ;;
        stop)
            [ $# -ge 1 ] || { echo "stop requires a suffix (shell|claude|codex)" >&2; exit 2; }
            docker_stop "$1"
            ;;
        rm)
            [ $# -ge 1 ] || { echo "rm requires a suffix (shell|claude|codex)" >&2; exit 2; }
            docker_rm "$1"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo "Unknown command: ${cmd}" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
