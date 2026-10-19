#!/usr/bin/env bash
# Adapted from the uploaded vllmBench_lite.sh. Keep one server for all cases.
# Defaults retained: warmups=16, cache OFF, CPUs 80-159,240-319, backend=vllm.
# Requires the matching files in litehost_compat/ (included in the complete ZIP).
set -uo pipefail

: "${MODEL_PATH:?ERROR: 必须设置 MODEL_PATH}"
TEST_ID="litehost_compat_$(date +%Y%m%d_%H%M%S)"
PREFIX="${PREFIX:-/opt/corex311}"
PYTHON_BIN="${PYTHON_BIN:-${PREFIX}/bin/python3}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GPUS_PER_INSTANCE="${GPUS_PER_INSTANCE:-4}"
PORT="${PORT:-8848}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_WARMUPS="${NUM_WARMUPS:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
CASE_INTERVAL="${CASE_INTERVAL:-60}"
CPU_RANGE="${CPU_RANGE-80-159,240-319}"
# ===== kunpeng-affinity-plugin 绑核开关(KUNPENG_AFFINITY_PLUGIN)=====
#   off (默认) -> 用脚本原生的静态 taskset cpuset(CPU_RANGE),不加载插件
#   on         -> 让插件按 GPU->NUMA 做 per-worker 绑核。此时会:
#                 * 禁用 taskset SERVER_PREFIX(两者互斥,避免双重绑核打架)
#                 * 注入 sitecustomize + PYTHONPATH 加载插件并 register()
#                 * SERVER_ARGS 加 --numa-bind(激活插件 should_resolve)
#                 * 设 KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1(走插件 generic 链,
#                   日志出 selected NUMA nodes source=generic,而非 native)
KUNPENG_AFFINITY_PLUGIN="${KUNPENG_AFFINITY_PLUGIN:-off}"
KUNPENG_AFFINITY_REPO="${KUNPENG_AFFINITY_REPO:-/home/qch/tools/kunpeng-affinity-plugin}"
KUNPENG_AFFINITY_IXSMI="${KUNPENG_AFFINITY_IXSMI:-/home/qch/kunpeng-affinity-fix/ixsmi-run}"
KUNPENG_AFFINITY_PROVIDER="${KUNPENG_AFFINITY_PROVIDER:-iluvatar-runtime-pci}"
HOST_MEASURE="${HOST_MEASURE:-lite}"         # lite | off (never enable Torch Profiler here)
HOST_LINK_PERCENT="${HOST_LINK_PERCENT:-0}" # 5 means 5%, computed offline
HOST_TIMEOUT="${HOST_TIMEOUT:-120}"
HOST_MAX_SCOPES="${HOST_MAX_SCOPES:-200000}"
HOST_DRY_RUN="${HOST_DRY_RUN:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Do not use an inherited OLD LIGHTHOST_HOME silently.
if [[ -f "$SCRIPT_DIR/lhctl.py" ]]; then
    HOST_TOOLKIT="${HOST_TOOLKIT:-$SCRIPT_DIR}"
else
    HOST_TOOLKIT="${HOST_TOOLKIT:-$SCRIPT_DIR/litehost_compat}"
fi
HOST_TOOLKIT="$(cd -- "$HOST_TOOLKIT" 2>/dev/null && pwd)" || { echo 'ERROR: 解压完整 litehost_compat.zip 或设置 HOST_TOOLKIT' >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$HOST_TOOLKIT/lhctl.py" ]] || { echo 'ERROR: Python 或配套文件缺失' >&2; exit 2; }
[[ "$HOST_MEASURE" == lite || "$HOST_MEASURE" == off ]] || { echo 'ERROR: HOST_MEASURE=lite|off' >&2; exit 2; }
[[ "$GPUS_PER_INSTANCE" == 4 ]] || { echo 'ERROR: 此版要求四卡 TP=4, PP=1, DP=1' >&2; exit 2; }
[[ "$TEST_ID" =~ ^[A-Za-z0-9_.-]+$ && "$TEST_ID" != . && "$TEST_ID" != .. ]] || { echo 'ERROR: TEST_ID 不能包含路径分隔符' >&2; exit 2; }

BASE_DIR="$(pwd)"
RUN_DIR="${BASE_DIR}/vllmBench/${TEST_ID}"
LOG_DIR="${RUN_DIR}/logs"
SERVER_LOG="${LOG_DIR}/server.log"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
SESSION_FILE="${RUN_DIR}/session.json"
PID_FILE="${RUN_DIR}/server.pid"
SERVER_PID=""; BENCH_PID=""; CLEANED=0; INIT_DONE=0

# Original case matrix: only 128/128 enabled; uncomment the others as before.
CASES=(
#   "128 128"
#   "128 1024"
#   "128 2048"
#   "1024 128"
#   "1024 1024"
#   "1024 2048"
    "2048 128"
#   "2048 1024"
#   "2048 2048"
)
# Optional text file, one 'input output' case per nonblank/non-comment line.
if [[ -n "${CASES_FILE:-}" ]]; then
    [[ -f "$CASES_FILE" ]] || { echo "ERROR: 找不到 CASES_FILE=$CASES_FILE" >&2; exit 2; }
    mapfile -t CASES < <(grep -Ev '^[[:space:]]*(#|$)' "$CASES_FILE")
fi
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
# Child processes only: drop old measurement/audit injection; keep CoreX library paths.
CLEAN_ENV=(env -u LD_PRELOAD -u LD_AUDIT -u LD_DEBUG -u LD_DEBUG_OUTPUT -u LD_BIND_NOT
    -u LD_PROFILE -u VLLM_TORCH_PROFILER_DIR -u PROFILER_CONFIG -u TORCH_PROFILER_DIR)
SERVER_PREFIX=()

# ===== 绑核方式选择: 插件 vs 原生 taskset =====
# 插件 on: 禁用 taskset(两者互斥),由插件按 GPU->NUMA per-worker 绑核;
#          插件需要 --numa-bind 才激活,见 kunpeng_affinity/adapters/vllm_lifecycle.py should_resolve
# 插件 off: 沿用原生静态 taskset -c $CPU_RANGE(旧行为)
PLUGIN_BOOTSTRAP="${PLUGIN_BOOTSTRAP:-}"
if [[ "$KUNPENG_AFFINITY_PLUGIN" == on ]]; then
    # 校验插件仓库存在 + 目标 python 能 import
    [[ -d "$KUNPENG_AFFINITY_REPO/src" ]] || { echo "ERROR: 插件仓库不存在 $KUNPENG_AFFINITY_REPO/src" >&2; exit 2; }
    # 注意: PLUGIN_BOOTSTRAP 只算路径,不在这里 mkdir。
    # 因为 mkdir -p $RUN_DIR/bootstrap 会先把 RUN_DIR 建出来,而脚本稍后
    # 会检查 [[ -e $RUN_DIR ]] 报“已存在”,自己拦自己。bootstrap 的创建
    # 统一放到 start_server()(RUN_DIR 校验通过后)再执行。
    PLUGIN_BOOTSTRAP="${RUN_DIR}/bootstrap"
    # 插件 on 时去掉 taskset 静态绑核,避免与插件 numactl 绑核冲突
    SERVER_PREFIX=()
    log "KUNPENG_AFFINITY_PLUGIN=on: 禁用 taskset,改用插件 generic 绑核"
# else
#     [[ -n "$CPU_RANGE" ]] && SERVER_PREFIX=(taskset -c "$CPU_RANGE")
fi

SERVER_ARGS=(
    -m vllm.entrypoints.openai.api_server
    --host 127.0.0.1 --port "$PORT" --model "$MODEL_PATH" --trust_remote_code --dtype auto
    --max_model_len 4096 --distributed_executor_backend mp --pipeline_parallel_size 1
    --tensor_parallel_size "$GPUS_PER_INSTANCE" --data_parallel_size 1
    --quantization compressed-tensors --gpu_memory_utilization 0.9 --kv_cache_dtype auto
    --enable_chunked_prefill --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "mode":0}'
    --async-scheduling --no-enable-prefix-caching
)
[[ "$HOST_MEASURE" == lite ]] && SERVER_ARGS+=(--worker-cls light_host_worker.LightHostWorker)
# 插件 on: 加 --numa-bind(激活插件 should_resolve)+ 强制 generic 链
if [[ "$KUNPENG_AFFINITY_PLUGIN" == on ]]; then
    SERVER_ARGS+=(--numa-bind)
fi

cleanup() {
    if [[ "$CLEANED" == 1 ]]; then return; fi
    CLEANED=1
    if [[ -n "$BENCH_PID" ]]; then
        "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/procutil.py" stop "$RUN_DIR/bench.process.json" || true
        wait "$BENCH_PID" 2>/dev/null || true
        BENCH_PID=""
    fi
    if [[ -n "$SERVER_PID" ]]; then
        log "停止本次服务 PID=$SERVER_PID（不使用全局 pkill）"
        "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/procutil.py" stop "$RUN_DIR/server.process.json" || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""; rm -f -- "$PID_FILE"
    fi
    if [[ "$INIT_DONE" == 1 ]]; then
        "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/lhctl.py" cleanup-control --session "$SESSION_FILE" || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_gpu_memory() {
    local gpu free_mib
    local -a gpus=()
    IFS=',' read -r -a gpus <<< "$GPU_LIST"
    if ! command -v ixsmi >/dev/null; then log 'WARNING: ixsmi 不可用，跳过显存预检'; return; fi
    for gpu in "${gpus[@]}"; do
        free_mib=$(ixsmi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]]; then
            log "GPU $gpu: free=${free_mib} MiB（原预检阈值 20480 MiB）"
            (( free_mib >= 20480 )) || log "WARNING: GPU $gpu 显存偏低，请确认测试卡已预留"
        else log "WARNING: 无法读取 GPU $gpu 显存"; fi
    done
    return 0
}
wait_for_server() {
    local elapsed=0 code
    while (( elapsed < 600 )); do
        kill -0 "$SERVER_PID" 2>/dev/null || { log "ERROR: 服务已退出，查看 $SERVER_LOG"; return 1; }
        code=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
        [[ "$code" == 200 ]] && { log "服务就绪 (${elapsed}s)"; return 0; }
        sleep 5; elapsed=$((elapsed+5))
    done
    log 'ERROR: 服务启动超时'; return 1
}
start_server() {
    # Refuse an occupied port. Never kill an unrelated service by name/port.
    "${CLEAN_ENV[@]}" "$PYTHON_BIN" - "$PORT" <<'PY' || return 1
import socket,sys
with socket.socket() as s:s.bind(('127.0.0.1',int(sys.argv[1])))
PY
    check_gpu_memory "$GPU_LIST"
    # 插件 on: 此时 RUN_DIR 已通过“已存在”检查且已 mkdir,才创建 bootstrap(避免自拦)
    if [[ "$KUNPENG_AFFINITY_PLUGIN" == on ]]; then
        mkdir -p "$PLUGIN_BOOTSTRAP"
        if [[ ! -f "$PLUGIN_BOOTSTRAP/sitecustomize.py" ]]; then
            cat > "$PLUGIN_BOOTSTRAP/sitecustomize.py" <<'PY'
from kunpeng_affinity.vllm_plugin import register
register()
PY
        fi
    fi
    local -a env_args=(CUDA_VISIBLE_DEVICES="$GPU_LIST" VLLM_ENFORCE_CUDA_GRAPH=1 VLLM_USE_V1=1)
    # 统一组装 PYTHONPATH,避免多次 PYTHONPATH= 互相覆盖(后者胜)。
    # 插件 sitecustomize 需在所有 sitecustomize 候选之前,确保 register() 最早生效;
    # 因此始终把插件 bootstrap+src 放最前列,再依次追加 HOST_TOOLKIT 等。
    local merge_py="${PYTHONPATH:-}"
    if [[ "$KUNPENG_AFFINITY_PLUGIN" == on ]]; then
        merge_py="${PLUGIN_BOOTSTRAP}:${KUNPENG_AFFINITY_REPO}/src${merge_py:+:$merge_py}"
        env_args+=(KUNPENG_AFFINITY_MODE=auto KUNPENG_AFFINITY_PROVIDER="$KUNPENG_AFFINITY_PROVIDER")
        env_args+=(KUNPENG_AFFINITY_IXSMI="$KUNPENG_AFFINITY_IXSMI" KUNPENG_AFFINITY_VLLM_FORCE_GENERIC=1)
    else
        # 插件 off: 仅追加 HOST_TOOLKIT(见下),无需额外处理
        :
    fi
    if [[ "$HOST_MEASURE" == lite ]]; then
        merge_py="${merge_py:+$merge_py:}${HOST_TOOLKIT}"
        env_args+=(LD_PRELOAD="$HOST_TOOLKIT/liblitehost.so"
            LIGHTHOST_LIB="$HOST_TOOLKIT/liblitehost.so" LIGHTHOST_SESSION="$SESSION_FILE")
    fi
    [[ -n "$merge_py" ]] && env_args+=(PYTHONPATH="$merge_py")
    "${CLEAN_ENV[@]}" "${env_args[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/procutil.py" exec "$RUN_DIR/server.process.json" -- \
        "${SERVER_PREFIX[@]}" "$PYTHON_BIN" "${SERVER_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!; echo "$SERVER_PID" > "$PID_FILE"
    log "服务 PID=$SERVER_PID；所有 case 复用；日志=$SERVER_LOG"
    wait_for_server
    if [[ "$KUNPENG_AFFINITY_PLUGIN" == on ]]; then
        verify_plugin_binding "$SERVER_LOG"
    fi
}

# 校验 kunpeng-affinity 插件绑核是否真正生效(仅告警,不中断 benchmark)
#   ✅ 有 'selected NUMA nodes' + 'source=generic' = 插件 generic 链绑核
#   ❌ 出现 'Auto-detected NUMA nodes' = 仍走 vLLM 原生绑核(插件 defer),非插件
#   两者都无 = 插件未动作(可能 numa_bind 未生效)
verify_plugin_binding() {
    local slog="$1"
    if [[ ! -f "$slog" ]]; then log "WARNING: 找不到 server.log,无法校验插件绑核"; return; fi
    if grep -q "selected NUMA nodes" "$slog" 2>/dev/null; then
        log "✅ 插件绑核证据: $(grep -o 'selected NUMA nodes=[^ ]*' "$slog" | head -1) $(grep -oE 'source=(generic|native)' "$slog" | head -1)"
    elif grep -q "Auto-detected NUMA nodes" "$slog" 2>/dev/null; then
        log "❌ 插件未绑核: 仍走 vLLM 原生 (Auto-detected $(grep -o 'Auto-detected NUMA nodes: \[[^]]*\]' "$slog" | head -1))。检查 FORCE_GENERIC/--numa-bind 是否注入"
    elif grep -q "installed vLLM subprocess hook" "$slog" 2>/dev/null; then
        log "❌ 插件 hook 已装但无 selected NUMA 决策(numu_bind 未生效?)——请查看 $slog 顶部 kunpeng-affinity 行"
    else
        log "❌ 未检测到插件 hook,检查 PYTHONPATH/sitecustomize 注入"
    fi
}
run_case() {
    local input_len="$1" output_len="$2"
    local case_id="bs${MAX_CONCURRENCY}_in${input_len}_out${output_len}"
    local case_log="${LOG_DIR}/bench_${case_id}.log" case_dir="${RUN_DIR}/litehost/${case_id}"
    local -a client env_args
    env_args=()
    if [[ "$HOST_MEASURE" == lite ]]; then
        "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/lhctl.py" prepare --session "$SESSION_FILE" \
            --case-id "$case_id" --num-prompts "$NUM_PROMPTS" --input-len "$input_len" --output-len "$output_len" || return 1
        client=("$HOST_TOOLKIT/bench_driver.py")
        env_args=(LIGHTHOST_SESSION="$SESSION_FILE" LIGHTHOST_CASE_DIR="$case_dir")
    else
        client=(-m vllm.entrypoints.cli.main bench serve)
    fi
    log "==== $case_id | mode=$HOST_MEASURE | warmup=$NUM_WARMUPS | 同一服务 ===="
    "${CLEAN_ENV[@]}" "${env_args[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/procutil.py" exec "$RUN_DIR/bench.process.json" -- \
        "$PYTHON_BIN" "${client[@]}" \
        --backend vllm \
        --num_prompts "$NUM_PROMPTS" --num-warmups "$NUM_WARMUPS" --max-concurrency "$MAX_CONCURRENCY" \
        --dataset_name random --random_input_len "$input_len" --random_output_len "$output_len" \
        --host 127.0.0.1 --port "$PORT" --model "$MODEL_PATH" --trust_remote_code \
        --ignore_eos --top_p 1 --top_k -1 --temperature 0 --seed 42 \
        --save-result --result-dir "$RUN_DIR" --result_filename "${case_id}.json" > "$case_log" 2>&1 &
    BENCH_PID=$!
    local rc=0
    wait "$BENCH_PID" || rc=$?
    BENCH_PID=""
    if (( rc != 0 )); then log "ERROR: case 失败 rc=$rc；查看 $case_log"; return "$rc"; fi
    if [[ "$HOST_MEASURE" == lite ]]; then
        "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/lhctl.py" finish "$case_dir" \
            --link-percent "$HOST_LINK_PERCENT" || return 1
    fi
    log "完成 $case_id → ${RUN_DIR}/${case_id}.json"
}

# Validate cheap inputs before model initialization.
"${CLEAN_ENV[@]}" "$PYTHON_BIN" - "$GPU_LIST" "$NUM_PROMPTS" "$NUM_WARMUPS" "$MAX_CONCURRENCY" "$HOST_LINK_PERCENT" "$CASE_INTERVAL" "${CASES[@]}" <<'PY' || exit 2
import math,sys
ids,n,w,c,p,interval,*cases=sys.argv[1:]
if len(ids.split(','))!=4 or len(set(ids.split(',')))!=4:raise ValueError('需要四张不同 GPU')
if int(n)<=0 or int(w)<0 or int(c)<=0:raise ValueError('请求/预热/并发参数无效')
if not math.isfinite(float(p)) or not 0<=float(p)<=100 or float(interval)<0:raise ValueError('系数或间隔无效')
if not cases or len(cases)!=len(set(cases)):raise ValueError('case 列表为空或重复')
for x in cases:
    parts=x.split()
    if len(parts)!=2 or min(map(int,parts))<=0 or sum(map(int,parts))>4096:raise ValueError('case 需要两个正整数且总长 <= 4096')
PY
log "TEST_ID=$TEST_ID | MODE=$HOST_MEASURE | WARMUP=$NUM_WARMUPS | CPU_RANGE=$CPU_RANGE"
log "KUNPENG_AFFINITY_PLUGIN=$KUNPENG_AFFINITY_PLUGIN | prefix cache=OFF | backend=vllm | 同一服务运行 ${#CASES[@]} 个 case | RUN_DIR=$RUN_DIR"
if [[ "$HOST_DRY_RUN" == 1 ]]; then
    printf 'SERVER: '; printf '%q ' "${SERVER_PREFIX[@]}" "$PYTHON_BIN" "${SERVER_ARGS[@]}"; printf '\n'
    printf 'CASES: '; printf '%s; ' "${CASES[@]}"; printf '\n'
    exit 0
fi
[[ -d "$MODEL_PATH" ]] || { log 'ERROR: MODEL_PATH 不是本地目录'; exit 2; }
if [[ -e "$RUN_DIR" ]]; then log 'ERROR: RUN_DIR 已存在，请换一个 TEST_ID，避免混入旧文件'; exit 2; fi
mkdir -p "$LOG_DIR"
if [[ "$HOST_MEASURE" == lite ]]; then
    "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/lhctl.py" init --run-dir "$RUN_DIR" \
        --timeout "$HOST_TIMEOUT" --max-scopes "$HOST_MAX_SCOPES" > "$LOG_DIR/preflight.log" 2>&1 || { cat "$LOG_DIR/preflight.log"; exit 2; }
    INIT_DONE=1
fi
if ! start_server; then log 'FATAL: 启动失败'; exit 1; fi
FAILED=0
for idx in "${!CASES[@]}"; do
    read -r input_len output_len <<< "${CASES[$idx]}"
    if ! run_case "$input_len" "$output_len"; then
        FAILED=1; log '停止后续 case，避免未完成请求或测量状态污染下一轮'; break
    fi
    if (( idx < ${#CASES[@]}-1 )); then log "等待 ${CASE_INTERVAL}s"; sleep "$CASE_INTERVAL"; fi
done
"${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/lhctl.py" summary "$RUN_DIR" || FAILED=1
if [[ "$HOST_MEASURE" == lite && "$FAILED" == 0 ]]; then
    "${CLEAN_ENV[@]}" "$PYTHON_BIN" "$HOST_TOOLKIT/analyze_litehost.py" "$RUN_DIR/litehost" \
        --link-percent "$HOST_LINK_PERCENT" --out "$RUN_DIR/host_summary" || FAILED=1
fi
cleanup
(( FAILED == 0 )) || { log "部分失败，保留诊断：$RUN_DIR"; exit 2; }
log "全部完成 | benchmark=$SUMMARY_CSV | host=$RUN_DIR/host_summary/report.txt"