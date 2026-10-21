#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COMMON_DIR=$(dirname "$SCRIPT_DIR")
SOURCE_ROOT=$(dirname "$COMMON_DIR")
set -a
source "$COMMON_DIR/config.env"
set +a
COLLECTION_PROFILE=${COLLECTION_PROFILE:-full}
# ⚠️ 2026-09-29: 用户将手动微调事件号多次采集端到端 PMU。
#   强制只做端到端(execute_model_to_sample_tokens) PMU + time, 不采 hotspot/perf、不采 frequency,
#   与传入的 COLLECTION_PROFILE / config.env 的 FREQUENCY_ENABLE 无关(硬性短路)。
COLLECTION_PROFILE=full
export COLLECTION_PROFILE
FREQUENCY_ENABLE=0
export FREQUENCY_ENABLE
RUN_ROOT=${RUN_ROOT:-$SOURCE_ROOT/results/920b}
"$PYTHON_BIN" -c 'import openpyxl' >/dev/null 2>&1 || {
    printf 'Missing openpyxl; install scripts/requirements-report.txt\n' >&2
    exit 6
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'Exists: %s\n' "$RUN_ROOT" >&2; exit 2; }
install -d -m 755 "$RUN_ROOT"
: > "$RUN_ROOT/commands.txt"

record_command() {
    local title=$1
    local stdout_path=$2
    local merge_stderr=$3
    local background=$4
    shift 4
    {
        printf '[%s]\n' "$title"
        printf '%q ' "$@"
        if [[ -n "$stdout_path" ]]; then
            printf '> %q' "$stdout_path"
            [[ "$merge_stderr" == 1 ]] && printf ' 2>&1'
        fi
        [[ "$background" == 1 ]] && printf ' &'
        printf '\n\n'
    } >> "$RUN_ROOT/commands.txt"
}

if [[ -z "$VLLM_SITE" ]]; then
    VLLM_SITE=$("$PYTHON_BIN" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
fi
RUNTIME=$(mktemp -d /tmp/vllm.XXXXXX)
source "$COMMON_DIR/session.sh"
cp -rs "$VLLM_SITE" "$RUNTIME/vllm"
# ⚠️ topdown 适配(2026-09-22): runtime 以容器 0.23 站点为基底, 只覆盖端到端插桩文件。
#   端到端 topdown 插桩只改 gpu_worker.py(execute_model->sample_tokens) + kperf_instrument.py,
#   因此这里仅覆盖这两个文件, 其余保持容器 0.23 原版。
# ⚠️ 2026-10-19: 提交 7cb464aa2 将整棵 vLLM 树拷入 vllm/(包体移到 vllm/vllm/),
#   原源路径 vllm/v1/worker/gpu_worker.py 自此不存在, 覆盖被 [[ -e ]] 静默跳过。
#   两种布局都尝试, 覆盖目标(runtime 内的包路径)固定不变。
OVERRIDE_TARGET="vllm/v1/worker/gpu_worker.py"
OVERRIDE_SOURCES=(
    "vllm/v1/worker/gpu_worker.py"
    "vllm/vllm/v1/worker/gpu_worker.py"
)
for rel in "${OVERRIDE_SOURCES[@]}"; do
    file="$SOURCE_ROOT/$rel"
    if [[ -e "$file" ]]; then
        target="$RUNTIME/$OVERRIDE_TARGET"
        mkdir -p "$(dirname "$target")"
        ln -sfn "$file" "$target"
        break
    fi
done
ln -s "$SOURCE_ROOT/kperf_instrument.py" "$RUNTIME/kperf_instrument.py"
export SOURCE_ROOT VLLM_PYTHONPATH=$RUNTIME

# ⚠️ 对齐 vllmBench_lite(2026-10-19): runner 环境变量留空不传(空串会被 vLLM 布尔解析拒绝)。
RUNTIME_ENV=()
if [[ -n "${VLLM_USE_V2_MODEL_RUNNER:-}" ]]; then
    RUNTIME_ENV+=(VLLM_USE_V2_MODEL_RUNNER="$VLLM_USE_V2_MODEL_RUNNER")
fi
if [[ -n "${VLLM_USE_V1:-}" ]]; then RUNTIME_ENV+=(VLLM_USE_V1="$VLLM_USE_V1"); fi
RUNTIME_COMMAND=(
    env
    PYTHONPATH="$VLLM_PYTHONPATH"
    "${RUNTIME_ENV[@]}"
    "$PYTHON_BIN" -c
    'import os, sys, torch, vllm; from vllm.v1.worker.gpu import model_runner; from vllm.model_executor.models import qwen3; print(sys.version); print(vllm.__version__); print(torch.__version__); print(os.path.realpath(vllm.__file__)); print(os.path.realpath(model_runner.__file__)); print(os.path.realpath(qwen3.__file__))'
)
record_command "runtime identity" "$RUN_ROOT/runtime.txt" 0 0 \
    "${RUNTIME_COMMAND[@]}"
"${RUNTIME_COMMAND[@]}" > "$RUN_ROOT/runtime.txt"

# ⚠️ 2026-09-29: hotspot/perf 档已禁用(只做端到端 PMU)。此段已移除,
#   即使 COLLECTION_PROFILE 曾被设为 hotspot 也不会触发热点采 perf。

# ⚠️ 对齐 vllmBench 批量负载(2026-10-19): decode 步数取决于并发/调度/预热, 无法预知;
#   默认 0 = parse_run 跳过精确数量校验(仅保留 raw/selected/invalid 的基本检查)。
EXPECTED_CALLS=${EXPECTED_CALLS:-0}
start_session
END_TO_END_STAGE=execute_model_to_sample_tokens
END_TO_END_ROOT="$RUN_ROOT/end_to_end"

# =====================================================================
# [阶段 1] 函数阶段 — 已注释停用
#   原因: 函数阶段依赖 kperf 插桩的 8 个函数(add_requests/prepare_inputs/...),
#         而当前容器 vLLM 0.23 缺 prepare_attn_runner/async_output_init 等，
#         插桩在 0.23 上不可用。故只采端到端(topdown) + hotspot。
#   如需恢复: 取消下面整段注释 + 保留 COLLECTION_PROFILE=full。
# =====================================================================
# if [[ "$COLLECTION_PROFILE" == full ]]; then
# RUN_ROOT="$RUN_ROOT" "$COMMON_DIR/run_one.sh" time
# PARSE_COMMAND=(
#     "$PYTHON_BIN" "$COMMON_DIR/parse_run.py" "$RUN_ROOT/time"
#     --mode time
#     --expected-calls "$EXPECTED_CALLS"
# )
# record_command "time parse" "" 0 0 "${PARSE_COMMAND[@]}"
# "${PARSE_COMMAND[@]}"
#
# while IFS='|' read -r label codes; do
#     RUN_ROOT="$RUN_ROOT" "$COMMON_DIR/run_one.sh" "$label" "$codes" "$codes"
#     PARSE_COMMAND=(
#         "$PYTHON_BIN" "$COMMON_DIR/parse_run.py" "$RUN_ROOT/$label"
#         --event-names "$codes"
#         --expected-calls "$EXPECTED_CALLS"
#     )
#     record_command "$label parse" "" 0 0 "${PARSE_COMMAND[@]}"
#     "${PARSE_COMMAND[@]}"
# done <<EOF
# topdown|$EVENTS_920B_TOPDOWN
# flush|$EVENTS_920B_FLUSH
# badspec_branch|$EVENTS_920B_BADSPEC_BRANCH
# backend_core|$EVENTS_920B_BACKEND_CORE
# backend_memory|$EVENTS_920B_BACKEND_MEMORY
# icache|$EVENTS_920B_ICACHE
# dcache|$EVENTS_920B_DCACHE
# l3|$EVENTS_920B_L3
# tlb1|$EVENTS_920B_TLB1
# tlb2|$EVENTS_920B_TLB2
# branch|$EVENTS_920B_BRANCH
# imix|$EVENTS_920B_IMIX
# imix2|$EVENTS_920B_IMIX2
# EOF
#
# fi
printf '%s\n' "$COLLECTION_PROFILE" > "$RUN_ROOT/collection_profile"

install -d -m 755 "$END_TO_END_ROOT"
: > "$END_TO_END_ROOT/commands.txt"
KPERF_TARGET="$END_TO_END_STAGE" KPERF_QUALIFIER=run_fullgraph \
    RUN_ROOT="$END_TO_END_ROOT" "$COMMON_DIR/run_one.sh" time
PARSE_COMMAND=(
    "$PYTHON_BIN" "$COMMON_DIR/parse_run.py" "$END_TO_END_ROOT/time"
    --mode time
    --profile end_to_end
    --expected-calls "$EXPECTED_CALLS"
)
record_command "end-to-end time parse" "" 0 0 "${PARSE_COMMAND[@]}"
"${PARSE_COMMAND[@]}"

while IFS='|' read -r label codes; do
    KPERF_TARGET="$END_TO_END_STAGE" KPERF_QUALIFIER=run_fullgraph \
        RUN_ROOT="$END_TO_END_ROOT" \
        "$COMMON_DIR/run_one.sh" "$label" "$codes" "$codes"
    PARSE_COMMAND=(
        "$PYTHON_BIN" "$COMMON_DIR/parse_run.py" "$END_TO_END_ROOT/$label"
        --event-names "$codes"
        --profile end_to_end
        --expected-calls "$EXPECTED_CALLS"
    )
    record_command "end-to-end $label parse" "" 0 0 "${PARSE_COMMAND[@]}"
    "${PARSE_COMMAND[@]}"



# done <<EOF
# topdown|$EVENTS_920B_TOPDOWN
# flush|$EVENTS_920B_FLUSH
# badspec_branch|$EVENTS_920B_BADSPEC_BRANCH
# backend_core|$EVENTS_920B_BACKEND_CORE
# backend_memory|$EVENTS_920B_BACKEND_MEMORY
# icache|$EVENTS_920B_ICACHE
# dcache|$EVENTS_920B_DCACHE
# l3|$EVENTS_920B_L3
# tlb1|$EVENTS_920B_TLB1
# tlb2|$EVENTS_920B_TLB2
# branch|$EVENTS_920B_BRANCH
# imix|$EVENTS_920B_IMIX
# imix2|$EVENTS_920B_IMIX2
# EOF

# done <<EOF
# bp_not_pred_br|$EVENTS_BP_NOT_PRED_BR
# EOF


# done <<EOF
# br_spec|$BR_SPEC
# EOF

# done <<EOF
# br_mis_pred1|$BR_MIS_PRED1
# br_mis_pred2|$BR_MIS_PRED2
# EOF


# done <<EOF
# if_bp_pred_br1|$IF_BP_PRED_BR1
# if_bp_pred_br2|$IF_BP_PRED_BR2
# if_bp_pred_br3|$IF_BP_PRED_BR3
# EOF

# done <<EOF
# br_imm|$EVENTS_BR_IMM
# br_ret|$EVENTS_BR_RET
# br_indnr|$EVENTS_BR_INDNR
# btb_hit|$EVENTS_BTB_HIT
# btb_ubtb|$EVENTS_BTB_UBTB
# btb_fbtb|$EVENTS_BTB_FBTB
# btb_main|$EVENTS_BTB_MAIN
# EOF

# 预测源结构细分 → 替换/冲突/多命中 (2026-10-09, 活跃)
# done <<EOF
# if_bp_pred_br_ubtb|$IF_BP_PRED_BR_UBTB
# if_bp_pred_br_fbtb|$IF_BP_PRED_BR_FBTB
# if_bp_pred_br_btb|$IF_BP_PRED_BR_BTB
# EOF


# done <<EOF
# br_spec|$BR_SPEC
# br_mis_pred1|$BR_MIS_PRED1
# br_mis_pred2|$BR_MIS_PRED2
# if_bp_pred_br1|$IF_BP_PRED_BR1
# if_bp_pred_br2|$IF_BP_PRED_BR2
# if_bp_pred_br3|$IF_BP_PRED_BR3
# if_bp_pred_br_ubtb|$IF_BP_PRED_BR_UBTB
# if_bp_pred_br_fbtb|$IF_BP_PRED_BR_FBTB
# if_bp_pred_br_btb|$IF_BP_PRED_BR_BTB
# EOF

# ── 2026-10-15: 定向 RET/IND 完整失败率采集 (先前活跃, 本次注释, 改采 UNPRED_TKN_SUB) ──
# done <<EOF
# br_misp_ret|$BR_MISP_RET
# br_misp_ind|$BR_MISP_IND
# EOF

# ── 2026-10-15: 未预测-taken 分支(0x1009) 中 RET/IND 各自占比 ──
done <<EOF
unpred_tkn_sub|$UNPRED_TKN_SUB
EOF


# if [[ "$COLLECTION_PROFILE" == full ]]; then
#     RUN_ROOT="$RUN_ROOT" "$COMMON_DIR/run_one.sh" hotspot
# fi
# if [[ ${FREQUENCY_ENABLE:-0} == 1 ]]; then
#     RUN_ROOT="$RUN_ROOT" "$COMMON_DIR/run_one.sh" frequency
# fi
stop_session
SUMMARY_COMMAND=("$PYTHON_BIN" "$SCRIPT_DIR/summary.py" "$RUN_ROOT")
record_command "summary" "" 0 0 "${SUMMARY_COMMAND[@]}"
"${SUMMARY_COMMAND[@]}"

if [[ ${SKIP_XLSX:-0} == 1 ]]; then
    printf 'SKIP_XLSX=1: 跳过 Excel 报告生成 (summary.csv 已输出)\n'
    exit 0
fi

BUILD_COMMAND=(
    "$PYTHON_BIN" "$COMMON_DIR/build_xlsx.py" "$RUN_ROOT"
    --config "$SCRIPT_DIR/report_config.json"
    --chip "$(basename "$SCRIPT_DIR")"
    --version "$VLLM_VERSION_SHORT"
    --model-short "$MODEL_SHORT"
    --input-len "$RANDOM_INPUT_LEN"
    --output-len "$RANDOM_OUTPUT_LEN"
    --compact-report
    --include-end-to-end
)
record_command "Excel report" "" 0 0 "${BUILD_COMMAND[@]}"
"${BUILD_COMMAND[@]}"
