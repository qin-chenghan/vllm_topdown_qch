#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger(__name__)
STAGES = (
    "add_requests",
    "prepare_inputs",
    "prepare_attn_runner",
    "prepare_attn_model_state",
    "run_fullgraph",
    "sample",
    "async_output_init",
    "postprocess_sampled",
)
END_TO_END_STAGE = "execute_model_to_sample_tokens"
REPORT_STAGES = (END_TO_END_STAGE, *STAGES)
GROUPS = (
    "time",
    "topdown",
    "flush",
    "badspec_branch",
    "backend_core",
    "backend_memory",
    "icache",
    "dcache",
    "l3",
    "tlb1",
    "tlb2",
    "branch",
    "imix",
    "imix2",
    "br_imm",
    "br_ret",
    "br_indnr",
    "btb_hit",
    "btb_ubtb",
    "btb_fbtb",
    "btb_main",
    "br_spec",
    "br_mis_pred1",
    "br_mis_pred2",
    "if_bp_pred_br1",
    "if_bp_pred_br2",
    "if_bp_pred_br3",
    "if_bp_pred_br_ubtb",
    "if_bp_pred_br_fbtb",
    "if_bp_pred_br_btb",
    "br_misp_ret",
    "br_misp_ind",
    "unpred_tkn_sub",
)
CYCLES_SHARE_METRIC = "cycle占比"
PERCENT_METRICS = {
    "CPU利用率",
    CYCLES_SHARE_METRIC,
    "Retire",
    "FrontendBound",
    "Fetch Latency Bound",
    "Fetch Bandwidth Bound",
    "Idle by iCache Miss",
    "Idle by iTLB Miss",
    "Flush",
    "Branch Flush",
    "OoO Flush",
    "SP Flush",
    "Branch Mispredicts",
    "Indirect Branch",
    "Push Branch",
    "Pop Branch",
    "Other Branch",
    "Machine Clears",
    "Nuke Flush",
    "Other Flush",
    "BadSpec",
    "BackendBound",
    "Core Bound",
    "Resource Bound",
    "FDIV Stall",
    "DIV Stall",
    "FSU Stall",
    "Exe Ports Util",
    "Memory Bound",
    "L1 Bound",
    "L2 Bound",
    "L3 Bound",
    "Mem Bound",
    "Store Bound",
    "dp_spec",
    "ld_spec",
    "st_spec",
    "branch_spec",
    "ase_spec",
    "br missrate",
    "l1i missrate",
    "l2i missrate",
    "l1d missrate",
    "L2d missrate",
    "L3 missrate",
    "itlb missrate",
    "dtlb missrate",
    "stlb missrate",
    "br imm%",
    "br imm mis%",
    "br ret%",
    "br ret mis%",
    "br indnr%",
    "br indnr mis%",
    "uBTB命中%",
    "fBTB命中%",
    "BTB命中%",
    "uBTB替换%",
    "uBTB冲突%",
    "fBTB替换%",
    "fBTB多命中%",
    "BTB替换%",
    "BTB多命中%",
    "BP_PRED_BR%",
    "BP_NOT_PRED_BR%",
    "BR_MIS_PRED%",
    "IF_BP_MISP_BR_DIR%",
    "IF_BP_MISP_BR_RET%",
    "IF_BP_MISP_BR_IND%",
    "IF_BP_MISP_BR_BLR%",
    "IF_BP_MISP_BR_BL%",
    "IF_BP_MISP_BR_TGT%",
    "IF_BP_MISP_BR_LP%",
    "IF_BP_PRED_BR_SP%",
    "IF_BP_PRED_BR_UBTB%",
    "IF_BP_PRED_BR_BTB%",
    "IF_BP_PRED_BR_COND%",
    "IF_BP_PRED_BR_RET%",
    "IF_BP_PRED_BR_IND%",
    "IF_BP_PRED_BR_BLR%",
    "IF_BP_PRED_BR_BL%",
    "IF_BP_PRED_BR_LP%",
    "IF_BP_PRED_BR_FBTB%",
    "IF_BP_PRED_BR_VBTB%",
    "IF_BP_UBTB_REPLACE%",
    "IF_BP_UBTB_CONFLICT%",
    "IF_BP_FBTB_REPLACE%",
    "IF_BP_FBTB_MULTI_HIT%",
    "IF_BP_BTB_REPLACE%",
    "IF_BP_BTB_MULTI_HIT%",
    "BR_MISP_RET/RET",
    "BR_MISP_RET/MISP",
    "BR_MISP_IND/IND",
    "BR_MISP_IND/MISP",
    "BR_UNPRED_RET_SUB%",
    "BR_UNPRED_IND_SUB%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def read_rows(
    root: Path, group: str, stage: str, worker: str | None = None
) -> list[dict[str, str]]:
    path = root / group / "parsed" / f"{stage}.csv"
    if not path.is_file():
        # ⚠️ 0.23 适配(2026-09-22): 8 阶段(非端到端)无采集数据, 文件缺失时返回空而非报错。
        return []
    if (root / "collection_profile").is_file() and (
        root / "collection_profile"
    ).read_text().strip() == "end_to_end":
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
            if row["valid"] == "1"
        ]
    # ⚠️ TP=4 适配(2026-10-19): 按 worker 列过滤; 旧数据无 worker 列时
    #   每行 worker 为空, 指定具体 worker 会得到空集(只有 all 列有数)。
    if worker is not None:
        rows = [row for row in rows if (row.get("worker") or "") == worker]
    # ⚠️ 2026-09-29: 端到端(execute_model_to_sample_tokens)模式下, 第一个样本
    #   (global_call=1) 是 prefill+首 token, cycles 约为其余稳态 decode 样本的 ~52 倍
    #   (实测 row1 4.15G vs 单个 decode ~78M), 占端到端全样本 cycles ~34%,
    #   会把 FrontendBound 从 ~25.6% 顶到 ~37.6%、BackendBound 从 ~42.6% 拉到 ~31.1%。
    #   这里剔除该 prefill 样本, 只统计稳态 decode, 与 203 及旧脚本口径一致,
    #   避免把 prefill 混进 decode 的 topdown/时间统计。pipeline(8阶段)模式无此问题。
    if stage == END_TO_END_STAGE:
        rows = [row for row in rows if row.get("global_call") != "1"]
    return rows


def mean(
    rows: list[dict[str, str]], value: Callable[[dict[str, str]], float]
) -> float | None:
    values = [value(row) for row in rows]
    return sum(values) / len(values) if values else None


def total(rows: list[dict[str, str]], fields: tuple[str, ...]) -> float:
    return sum(sum(float(row[field]) for field in fields) for row in rows)


def ratio(
    rows: list[dict[str, str]],
    numerator: tuple[str, ...],
    denominator: tuple[str, ...],
    scale: float = 1.0,
) -> float | None:
    denominator_sum = total(rows, denominator)
    if denominator_sum <= 0:
        return None
    return scale * total(rows, numerator) / denominator_sum


def add(values: tuple[float | None, ...]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def multiply(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left * right


def residual(
    parent: float | None,
    children: tuple[float | None, ...],
) -> float | None:
    if parent is None or any(child is None for child in children):
        return None
    return parent - sum(child for child in children if child is not None)


def residual_ratio(
    rows: list[dict[str, str]],
    total_field: str,
    subtract_fields: tuple[str, ...],
) -> float | None:
    denominator = total(rows, (total_field,))
    if denominator <= 0:
        return None
    return (denominator - total(rows, subtract_fields)) / denominator


def average_frequency_mhz(
    cycles: float | None,
    time_us: float | None,
) -> float | None:
    if cycles is None or time_us is None or time_us <= 0:
        return None
    return cycles / time_us


def latency_proxy(
    rows: list[dict[str, str]],
    accesses: str,
    refills: str,
    cycles: str,
) -> float | None:
    cycle_sum = total(rows, (cycles,))
    if cycle_sum <= 0:
        return None
    access_sum = total(rows, (accesses,))
    refill_sum = total(rows, (refills,))
    return ((access_sum - refill_sum) * 15 + refill_sum * 100) / cycle_sum


def stage_metrics(
    root: Path, stage: str, worker: str | None = None
) -> dict[str, float | None]:
    timing = read_rows(root, "time", stage, worker)
    topdown = read_rows(root, "topdown", stage, worker)
    flush = read_rows(root, "flush", stage, worker)
    badspec_branch = read_rows(root, "badspec_branch", stage, worker)
    backend_core = read_rows(root, "backend_core", stage, worker)
    backend_memory = read_rows(root, "backend_memory", stage, worker)
    imix = read_rows(root, "imix", stage, worker)
    imix2 = read_rows(root, "imix2", stage, worker)
    branch = read_rows(root, "branch", stage, worker)
    icache = read_rows(root, "icache", stage, worker)
    dcache = read_rows(root, "dcache", stage, worker)
    l3 = read_rows(root, "l3", stage, worker)
    tlb1 = read_rows(root, "tlb1", stage, worker)
    tlb2 = read_rows(root, "tlb2", stage, worker)

    # 精细化分支/BTB 三组 (2026-09-26)
    br_imm = read_rows(root, "br_imm", stage, worker)
    br_ret = read_rows(root, "br_ret", stage, worker)
    br_indnr = read_rows(root, "br_indnr", stage, worker)

    # BTB 深入分析 四组 (2026-09-26)
    btb_hit = read_rows(root, "btb_hit", stage, worker)
    btb_ubtb = read_rows(root, "btb_ubtb", stage, worker)
    btb_fbtb = read_rows(root, "btb_fbtb", stage, worker)
    btb_main = read_rows(root, "btb_main", stage, worker)

    # 预测器覆盖分析 (2026-10-09) — 组 br_spec, 全部以 0x0012(BR_PRED 总分支)为分母
    br_spec = read_rows(root, "br_spec", stage, worker)

    # 分支误预测拆解 (2026-10-09) — 两组 br_mis_pred1 / br_mis_pred2
    # 中心思想: 0x0010(BR_MIS_PRED)=误预测分支总数, 作为所有细分事件的统一分母。
    #   两组各自都含 0x0010, 保证组内自给(避免跨组相减噪声)。
    #   组1: IF_BP_MISP_BR_DIR(0x1004 方向) / RET(0x100d 返回) / IND(0x1010 间接非Call)
    #   组2: IF_BP_MISP_BR_BLR(0x1013 带link寄存器) / BL(0x1016 直接call) / TGT(0x1017 目标) / LP(0x1019 loop)
    # 各细分误预测占比 = 该事件 / 0x0010(同组内)。
    br_mis_pred1 = read_rows(root, "br_mis_pred1", stage, worker)
    br_mis_pred2 = read_rows(root, "br_mis_pred2", stage, worker)

    # 预测源拆解 (2026-10-09) — 三组 if_bp_pred_br1/2/3, 分母统一为 0x1000(IF_BP_PRED_BR 总预测)
    #   0x1000(总预测)在每组都包含, 组内自给(避免跨组相减噪声)。
    #   组1: SP(0x1001 静态) / UBTB(0x1002) / BTB(0x1003) / COND(0x1005 条件)
    #   组2: RET(0x100c 返回) / IND(0x100f 间接非Call) / BLR(0x1012 带link寄存器) / BL(0x1015 直接call)
    #   组3: LP(0x1018 loop) / FBTB(0x101d) / VBTB(0x101e)
    # 预测占比 = 该事件 / 0x1000(同组内)。
    if_bp_pred_br1 = read_rows(root, "if_bp_pred_br1", stage, worker)
    if_bp_pred_br2 = read_rows(root, "if_bp_pred_br2", stage, worker)
    if_bp_pred_br3 = read_rows(root, "if_bp_pred_br3", stage, worker)

    # 预测源结构替换/冲突/多命中 (2026-10-09) — 组 if_bp_pred_br_ubtb/fbtb/btb
    # 用户指定口径(不使用旧 btb_* 组):
    #   uBTB: 0x1002(命中)分母; 替换=0x101a/0x1002, 冲突=0x101b/0x1002
    #   fBTB: 0x101d(命中)分母; 替换=0x1021/0x101d, 多命中=0x1024/0x101d
    #   BTB : 0x1003(命中)分母; 替换=0x101c/0x1003, 多命中=0x101f/0x1003
    if_bp_pred_br_ubtb = read_rows(root, "if_bp_pred_br_ubtb", stage, worker)
    if_bp_pred_br_fbtb = read_rows(root, "if_bp_pred_br_fbtb", stage, worker)
    if_bp_pred_br_btb = read_rows(root, "if_bp_pred_br_btb", stage, worker)

    # 定向分支指令完整失败率 (2026-10-15) — 组 br_misp_ret / br_misp_ind
    # 中心思想: BR_MIS_PRED(0x0010)=所有分支失败(含没预测但跳转 + 被预测但失败);
    #   但 0x0010 不能细分到具体指令类(IF_BP_MISP_BR_* 只覆盖被预测但失败的子项),
    #   且 BR_UNPRED_TKN_SPEC(0x1009) 无法细分指令类。
    #   故用同组事件自给拆解特定指令类的"完整失败":
    #     失败总数 = (该指令类总数 - 被预测数) + 被预测且失败数
    #     = 没被预测的 + 被预测但失败的
    #   RET: 总数=0x100b, 被预测=0x100c, 被预测失败=0x100d, 全失败=0x0010
    #   IND: 总数=0x100e, 被预测=0x100f, 被预测失败=0x1010, 全失败=0x0010
    #   BR_MISP_XXX/XXX = 失败总数 / 该指令类总数 (该指令类真失败率)
    #   BR_MISP_XXX/MISP = 失败总数 / 0x0010   (该指令类失败占全部分支失败的比例)
    br_misp_ret = read_rows(root, "br_misp_ret", stage, worker)
    br_misp_ind = read_rows(root, "br_misp_ind", stage, worker)

    # 未预测-taken 分支 (0x1009) 中 RET/IND 各自占比 (2026-10-15) — 组 unpred_tkn_sub
    # 0x1009(IF_BR_UNPRED_TKN_SPEC) = 预测器未给出预测但最终跳转的分支总数,
    #   BR_UNPRED_TKN_SPEC 无法直接细分到具体指令类, 故用同组事件自给拆解:
    #     RET 未预测数 = 0x100b(总数) - 0x100c(被预测)
    #     IND 未预测数 = 0x100e(总数) - 0x100f(被预测)
    #   BR_MISP_RET/UNPRED_TKN = (0x100b-0x100c) / 0x1009
    #   BR_MISP_IND/UNPRED_TKN = (0x100e-0x100f) / 0x1009
    unpred_tkn_sub = read_rows(root, "unpred_tkn_sub", stage, worker)
    retire = ratio(topdown, ("0x0008",), ("0x0011",), 1 / 6)
    frontend = ratio(topdown, ("0x2011",), ("0x0011",), 1 / 6)
    fetch_latency = ratio(topdown, ("0x2012",), ("0x0011",))
    fetch_bandwidth = None
    if frontend is not None and fetch_latency is not None:
        fetch_bandwidth = frontend - fetch_latency
    branch_flush = ratio(flush, ("0x0010",), ("0x0011",), 5)
    ooo_flush = ratio(flush, ("0x2010",), ("0x0011",), 9)
    sp_flush = ratio(flush, ("0x104f",), ("0x0011",), 5)
    cycles = total(topdown, ("0x0011",))
    bad_spec = None
    if cycles > 0:
        bad_spec = (total(topdown, ("0x001b",)) - total(topdown, ("0x0008",))) / (
            6 * cycles
        )
    topdown_parts = (retire, frontend, bad_spec)
    backend = None
    if all(value is not None for value in topdown_parts):
        backend = 1 - sum(value for value in topdown_parts if value is not None)
    branch_mispredicts = multiply(
        bad_spec,
        ratio(flush, ("0x0010",), ("0x0010", "0x2010")),
    )
    machine_clears = residual(bad_spec, (branch_mispredicts,))
    indirect_branch = ratio(badspec_branch, ("0x1010",), ("0x0010",))
    push_branch = ratio(
        badspec_branch,
        ("0x1013", "0x1016"),
        ("0x0010",),
    )
    pop_branch = ratio(badspec_branch, ("0x100d",), ("0x0010",))
    other_branch = residual_ratio(
        badspec_branch,
        "0x0010",
        ("0x1013", "0x1016", "0x100d"),
    )
    nuke_flush = ratio(flush, ("0x200f",), ("0x2010",))
    other_flush = residual(1.0, (nuke_flush,))

    memory_bound = ratio(
        backend_memory,
        ("0x7005", "0x7006"),
        ("0x0011",),
    )
    core_bound = residual(
        ratio(backend_core, ("0x7001",), ("0x0011",)),
        (memory_bound,),
    )
    fdiv_stall = ratio(backend_core, ("0x7002",), ("0x0011",))
    div_stall = ratio(backend_core, ("0x7003",), ("0x0011",))
    fsu_stall = ratio(backend_core, ("0x7004",), ("0x0011",))
    exe_ports_util = residual(
        core_bound,
        (fdiv_stall, div_stall, fsu_stall),
    )
    l1_bound = residual(
        ratio(backend_memory, ("0x7005",), ("0x0011",)),
        (ratio(backend_memory, ("0x7007",), ("0x0011",)),),
    )
    l2_bound = residual(
        ratio(backend_memory, ("0x7007",), ("0x0011",)),
        (ratio(backend_memory, ("0x7008",), ("0x0011",)),),
    )
    l3_bound = residual(
        ratio(backend_memory, ("0x7008",), ("0x0011",)),
        (ratio(backend_memory, ("0x7009",), ("0x0011",)),),
    )
    branch_spec = add(
        (
            ratio(imix, ("0x0078",), ("0x001b",)),
            ratio(imix2, ("0x0079", "0x007a"), ("0x001b",)),
        )
    )
    time_us = mean(timing, lambda row: float(row["wall_time_us"]))
    average_cycles = mean(topdown, lambda row: float(row["0x0011"]))

    return {
        "CPU利用率": ratio(
            timing,
            ("thread_cpu_time_us",),
            ("wall_time_us",),
        ),
        "频率(MHz)": average_frequency_mhz(average_cycles, time_us),
        "time(us)": time_us,
        "cycles": average_cycles,
        "instructions": mean(topdown, lambda row: float(row["0x0008"])),
        "IPC": ratio(topdown, ("0x0008",), ("0x0011",)),
        "Retire": retire,
        "FrontendBound": frontend,
        "Fetch Latency Bound": fetch_latency,
        "Idle by iTLB Miss": latency_proxy(
            tlb2,
            "0x0030",
            "0x002e",
            "0x0011",
        ),
        "Idle by iCache Miss": latency_proxy(
            icache,
            "0x0027",
            "0x0028",
            "0x0011",
        ),
        "Flush": add((branch_flush, ooo_flush, sp_flush)),
        "Branch Flush": branch_flush,
        "OoO Flush": ooo_flush,
        "SP Flush": sp_flush,
        "Fetch Bandwidth Bound": fetch_bandwidth,
        "BadSpec": bad_spec,
        "Branch Mispredicts": branch_mispredicts,
        "Indirect Branch": indirect_branch,
        "Push Branch": push_branch,
        "Pop Branch": pop_branch,
        "Other Branch": other_branch,
        "Machine Clears": machine_clears,
        "Nuke Flush": nuke_flush,
        "Other Flush": other_flush,
        "BackendBound": backend,
        "Core Bound": core_bound,
        "Resource Bound": None,
        "FDIV Stall": fdiv_stall,
        "DIV Stall": div_stall,
        "FSU Stall": fsu_stall,
        "Exe Ports Util": exe_ports_util,
        "Memory Bound": memory_bound,
        "L1 Bound": l1_bound,
        "L2 Bound": l2_bound,
        "L3 Bound": l3_bound,
        "Mem Bound": ratio(backend_memory, ("0x7009",), ("0x0011",)),
        "Store Bound": ratio(backend_memory, ("0x7006",), ("0x0011",)),
        "dp_spec": ratio(imix, ("0x0073",), ("0x001b",)),
        "ld_spec": ratio(imix, ("0x0070",), ("0x001b",)),
        "st_spec": ratio(imix2, ("0x0071",), ("0x001b",)),
        "branch_spec": branch_spec,
        "ase_spec": ratio(imix, ("0x8005",), ("0x001b",)),
        "br missrate": ratio(branch, ("0x0022",), ("0x0021",)),
        "br mpki": ratio(branch, ("0x0022",), ("0x0008",), 1000),
        "l1i missrate": ratio(icache, ("0x0001",), ("0x0014",)),
        "l1i mpki": ratio(icache, ("0x0001",), ("0x0008",), 1000),
        "l2i missrate": ratio(icache, ("0x0028",), ("0x0027",)),
        "l2i mpki": ratio(icache, ("0x0028",), ("0x0008",), 1000),
        "l1d missrate": ratio(dcache, ("0x0003",), ("0x0004",)),
        "l1d mpki": ratio(dcache, ("0x0003",), ("0x0008",), 1000),
        "L2d missrate": ratio(dcache, ("0x0017",), ("0x0016",)),
        "L2d mpki": ratio(dcache, ("0x0017",), ("0x0008",), 1000),
        "L3 missrate": ratio(l3, ("0x002a",), ("0x002b",)),
        "L3 mpki": ratio(l3, ("0x002a",), ("0x0008",), 1000),
        "itlb missrate": ratio(tlb1, ("0x0002",), ("0x0026",)),
        "itlb mpki": ratio(tlb1, ("0x0002",), ("0x0008",), 1000),
        "dtlb missrate": ratio(tlb1, ("0x0005",), ("0x0025",)),
        "dtlb mpki": ratio(tlb1, ("0x0005",), ("0x0008",), 1000),
        "stlb missrate": ratio(tlb2, ("0x002d", "0x002e"), ("0x002f", "0x0030")),
        "stlb mpki": ratio(tlb2, ("0x002d", "0x002e"), ("0x0008",), 1000),
        # ── 精细化分支/BTB 三组 (2026-09-26) ──
        # br imm = 0x0078; imm% = 0x0078/(0x0078+0x007a); imm mis% = 0x8111/(0x8110+0x8111)
        "br imm": total(br_imm, ("0x0078",)) if br_imm else None,
        "br imm%": ratio(br_imm, ("0x0078",), ("0x0078", "0x007a")),
        "br imm mis%": ratio(br_imm, ("0x8111",), ("0x8110", "0x8111")),
        # br ret = 0x0079; ret% = 0x0079/(0x0078+0x007a); ret mis% = 0x8115/(0x8114+0x8115)
        "br ret": total(br_ret, ("0x0079",)) if br_ret else None,
        "br ret%": ratio(br_ret, ("0x0079",), ("0x0078", "0x007a")),
        "br ret mis%": ratio(br_ret, ("0x8115",), ("0x8114", "0x8115")),
        # br indnr = 0x007a-0x0079; indnr% = (0x007a-0x0079)/(0x0078+0x007a); indnr mis% = 0x8117/(0x8116+0x8117)
        "br indnr": (total(br_indnr, ("0x007a",)) - total(br_indnr, ("0x0079",)))
        if br_indnr else None,
        "br indnr%": (lambda s, a, d: (a - s) / d if d > 0 else None)(
            total(br_indnr, ("0x0079",)),
            total(br_indnr, ("0x007a",)),
            total(br_indnr, ("0x0078", "0x007a")),
        ),
        "br indnr mis%": ratio(br_indnr, ("0x8117",), ("0x8116", "0x8117")),
        # ── BTB 深入分析 (2026-09-26) ──
        # 命中类以 0x1000(总预测)为分母
        "uBTB命中%": ratio(btb_hit, ("0x1002",), ("0x1000",)),
        "fBTB命中%": ratio(btb_hit, ("0x101d",), ("0x1000",)),
        "BTB命中%": ratio(btb_hit, ("0x1003",), ("0x1000",)),
        # uBTB 替换/冲突 以 0x1002(uBTB命中)为分母
        "uBTB替换%": ratio(btb_ubtb, ("0x101a",), ("0x1002",)),
        "uBTB冲突%": ratio(btb_ubtb, ("0x101b",), ("0x1002",)),
        # fBTB 替换/多命中 以 0x101d(fBTB命中)为分母
        "fBTB替换%": ratio(btb_fbtb, ("0x1021",), ("0x101d",)),
        "fBTB多命中%": ratio(btb_fbtb, ("0x1024",), ("0x101d",)),
        # 主BTB 替换/多命中 以 0x1003(BTB命中)为分母
        "BTB替换%": ratio(btb_main, ("0x101c",), ("0x1003",)),
        "BTB多命中%": ratio(btb_main, ("0x101f",), ("0x1003",)),
        # ── 预测器覆盖分析 (2026-10-09) ──
        # 中心思想: 0x0012(BR_PRED) = 总分支指令数, 作为统一分母, 观察预测器覆盖缺口与误预测。
        # BP_PRED_BR    = 0x1000(预测器给出预测的分支)     / 0x0012
        # BP_NOT_PRED_BR= (0x1008+0x1009)(预测器未预测, 含未预测-taken/not-taken 投机) / 0x0012
        # BR_MIS_PRED   = 0x0010(误预测分支)                / 0x0012
        "BR_PRED total": total(br_spec, ("0x0012",)) if br_spec else None,
        "BP_PRED_BR%": ratio(br_spec, ("0x1000",), ("0x0012",)),
        "BP_NOT_PRED_BR%": ratio(br_spec, ("0x1008", "0x1009"), ("0x0012",)),
        "BR_MIS_PRED%": ratio(br_spec, ("0x0010",), ("0x0012",)),
        # 细分: 未预测分支中, taken 与 not-taken spec 的构成占比
        "BR_UNPRED_NTKN_SPEC%": ratio(br_spec, ("0x1008",), ("0x1008", "0x1009")),
        "BR_UNPRED_TKN_SPEC%": ratio(br_spec, ("0x1009",), ("0x1008", "0x1009")),
        # ── 分支误预测拆解 (2026-10-09) — 组 br_mis_pred1/2, 分母统一为 0x0010(BR_MIS_PRED) ──
        "BR_MIS_PRED total1": total(br_mis_pred1, ("0x0010",)) if br_mis_pred1 else None,
        "BR_MIS_PRED total2": total(br_mis_pred2, ("0x0010",)) if br_mis_pred2 else None,
        "IF_BP_MISP_BR_DIR%": ratio(br_mis_pred1, ("0x1004",), ("0x0010",)),
        "IF_BP_MISP_BR_RET%": ratio(br_mis_pred1, ("0x100d",), ("0x0010",)),
        "IF_BP_MISP_BR_IND%": ratio(br_mis_pred1, ("0x1010",), ("0x0010",)),
        "IF_BP_MISP_BR_BLR%": ratio(br_mis_pred2, ("0x1013",), ("0x0010",)),
        "IF_BP_MISP_BR_BL%": ratio(br_mis_pred2, ("0x1016",), ("0x0010",)),
        "IF_BP_MISP_BR_TGT%": ratio(br_mis_pred2, ("0x1017",), ("0x0010",)),
        "IF_BP_MISP_BR_LP%": ratio(br_mis_pred2, ("0x1019",), ("0x0010",)),
        # ── 预测源拆解 (2026-10-09) — 组 if_bp_pred_br1/2/3, 分母统一为 0x1000(总预测) ──
        "IF_BP_PRED_BR total1": total(if_bp_pred_br1, ("0x1000",)) if if_bp_pred_br1 else None,
        "IF_BP_PRED_BR total2": total(if_bp_pred_br2, ("0x1000",)) if if_bp_pred_br2 else None,
        "IF_BP_PRED_BR total3": total(if_bp_pred_br3, ("0x1000",)) if if_bp_pred_br3 else None,
        "IF_BP_PRED_BR_SP%": ratio(if_bp_pred_br1, ("0x1001",), ("0x1000",)),
        "IF_BP_PRED_BR_UBTB%": ratio(if_bp_pred_br1, ("0x1002",), ("0x1000",)),
        "IF_BP_PRED_BR_BTB%": ratio(if_bp_pred_br1, ("0x1003",), ("0x1000",)),
        "IF_BP_PRED_BR_COND%": ratio(if_bp_pred_br1, ("0x1005",), ("0x1000",)),
        "IF_BP_PRED_BR_RET%": ratio(if_bp_pred_br2, ("0x100c",), ("0x1000",)),
        "IF_BP_PRED_BR_IND%": ratio(if_bp_pred_br2, ("0x100f",), ("0x1000",)),
        "IF_BP_PRED_BR_BLR%": ratio(if_bp_pred_br2, ("0x1012",), ("0x1000",)),
        "IF_BP_PRED_BR_BL%": ratio(if_bp_pred_br2, ("0x1015",), ("0x1000",)),
        "IF_BP_PRED_BR_LP%": ratio(if_bp_pred_br3, ("0x1018",), ("0x1000",)),
        "IF_BP_PRED_BR_FBTB%": ratio(if_bp_pred_br3, ("0x101d",), ("0x1000",)),
        "IF_BP_PRED_BR_VBTB%": ratio(if_bp_pred_br3, ("0x101e",), ("0x1000",)),
        # ── 预测源结构 替换/冲突/多命中 (2026-10-09) — 组 if_bp_pred_br_ubtb/fbtb/btb, 用户指定口径 ──
        "IF_BP_UBTB_REPLACE%": ratio(if_bp_pred_br_ubtb, ("0x101a",), ("0x1002",)),
        "IF_BP_UBTB_CONFLICT%": ratio(if_bp_pred_br_ubtb, ("0x101b",), ("0x1002",)),
        "IF_BP_FBTB_REPLACE%": ratio(if_bp_pred_br_fbtb, ("0x1021",), ("0x101d",)),
        "IF_BP_FBTB_MULTI_HIT%": ratio(if_bp_pred_br_fbtb, ("0x1024",), ("0x101d",)),
        "IF_BP_BTB_REPLACE%": ratio(if_bp_pred_br_btb, ("0x101c",), ("0x1003",)),
        "IF_BP_BTB_MULTI_HIT%": ratio(if_bp_pred_br_btb, ("0x101f",), ("0x1003",)),
        # ── 定向分支指令完整失败率 (2026-10-15) — 组 br_misp_ret / br_misp_ind ──
        # 失败总数 = (指令总数 - 被预测数) + 被预测且失败数 = 没被预测的 + 被预测但失败的
        # RET: 总数=0x100b, 被预测=0x100c, 被预测失败=0x100d, 全局失败=0x0010
        "BR_MISP_RET total": total(br_misp_ret, ("0x100b",)) if br_misp_ret else None,
        "BR_MISP_RET failed": (lambda a, p, m: (a - p) + m)(
            total(br_misp_ret, ("0x100b",)),
            total(br_misp_ret, ("0x100c",)),
            total(br_misp_ret, ("0x100d",)),
        ) if br_misp_ret else None,
        "BR_MISP_RET/RET": (lambda f, a: f / a if a > 0 else None)(
            (lambda a, p, m: (a - p) + m)(
                total(br_misp_ret, ("0x100b",)),
                total(br_misp_ret, ("0x100c",)),
                total(br_misp_ret, ("0x100d",)),
            ),
            total(br_misp_ret, ("0x100b",)),
        ) if br_misp_ret else None,
        "BR_MISP_RET/MISP": (lambda f, m: f / m if m > 0 else None)(
            (lambda a, p, m: (a - p) + m)(
                total(br_misp_ret, ("0x100b",)),
                total(br_misp_ret, ("0x100c",)),
                total(br_misp_ret, ("0x100d",)),
            ),
            total(br_misp_ret, ("0x0010",)),
        ) if br_misp_ret else None,
        # IND: 总数=0x100e, 被预测=0x100f, 被预测失败=0x1010, 全局失败=0x0010
        "BR_MISP_IND total": total(br_misp_ind, ("0x100e",)) if br_misp_ind else None,
        "BR_MISP_IND failed": (lambda a, p, m: (a - p) + m)(
            total(br_misp_ind, ("0x100e",)),
            total(br_misp_ind, ("0x100f",)),
            total(br_misp_ind, ("0x1010",)),
        ) if br_misp_ind else None,
        "BR_MISP_IND/IND": (lambda f, a: f / a if a > 0 else None)(
            (lambda a, p, m: (a - p) + m)(
                total(br_misp_ind, ("0x100e",)),
                total(br_misp_ind, ("0x100f",)),
                total(br_misp_ind, ("0x1010",)),
            ),
            total(br_misp_ind, ("0x100e",)),
        ) if br_misp_ind else None,
        "BR_MISP_IND/MISP": (lambda f, m: f / m if m > 0 else None)(
            (lambda a, p, m: (a - p) + m)(
                total(br_misp_ind, ("0x100e",)),
                total(br_misp_ind, ("0x100f",)),
                total(br_misp_ind, ("0x1010",)),
            ),
            total(br_misp_ind, ("0x0010",)),
        ) if br_misp_ind else None,
        # ── 未预测-taken 分支中 RET/IND 占比 (2026-10-15) — 组 unpred_tkn_sub, 分母为 0x1009 ──
        # 0x1009(IF_BR_UNPRED_TKN_SPEC)=预测器未预测但最终跳转的分支总数
        "BR_UNPRED_TKN total": total(unpred_tkn_sub, ("0x1009",)) if unpred_tkn_sub else None,
        "BR_UNPRED_RET_SUB%": (lambda a, p, d: (a - p) / d if d > 0 else None)(
            total(unpred_tkn_sub, ("0x100b",)),
            total(unpred_tkn_sub, ("0x100c",)),
            total(unpred_tkn_sub, ("0x1009",)),
        ) if unpred_tkn_sub else None,
        "BR_UNPRED_IND_SUB%": (lambda a, p, d: (a - p) / d if d > 0 else None)(
            total(unpred_tkn_sub, ("0x100e",)),
            total(unpred_tkn_sub, ("0x100f",)),
            total(unpred_tkn_sub, ("0x1009",)),
        ) if unpred_tkn_sub else None,
    }


def format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "未采集"
    if metric in PERCENT_METRICS:
        return f"{value:.2%}"
    return f"{value:.2f}"


def pipeline_cycle_shares(
    values: dict[str, dict[str, float | None]],
) -> dict[str, float | None]:
    stage_cycles = [values[stage]["cycles"] for stage in STAGES]
    available_cycles = [value for value in stage_cycles if value is not None]
    shares: dict[str, float | None] = dict.fromkeys(STAGES)
    if len(available_cycles) != len(STAGES):
        return shares
    total_cycles = sum(available_cycles)
    if total_cycles <= 0:
        return shares
    return {
        stage: cycles / total_cycles
        for stage, cycles in zip(STAGES, available_cycles, strict=True)
    }


def write_quality(root: Path) -> None:
    fields = (
        "stage",
        "expected_selected",
        "raw",
        "selected",
        "valid",
        "invalid",
        "status",
    )
    with (root / "collection_quality.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", *fields])
        quality_roots = (
            (root, ""),
            (root / "end_to_end", "end_to_end/"),
        )
        for source_root, prefix in quality_roots:
            if (
                not prefix
                and (root / "collection_profile").is_file()
                and (root / "collection_profile").read_text().strip() == "end_to_end"
            ):
                continue
            for group in GROUPS:
                # ⚠️ 0.23 适配(2026-09-22): 8 阶段(root/ 前缀)无采集数据时文件缺失, 跳过。
                qpath = source_root / group / "collection_quality.csv"
                if not qpath.is_file():
                    continue
                with qpath.open(
                    newline="", encoding="utf-8-sig"
                ) as source:
                    for row in csv.DictReader(source):
                        writer.writerow(
                            [f"{prefix}{group}", *(row[field] for field in fields)]
                        )


def discover_workers(stage_root: Path, stage: str) -> list[str]:
    """从 end_to_end/*/parsed/{stage}.csv 的 worker 列收集出现过的 worker pid。"""
    workers: set[str] = set()
    for path in sorted(stage_root.glob(f"*/parsed/{stage}.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                worker = (row.get("worker") or "").strip()
                if worker:
                    workers.add(worker)
    return sorted(workers, key=int)


def main() -> int:
    args = parse_args()
    # ⚠️ 0.23 适配(2026-09-22): 本采集只做端到端(execute_model_to_sample_tokens),
    #   8 阶段插桩已移除无条件数据。summary 只输出端到端列, 不再遍历 8 阶段。
    # ⚠️ TP=4 适配(2026-10-19): KPERF 行带 worker pid → summary.csv 除汇总列(all)
    #   外, 每个 TP worker 各出一列分开统计; 旧数据无 worker 列时只有 all 列。
    stage_root = args.run_root / "end_to_end"
    columns: list[tuple[str, str | None]] = [("all", None)]
    columns.extend((f"w{worker}", worker) for worker in discover_workers(stage_root, END_TO_END_STAGE))
    values = {
        label: stage_metrics(stage_root, END_TO_END_STAGE, worker=worker)
        for label, worker in columns
    }
    metrics = list(values["all"])
    output = args.run_root / "summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *(label for label, _ in columns)])
        for metric in metrics:
            writer.writerow(
                [
                    metric,
                    *(
                        format_value(metric, values[label][metric])
                        for label, _ in columns
                    ),
                ]
            )
            if metric == "cycles":
                writer.writerow(
                    [
                        CYCLES_SHARE_METRIC,
                        *("不适用" for _ in columns),
                    ]
                )
        writer.writerow([""])
        writer.writerow(
            [
                "热点函数占比：",
                *(
                    "见热点函数" if (args.run_root / "hotspot").exists() else "未采集"
                    for _ in columns
                ),
            ]
        )
    write_quality(args.run_root)
    LOGGER.info("wrote %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
