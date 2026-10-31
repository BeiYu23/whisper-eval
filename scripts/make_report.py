"""从 bench.json + wer.json 生成 summary.md（保证两者永远自洽）

用法:
    python scripts/make_report.py --reports reports/
"""
import argparse
import json
import statistics
from pathlib import Path


ORDER = ["pytorch_fp32", "onnx_fp16", "onnx_int8", "tensorrt_fp16"]

LABEL = {
    "pytorch_fp32": "PyTorch FP32（基线）",
    "onnx_fp16": "ONNX Runtime FP16",
    "onnx_int8": "ONNX Runtime INT8（动态量化）",
    "tensorrt_fp16": "TensorRT FP16",
    "onnx_int8_fp16_fallback": "INT8 + FP16 算子回退",
}

SCEN_LABEL = {"quiet": "安静", "noisy": "嘈杂", "proper_noun": "专有名词"}


def pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    d = Path(args.reports)

    bench = json.loads((d / "bench.json").read_text(encoding="utf-8"))
    wer = json.loads((d / "wer.json").read_text(encoding="utf-8")) if (d / "wer.json").exists() else {}
    cfg_meta = bench.get("meta", {})

    rows = []
    base_med = None
    for r in bench["results"]:
        name = r["config"]
        lat = r["latency_s"]
        med = statistics.median(lat)
        if base_med is None:
            base_med = med
        w = wer.get("by_config", {}).get(name, {}).get("wer")
        rows.append({
            "name": name,
            "label": LABEL.get(name, name),
            "median_ms": med * 1000,
            "p95_ms": pct(lat, 0.95) * 1000,
            "rtf": statistics.median(r["rtf"]),
            "vram": r.get("peak_vram_gb", 0.0),
            "speedup": base_med / med,
            "wer": w,
        })

    lines = []
    lines.append("# Whisper 四配置评测报告")
    lines.append("")
    lines.append("| 配置 | 中位延迟 | P95 延迟 | 中位 RTF | 峰值显存 | 相对 FP32 | WER |")
    lines.append("|---|---|---|---|---|---|---|")
    for x in rows:
        w = f"{x['wer']*100:.2f}%" if x["wer"] is not None else "n/a"
        lines.append(
            f"| {x['label']} | {x['median_ms']:.0f} ms | {x['p95_ms']:.0f} ms | "
            f"{x['rtf']:.4f} | {x['vram']:.2f} GB | {x['speedup']:.2f}× | {w} |")

    n = len(bench["results"][0]["latency_s"])
    total_s = bench["audio_seconds"]
    lines.append("")
    lines.append(f"样本数：{n}　音频总时长：{total_s/60:.1f} 分钟"
                 f"（均长 {total_s/n:.1f} s）　硬件：RTX 5070 Ti 16GB")

    # 分场景
    by_scen = wer.get("by_scenario", {})
    if by_scen:
        scen_keys = list(next(iter(by_scen.values())).keys())
        lines.append("")
        lines.append("## 掉点分场景拆解")
        lines.append("")
        lines.append("| 场景 | 样本 | " + " | ".join(
            LABEL.get(k, k) for k in ORDER if k in by_scen) + " |")
        lines.append("|---|---|" + "---|" * len([k for k in ORDER if k in by_scen]))
        scen_n = wer.get("scenario_samples", {})
        for sk in scen_keys:
            cells = []
            for k in ORDER:
                if k not in by_scen:
                    continue
                v = by_scen[k].get(sk)
                cells.append(f"{v*100:.2f}%" if v is not None else "-")
            lines.append(f"| {SCEN_LABEL.get(sk, sk)} | {scen_n.get(sk, '-')} | " + " | ".join(cells) + " |")

    # 回退方案
    fb = wer.get("fallback_detail")
    if fb:
        lines.append("")
        lines.append("## 算子回退（混合精度）")
        lines.append("")
        lines.append(f"- **策略**：{fb.get('strategy','')}")
        lines.append(f"- **回退层数**：{fb.get('excluded_layers','-')}　"
                     f"**延迟**：{fb.get('latency_ms','-')} ms　**显存**：{fb.get('vram_gb','-')} GB")
        lines.append(f"- **精度收益**：WER 改善 {fb.get('wer_gain',0)*100:.2f} 个百分点，"
                     f"代价 {fb.get('latency_cost',0)*100:.0f}% 延迟")

    meta = cfg_meta
    if meta.get("engine_build_s"):
        lines.append("")
        lines.append("## 冷启动")
        lines.append("")
        for k, v in meta["engine_build_s"].items():
            lines.append(f"- {LABEL.get(k, k)} 构建耗时：{v} s")
    if meta.get("concurrency"):
        lines.append("")
        lines.append("## 并发能力")
        lines.append("")
        for k, v in meta["concurrency"].items():
            lines.append(f"- {LABEL.get(k, k)}：{v['streams']} 路并发，P95 {v['p95_latency_ms']} ms")

    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {d/'summary.md'} 已生成")
    for x in rows:
        print(f"  {x['label']:28s} {x['median_ms']:6.0f}ms  {x['speedup']:.2f}×  "
              f"{x['vram']:.2f}GB  WER={x['wer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
