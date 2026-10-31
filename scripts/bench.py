"""Whisper 四配置评测：PyTorch FP32 / ONNX FP16 / ONNX INT8 / TensorRT FP16

用法:
    python scripts/bench.py --manifest data/audio/manifest.jsonl --out reports/
产出:
    reports/bench.json      原始测量数据
    reports/summary.md      汇总表（可直接贴进简历）
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------- 公共工具

def load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == 16000, f"期望 16kHz，实际 {sr}"
    return data.astype(np.float32)


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q))


def fmt_ms(x: float) -> str:
    return f"{x*1000:.0f} ms"


# ---------------------------------------------------------------- PyTorch FP32

def bench_pytorch(model_name: str, audios: list[np.ndarray]) -> dict:
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    model.eval()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    lat, rtf, texts = [], [], []
    t_all = time.perf_counter()
    for audio in audios:
        mel = whisper.log_mel_spectrogram(audio)
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.decode(mel, language="en", fp16=False, temperature=0.0)
        dt = time.perf_counter() - t0
        lat.append(dt)
        rtf.append(dt / max(len(audio) / 16000, 1e-6))
        texts.append(res["text"].strip())

    wall = time.perf_counter() - t_all
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
    return {
        "config": "pytorch_fp32",
        "latency_s": lat, "rtf": rtf, "texts": texts,
        "wall_s": wall, "peak_vram_gb": peak,
        "device": device,
    }


# ---------------------------------------------------------------- ONNX Runtime

def bench_onnx(model_name: str, audios: list[np.ndarray], quant: str) -> dict:
    """quant: fp16 | int8"""
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    onnx_path = Path(f"models/whisper_{model_name}_{quant}.onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} 不存在，请先运行 scripts/export_onnx.py --quant {quant}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    feat = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_name}")

    lat, rtf, texts = [], [], []
    t_all = time.perf_counter()
    for audio in audios:
        inp = feat(audio, sampling_rate=16000, return_tensors="np")
        input_features = inp["input_features"].astype(np.float16 if quant == "fp16" else np.float32)
        feed = {sess.get_inputs()[0].name: input_features}
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        dt = time.perf_counter() - t0
        lat.append(dt)
        rtf.append(dt / max(len(audio) / 16000, 1e-6))
        texts.append(decode_onnx(sess, out, model_name))

    wall = time.perf_counter() - t_all
    peak = 0.0
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        peak = int(r.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        pass
    return {
        "config": f"onnx_{quant}",
        "latency_s": lat, "rtf": rtf, "texts": texts,
        "wall_s": wall, "peak_vram_gb": peak,
    }


def decode_onnx(sess, out, model_name: str) -> str:
    """从 ONNX 输出解码文本（简化版：贪心 argmax + tokenizer）。"""
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(f"openai/whisper-{model_name}")
    logits = out[0]
    ids = np.argmax(logits, axis=-1)
    if ids.ndim > 1:
        ids = ids[0]
    ids = ids[ids < tok.vocab_size]
    return tok.decode(ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------- TensorRT

def bench_tensorrt(model_name: str, audios: list[np.ndarray]) -> dict:
    import onnxruntime as ort
    onnx_path = Path(f"models/whisper_{model_name}_fp16.onnx")
    trt_path = Path(f"models/whisper_{model_name}_fp16.engine")
    if not trt_path.exists():
        raise FileNotFoundError(f"{trt_path} 不存在，请先运行 scripts/export_onnx.py --quant fp16 --trt")

    sess = ort.InferenceSession(str(trt_path), providers=["TensorrtExecutionProvider",
                                                          "CUDAExecutionProvider",
                                                          "CPUExecutionProvider"])
    from transformers import WhisperFeatureExtractor
    feat = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_name}")

    lat, rtf, texts = [], [], []
    t_all = time.perf_counter()
    for audio in audios:
        inp = feat(audio, sampling_rate=16000, return_tensors="np")
        feed = {sess.get_inputs()[0].name: inp["input_features"].astype(np.float16)}
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        dt = time.perf_counter() - t0
        lat.append(dt)
        rtf.append(dt / max(len(audio) / 16000, 1e-6))
        texts.append(decode_onnx(sess, out, model_name))
    wall = time.perf_counter() - t_all
    return {
        "config": "tensorrt_fp16",
        "latency_s": lat, "rtf": rtf, "texts": texts,
        "wall_s": wall, "peak_vram_gb": 0.0,
    }


# ---------------------------------------------------------------- 汇总

def summarize(results: list[dict]) -> dict:
    base = None
    for r in results:
        if r["config"] == "pytorch_fp32":
            base = r
    out = {"configs": {}}
    for r in results:
        lat = r["latency_s"]
        med = statistics.median(lat)
        out["configs"][r["config"]] = {
            "n": len(lat),
            "median_latency_s": med,
            "p95_latency_s": percentile(lat, 95),
            "median_rtf": statistics.median(r["rtf"]),
            "max_rtf": max(r["rtf"]),
            "throughput_1x": 1.0 / med,
            "wall_s": r["wall_s"],
            "peak_vram_gb": r.get("peak_vram_gb", 0.0),
            "speedup_vs_fp32": (base["latency_s"] and
                                statistics.median(base["latency_s"]) / med) if base else None,
        }
    return out


def write_markdown(summary: dict, path: Path, wer_by_config: dict | None = None) -> None:
    lines = ["# Whisper 四配置评测报告", ""]
    lines += ["| 配置 | 中位延迟 | P95 延迟 | 中位 RTF | 峰值显存 | 相对 FP32 | WER |",
              "|---|---|---|---|---|---|---|"]
    base = summary["configs"].get("pytorch_fp32", {})
    for name, c in summary["configs"].items():
        wer = ""
        if wer_by_config and name in wer_by_config:
            w = wer_by_config[name]
            wer = f"{w*100:.2f}%" if w is not None else "n/a"
        sp = c["speedup_vs_fp32"]
        sp_s = f"{sp:.2f}×" if sp else "-"
        lines.append(
            f"| {name} | {fmt_ms(c['median_latency_s'])} | {fmt_ms(c['p95_latency_s'])} | "
            f"{c['median_rtf']:.3f} | {c['peak_vram_gb']:.2f} GB | {sp_s} | {wer} |")
    lines.append("")
    lines.append(f"样本数：{next(iter(summary['configs'].values()))['n']}　"
                 f"硬件：RTX 5070 Ti 16GB")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/audio/manifest.jsonl")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--model", default="small")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="pytorch_fp32,onnx_fp16,onnx_int8,tensorrt_fp16")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("manifest 为空，先跑 scripts/prepare_data.py")
        return 1

    audios = [load_wav(r["path"]) for r in rows]
    total_audio_s = sum(len(a) / 16000 for a in audios)
    print(f"[bench] {len(audios)} 条音频，合计 {total_audio_s/60:.1f} 分钟")

    want = [c.strip() for c in args.configs.split(",") if c.strip()]
    results = []
    for cfg in want:
        print(f"\n[bench] === {cfg} ===")
        try:
            if cfg == "pytorch_fp32":
                r = bench_pytorch(args.model, audios)
            elif cfg == "onnx_fp16":
                r = bench_onnx(args.model, audios, "fp16")
            elif cfg == "onnx_int8":
                r = bench_onnx(args.model, audios, "int8")
            elif cfg == "tensorrt_fp16":
                r = bench_tensorrt(args.model, audios)
            else:
                print(f"未知配置 {cfg}，跳过")
                continue
            results.append(r)
            print(f"[bench] {cfg}: 中位 {fmt_ms(statistics.median(r['latency_s']))}")
        except Exception as e:
            print(f"[bench] {cfg} 失败: {e}")

    if not results:
        print("没有成功的配置")
        return 1

    (outdir / "bench.json").write_text(json.dumps(
        {"results": results, "audio_seconds": total_audio_s},
        ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(results)
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    write_markdown(summary, outdir / "summary.md")
    print(f"\n[bench] 报告：{outdir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
