"""Whisper 端侧推理四配置评测 —— 数据集准备与音频切分

产出:
    data/audio/<utt_id>.wav       16kHz 单声道 PCM
    data/audio/manifest.jsonl     每行 {"id","path","text","scenario"}

manifest 里的 text 是**参考文本（ground truth）**，WER 评测必需。
LibriSpeech 自带转写文本，因此默认数据集能直接算 WER。
自备音频若无标注，text 留空，WER 会被跳过（只出延迟/显存/RTF）。
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path


SR = 16000


def run_ffmpeg_slice(src: Path, dst: Path, start: float, dur: float) -> bool:
    """从长音频切一段 16k 单声道 wav。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", str(SR),
        "-c:a", "pcm_s16le", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return dst.exists() and dst.stat().st_size > 2048
    except subprocess.CalledProcessError:
        return False


def write_wav(dst: Path, tensor, sr: int) -> bool:
    """用 soundfile 写 wav（torchaudio 官方推荐路径，避免 torchaudio 2.x 的 IO 变动）。"""
    import numpy as np
    import soundfile as sf
    try:
        data = tensor.numpy() if hasattr(tensor, "numpy") else tensor
        if data.ndim > 1:
            data = data.mean(axis=0)
        if sr != SR:
            # torchaudio.functional.resample 需要 torchaudio；缺失时交给 ffmpeg 处理
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), data.astype("float32"), SR)
        return dst.stat().st_size > 2048
    except Exception:
        return False


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def build_from_librispeech(out: Path, rng: random.Random, limit: int) -> list[dict]:
    """从 LibriSpeech test-clean 构建带参考文本的评测集。"""
    try:
        import torchaudio
    except Exception as e:
        print(f"[data] torchaudio 不可用: {e}", file=sys.stderr)
        return []

    root = out / "_librispeech"
    try:
        ds = torchaudio.datasets.LIBRISPEECH(root=str(root), split="test-clean", download=True)
    except Exception as e:
        print(f"[data] LibriSpeech 下载失败: {e}", file=sys.stderr)
        return []

    print(f"[data] LibriSpeech test-clean: {len(ds)} 条")
    rows: list[dict] = []
    for i in range(len(ds)):
        if limit and len(rows) >= limit:
            break
        try:
            wav, sr, transcript, *_ = ds[i]
        except Exception:
            continue
        text = normalize_text(transcript)
        if not text:
            continue
        utt_id = f"ls{i:04d}"
        dst = out / f"{utt_id}.wav"
        if write_wav(dst, wav, sr):
            rows.append({
                "id": utt_id,
                "path": f"{utt_id}.wav",
                "text": text,
                "scenario": "librispeech",
                "source": "librispeech/test-clean",
            })
        if len(rows) % 20 == 0:
            print(f"[data] 已准备 {len(rows)} 条")
    return rows


def build_from_local(out: Path, src_dir: Path, rng: random.Random, limit: int) -> list[dict]:
    """从本地音频目录构建评测集。

    支持两种命名约定来提供参考文本：
      1) 同名 .txt / .md 文件（内容即参考文本）
      2) 文件名里带 <utt_id>|<text> 形式
    两者都没有时 text 留空 —— WER 会被跳过。
    """
    files = sorted(
        p for p in src_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}
    )
    if not files:
        return []
    print(f"[data] 本地音频 {len(files)} 个")

    rows: list[dict] = []
    for i, src in enumerate(files):
        if limit and len(rows) >= limit:
            break
        utt_id = f"loc{i:04d}"
        dst = out / f"{utt_id}.wav"

        # 参考文本来源
        text = ""
        stem_txt = src.with_suffix(".txt")
        stem_md = src.with_suffix(".md")
        for cand in (stem_txt, stem_md):
            if cand.exists():
                text = normalize_text(cand.read_text(encoding="utf-8", errors="ignore"))
                break
        if not text and "|" in src.stem:
            head, _, tail = src.stem.partition("|")
            utt_id = f"loc_{head.strip()}"
            text = normalize_text(tail)

        dur = rng.uniform(5.0, 20.0)
        start = rng.uniform(0.0, 30.0)
        if not run_ffmpeg_slice(src, dst, start, dur):
            continue
        rows.append({
            "id": utt_id,
            "path": f"{utt_id}.wav",
            "text": text,
            "scenario": "local",
            "source": src.name,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--src", default="", help="本地音频目录（留空则用 LibriSpeech）")
    ap.add_argument("--limit", type=int, default=60, help="样本条数，0 = 全部")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    rows: list[dict] = []
    if args.src:
        rows = build_from_local(out, Path(args.src), rng, args.limit)
    if not rows:
        rows = build_from_librispeech(out, rng, args.limit)

    if not rows:
        print("[data] 无可用音频。", file=sys.stderr)
        print("[data] 方案 A：把音频放进 data/raw/，同名 .txt 放参考文本", file=sys.stderr)
        print("[data] 方案 B：不传 --src，自动下载 LibriSpeech test-clean", file=sys.stderr)
        return 1

    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_s = 0.0
    for r in rows:
        try:
            import soundfile as sf
            info = sf.info(str(out / r["path"]))
            total_s += info.duration
        except Exception:
            pass
    labeled = sum(1 for r in rows if r["text"])

    print(f"\n[data] 完成：{len(rows)} 条，{total_s/60:.1f} 分钟")
    print(f"[data] 带参考文本：{labeled}/{len(rows)} 条"
          f"{'  ← WER 可评测' if labeled else '  ← 无标注，WER 将跳过'}")
    print(f"[data] manifest → {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
