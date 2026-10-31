"""Whisper 端侧推理四配置评测 —— 数据集准备与音频切分

用法:
    python scripts/prepare_data.py --out data/audio --minutes 30
生成:
    data/audio/<speaker>/<utt_id>.wav    16kHz 单声道 PCM
    data/audio/manifest.jsonl           每行: {"id","path","text","scenario"}
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def slice_audio(src: Path, dst: Path, start: float, dur: float, sr: int = 16000) -> bool:
    """用 ffmpeg 从长音频切一段 16k 单声道 wav。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return dst.exists() and dst.stat().st_size > 1024
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--minutes", type=int, default=30, help="总音频时长（分钟）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", default="", help="可选：本地音频目录，优先使用")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # 1) 优先使用本地音频（用户自己放的素材）
    local = Path(args.src) if args.src else Path("data/raw")
    local_files = []
    if local.exists():
        local_files = sorted(
            p for p in local.rglob("*")
            if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}
        )
    if local_files:
        print(f"[data] 发现本地音频 {len(local_files)} 个")
    else:
        print("[data] 未发现本地音频，尝试下载公开语音数据集…")
        code = prepare_torchaudio(out, rng)
        if code != 0:
            return code

    # 2) 切片
    manifest = out / "manifest.jsonl"
    n_ok = 0
    total_s = 0.0
    with manifest.open("w", encoding="utf-8") as fh:
        for i, src in enumerate(local_files):
            # 随机切 5~20s
            dur = rng.uniform(5.0, 20.0)
            start = rng.uniform(0.0, 30.0)
            utt_id = f"utt{i:04d}"
            dst = out / "slices" / f"{utt_id}.wav"
            if not slice_audio(src, dst, start, dur):
                continue
            n_ok += 1
            total_s += dur
            fh.write(json.dumps({
                "id": utt_id,
                "path": str(dst.relative_to(out)).replace("\\", "/"),
                "scenario": "local",
                "text": "",
            }, ensure_ascii=False) + "\n")
            if n_ok % 20 == 0:
                print(f"[data] 已切 {n_ok} 段 / {total_s/60:.1f} 分钟")

    print(f"[data] 完成：{n_ok} 段，共 {total_s/60:.1f} 分钟 → {manifest}")
    if n_ok == 0:
        print("[data] 警告：没有可用音频。请把音频放进 data/raw/ 或指定 --src", file=sys.stderr)
        return 1
    return 0


def prepare_torchaudio(out: Path, rng: random.Random) -> int:
    """下载 LibriSpeech test-clean 子集作为默认素材。"""
    try:
        import torchaudio
    except Exception as e:
        print(f"[data] torchaudio 不可用: {e}", file=sys.stderr)
        print("[data] 请手动准备音频：把文件放进 data/raw/ 后重跑 --src data/raw", file=sys.stderr)
        return 1
    try:
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(out / "_librispeech"), split="test-clean", download=True)
    except Exception as e:
        print(f"[data] LibriSpeech 下载失败: {e}", file=sys.stderr)
        return 1
    print(f"[data] LibriSpeech test-clean: {len(dataset)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
