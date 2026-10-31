"""WER 评测：对比各配置与参考文本的词错率

用法:
    python scripts/wer_eval.py --manifest data/audio/manifest.jsonl --report reports/bench.json
产出:
    reports/wer.json      各配置 WER + 分场景 WER
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def count_errors(ref: str, hyp: str) -> tuple[int, int]:
    """返回 (错误数, 参考词数)。

    jiwer.process_words(reference, hypothesis) 返回 WordOutput，
    其中 substitutions / deletions / insertions / hits 是各类型的计数。
    注意：process_words 需要两个位置参数，不能只传一个。
    """
    from jiwer import process_words
    out = process_words(normalize(ref), normalize(hyp))
    errors = out.substitutions + out.deletions + out.insertions
    words = out.substitutions + out.deletions + out.hits
    return errors, words


def compute_wer(ref: str, hyp: str) -> float | None:
    try:
        from jiwer import wer as jiwer_wer
    except ImportError:
        return None
    return jiwer_wer(normalize(ref), normalize(hyp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/audio/manifest.jsonl")
    ap.add_argument("--report", default="reports/bench.json")
    ap.add_argument("--out", default="reports/wer.json")
    args = ap.parse_args()

    manifest = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        manifest[row["id"]] = row

    bench = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = bench["results"]

    out = {"by_config": {}, "by_scenario": {}}
    for r in results:
        cfg = r["config"]
        # bench.json 里 texts 与 manifest 顺序一一对应
        rows = list(manifest.values())
        tot_err = tot_n = 0
        scen = defaultdict(lambda: [0, 0])
        scored = 0
        for i, hyp in enumerate(r["texts"]):
            if i >= len(rows):
                break
            row = rows[i]
            ref = row.get("text", "").strip()
            if not ref:
                continue
            scored += 1
            errors, words = count_errors(ref, hyp)
            tot_err += errors
            tot_n += words
            s = row.get("scenario", "unknown")
            scen[s][0] += errors
            scen[s][1] += words
        out["by_config"][cfg] = {
            "wer": (tot_err / tot_n) if tot_n else None,
            "errors": tot_err,
            "words": tot_n,
            "scored_utts": scored,
        }
        out["by_scenario"][cfg] = {
            k: (v[0] / v[1] if v[1] else None) for k, v in scen.items()
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("配置 WER：")
    for cfg, v in out["by_config"].items():
        w = v["wer"]
        print(f"  {cfg:16s} {('%.2f%%' % (w*100)) if w is not None else 'n/a':>8s}  "
              f"(errors={v['errors']}, words={v['words']})")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
