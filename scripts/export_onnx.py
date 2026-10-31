"""导出 ONNX（FP16 / INT8 动态量化）并构建 TensorRT engine

用法:
    python scripts/export_onnx.py --model small --quant fp16
    python scripts/export_onnx.py --model small --quant int8
    python scripts/export_onnx.py --model small --quant fp16 --trt
"""
import argparse
import shutil
from pathlib import Path


def export_fp16(model_name: str, out_path: Path) -> None:
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_name}", torch_dtype=torch.float16)
    model.eval()

    dummy = torch.randn(1, 80, 3000, dtype=torch.float16)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch", 2: "length"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[export] FP16 ONNX → {out_path}")


def export_int8(model_name: str, out_path: Path) -> None:
    """动态量化：权重 INT8、激活动态量化。无需校准集，部署最省事。"""
    import torch
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_name}")
    model.eval()

    from onnxruntime.quantization import QuantType, quantize_dynamic
    tmp = out_path.with_suffix(".fp32.onnx")
    dummy = torch.randn(1, 80, 3000, dtype=torch.float32)
    torch.onnx.export(
        model, (dummy,), str(tmp),
        input_names=["input_features"], output_names=["logits"],
        dynamic_axes={"input_features": {0: "batch", 2: "length"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17, do_constant_folding=True,
    )
    quantize_dynamic(
        model_input=str(tmp), model_output=str(out_path),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    tmp.unlink(missing_ok=True)
    print(f"[export] INT8 ONNX → {out_path}")


def build_trt(onnx_path: Path, engine_path: Path) -> None:
    """用 trtexec 构建 FP16 engine（需要 TensorRT 已安装）。"""
    import subprocess
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--minShapes=input_features:1x80x1000",
        "--optShapes=input_features:1x80x3000",
        "--maxShapes=input_features:1x80x3000",
        "--memPoolSize=workspace:4096M",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[export] TensorRT engine → {engine_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--quant", choices=["fp16", "int8"], required=True)
    ap.add_argument("--trt", action="store_true", help="同时构建 TensorRT engine")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = outdir / f"whisper_{args.model}_{args.quant}.onnx"

    if not onnx_path.exists():
        if args.quant == "fp16":
            export_fp16(args.model, onnx_path)
        else:
            export_int8(args.model, onnx_path)
    else:
        print(f"[export] 已存在，跳过：{onnx_path}")

    if args.trt:
        if args.quant != "fp16":
            print("[export] TensorRT 仅对 FP16 构建，请加 --quant fp16")
            return 1
        engine = outdir / f"whisper_{args.model}_fp16.engine"
        if not engine.exists():
            build_trt(onnx_path, engine)
        else:
            print(f"[export] 已存在，跳过：{engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
