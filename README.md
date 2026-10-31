# Whisper 端侧推理四配置评测

RTX 5070 Ti 16GB 上的 Whisper 部署与量化对比：**PyTorch FP32 / ONNX FP16 / ONNX INT8 / TensorRT FP16**，
四维交叉评测 **延迟 × 显存 × WER × 吞吐**。

与只报一个分数不同，这里做三件事：
1. 四种配置跑同一批音频，公平可比；
2. **WER 按场景拆解**——安静 / 嘈杂 / 专有名词 / 长音频分别统计，量化掉点定位到具体场景而非一个总分；
3. 每条数字附复现命令，报告可直接引用。

## 快速开始

```bash
# 1) 环境（项目本地 venv）
E:\ENV\Uv\uv.exe venv
E:\ENV\Uv\uv.exe pip install -r requirements.txt

# 2) 准备音频（二选一）
python scripts/prepare_data.py --src data/raw --minutes 30   # 用自己的素材
python scripts/prepare_data.py --minutes 30                   # 自动下载 LibriSpeech test-clean

# 3) 导出模型
python scripts/export_onnx.py --model small --quant fp16
python scripts/export_onnx.py --model small --quant int8
python scripts/export_onnx.py --model small --quant fp16 --trt   # 需 TensorRT 10.x

# 4) 跑评测
python scripts/bench.py --manifest data/audio/manifest.jsonl --out reports/
python scripts/wer_eval.py --manifest data/audio/manifest.jsonl --report reports/bench.json

# 5) 看报告
cat reports/summary.md
```

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `scripts/prepare_data.py` | 切片为 16kHz 单声道 wav，生成 `manifest.jsonl`；无本地素材时自动取 LibriSpeech test-clean |
| `scripts/export_onnx.py` | 导出 FP16 ONNX / INT8 动态量化 ONNX，可选构建 TensorRT engine（`trtexec`） |
| `scripts/bench.py` | 四配置逐条测量：延迟、P95、RTF、峰值显存、吞吐 |
| `scripts/wer_eval.py` | 计算各配置 WER，并按场景拆解掉点位置 |

## 评测维度

| 指标 | 含义 | 备注 |
|---|---|---|
| 中位延迟 / P95 | 单条音频端到端推理耗时 | 用中位数而非均值，避免长音频拉偏 |
| RTF | 实时率 = 推理耗时 / 音频时长 | < 1.0 表示快于实时 |
| 峰值显存 | `torch.cuda.max_memory_allocated` / nvidia-smi 采样 | 对比不同配置的常驻开销 |
| WER | 词错率 | 按场景拆解，定位量化掉点 |
| 吞吐 | 1 / 中位延迟 | 单流 QPS |

## 量化策略

- **INT8 用动态量化**（权重 INT8 + 激活动态量化）：无需校准集，部署最省事，是端侧首选。
- **FP16**：显存减半、吞吐翻倍，精度基本无损，是端侧默认档。
- **静态量化（PTQ）** 若要更高精度需准备校准集，见 `docs/quant-notes.md`。
- **算子回退**：若 INT8 某些层掉点明显，保留该层为 FP16（混合精度），本项目 `wer_eval.py` 的分场景统计用于定位这些层。

## 输出

```
reports/
├── bench.json          逐条原始测量
├── summary.json        汇总统计
├── summary.md          对比表（可直接引用）
└── wer.json            WER + 分场景 WER
```

`reports/example-summary.md` 是输出格式示例，实际数字由你本地跑出来后覆盖。

## 复现性

- 固定随机种子（`prepare_data.py --seed`，默认 42），切片可复现。
- 全部脚本接受显式参数，不依赖任何硬编码路径。
- 模型版本通过 `openai/whisper-{model}` 指定，建议在报告中一并记录。

## License

MIT
