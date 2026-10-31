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
python -m venv .venv
pip install -r requirements.txt

# 2) 准备音频（默认 LibriSpeech test-clean，自带转写文本 → 可直接算 WER）
python scripts/prepare_data.py --limit 60

# 3) 导出模型
python scripts/export_onnx.py --model small --quant fp16
python scripts/export_onnx.py --model small --quant int8
python scripts/export_onnx.py --model small --quant fp16 --trt   # 需 TensorRT 10.x

# 4) 跑评测
python scripts/bench.py --out reports/
python scripts/wer_eval.py --report reports/bench.json

# 5) 看报告
cat reports/summary.md
```

> 用自己的音频：`python scripts/prepare_data.py --src data/raw --limit 60`。
> 同名 `.txt` 放参考文本即可参与 WER；没有标注则只出延迟 / 显存 / RTF。

## 脚本说明

| 脚本 | 作用 |
|---|---|
| `scripts/prepare_data.py` | 构建评测集：默认拉 LibriSpeech test-clean（带转写文本），或切片本地音频；产出 `manifest.jsonl` |
| `scripts/export_onnx.py` | 导出 FP16 ONNX / INT8 动态量化 ONNX，可选构建 TensorRT engine（`trtexec`） |
| `scripts/bench.py` | 四配置逐条测量：延迟、P95、RTF、峰值显存、吞吐，并留档 `config.json` |
| `scripts/wer_eval.py` | 计算各配置 WER，并按场景拆解掉点位置 |

## CI 自检

每次 push 会触发 `.github/workflows/ci.yml`：

- **lint job**：四个脚本的语法检查 + CLI 参数自检（不装重型依赖也能跑）
- **selfcheck job**：合成音频端到端冒烟——生成带参考文本的合成数据集，跑通 WER 统计链路

CI 绿灯 = 脚本没坏，**任何人 clone 后都能复现你的数字**。

## 复现

环境、随机种子、样本规模、硬件配置全部留档在 `reports/config.json`，
复现步骤见 [`docs/reproduce.md`](docs/reproduce.md)。

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
├── wer.json            WER + 分场景 WER
└── config.json         本次运行的完整配置（模型 / 样本数 / 硬件 / 种子）
```

`reports/example-summary.md` 是输出格式示例，实际数字由你本地跑出来后覆盖。

## 复现性

- 固定随机种子（`prepare_data.py --seed`，默认 42），切片可复现。
- 全部脚本接受显式参数，不依赖任何硬编码路径。
- 模型版本通过 `openai/whisper-{model}` 指定，建议在报告中一并记录。
- 完整复现步骤见 [`docs/reproduce.md`](docs/reproduce.md)。

## License

MIT
