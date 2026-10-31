# 复现本项目的评测结果

本仓库的所有数字来自一次本地运行，配置记录在 `reports/config.json`。
任何人 clone 后都可按下面步骤复现。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti（16 GB） |
| 驱动 | 581.29 |
| CUDA | 12.x（PyTorch 自带 wheel） |
| Python | 3.12.11（项目本地 `.venv`） |
| OS | Windows 11 |

## 复现步骤

```bash
# 1. 环境
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements.txt

# 2. 数据（默认 LibriSpeech test-clean，自带转写文本 → 可直接算 WER）
python scripts/prepare_data.py --limit 60

# 3. 导出三个配置
python scripts/export_onnx.py --model small --quant fp16
python scripts/export_onnx.py --model small --quant int8
python scripts/export_onnx.py --model small --quant fp16 --trt   # 需 TensorRT 10.x

# 4. 测量
python scripts/bench.py --out reports/
python scripts/wer_eval.py --report reports/bench.json
```

## 一致性保证

- **固定随机种子**：`prepare_data.py --seed 42`，音频切片可复现。
- **样本固定**：`--limit 60` 时使用 LibriSpeech test-clean 的前 60 条，按固定顺序。
- **配置留档**：实际使用的模型名、样本数、硬件信息写入 `reports/config.json`。
- **CI 自检**：每次 push 会跑一轮最小规模端到端冒烟（合成音频 + WER 统计链路），
  保证脚本不会静默损坏——见 `.github/workflows/ci.yml` 的绿灯状态。

## 数字怎么读

报告分两层：

1. **总量**（`reports/summary.md`）：延迟中位数 / P95、RTF、峰值显存、相对 FP32 加速比、WER。
2. **分场景**（`reports/wer.json` 的 `by_scenario`）：量化掉点落在哪类输入上。

第 2 层是本项目的重点。INT8 的价值不只在「快多少」，还在「**掉在哪、能不能修**」——
定位到具体层后做 FP16 回退（混合精度），才是端侧量化的完整工程闭环。

## 已知限制

- TensorRT 配置需要单独安装 TensorRT 10.x（`trtexec` 不随 pip 分发），
  CI 环境不跑这一格，本地运行才有该行数据。
- WER 采用贪心解码（`temperature=0`），未做 beam search——与生产配置一致即可，
  但不要与文献中的 beam search 数字直接对比。
- 显存读数来自 `torch.cuda.max_memory_allocated` 与 `nvidia-smi` 采样，
  两者口径不同，横向比较时以同一配置内的相对变化为准。
