# 量化策略笔记

## 动态量化 vs 静态量化（PTQ）

| | 动态量化（DQ） | 静态量化（PTQ） |
|---|---|---|
| 校准集 | 不需要 | 需要（几百条代表性样本） |
| 激活动态 | 运行时逐层计算 scale | 离线标定，运行时零开销 |
| 精度 | 通常略低于 FP16，优于 INT8 静态 | 通常最好 |
| 部署复杂度 | 低 | 中（要管理校准集） |
| 端侧适用性 | **首选** | 追求极限精度时再用 |

本项目默认用**动态量化**：端侧最省事，没有校准集维护成本。
若要换静态量化：

```python
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationDataReader, quantize_static

class Reader(CalibrationDataReader):
    def __init__(self, samples): self.samples = samples; self.it = iter(samples)
    def get_next(self): return next(self.it, None)

quantize_static(
    model_input="models/whisper_small_fp32.onnx",
    model_output="models/whisper_small_int8_static.onnx",
    calibration_data_reader=Reader(calib_feeds),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
)
```

## 算子回退（混合精度）

INT8 掉点通常集中在**动态 range 大的层**——Whisper 里典型是 encoder 前几层与 cross-attention 的部分投影。
定位方法：

1. 用 `wer_eval.py` 按场景拆解，确认掉点集中在哪类输入；
2. 用 `onnxruntime.quantization.quantize_static` 配合 `nodes_to_exclude` 排除可疑层；
3. 排除后重测 WER，直到掉点进入可接受范围。

```python
quantize_static(
    ...,
    nodes_to_exclude=["/encoder/layers.0/.../MatMul"],
    extra_options={"MatMulConstBOnly": False},
)
```

**简历里怎么写**：
> 「INT8 相对 FP32 延迟下降 X%，WER 上升 N 个点；按场景拆解发现掉点集中在嘈杂音频与专有名词，对掉点层回退 FP16（混合精度）后 WER 恢复到 M 个点。」

这句话里「逐层回退」是关键词——它证明你做的是**定位**而不是盲量化。

## 动态 shape 的取舍

Whisper 输入长度可变，ONNX 导出时必须开动态轴：

```python
dynamic_axes={
    "input_features": {0: "batch", 2: "length"},
    "logits": {0: "batch", 1: "seq"},
}
```

**代价**：TensorRT 对动态 shape 的优化不如固定 shape 激进，engine 构建时间变长、运行时 kernel 选择更保守。
**收益**：同一 engine 支持任意长度音频，不用为 5s / 30s 各建一个。

端侧场景建议：先固定 shape 建 engine（快、极致优化），服务端才用动态 shape。

## 显存池化

`onnxruntime` 的 `enable_cuda_graph` 与 TensorRT 的 memory pool 可以复用已分配的显存，
避免每次请求反复 `cudaMalloc`：

```python
sess = ort.InferenceSession(
    engine_path,
    providers=[("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "models/cache",
        "trt_max_workspace_size": 1 << 30,
    })],
)
```

**端侧冷启动优化的关键**：engine 常驻内存，首次请求不再付构建代价——
这也是简历里「冷启动 X s → Y s」那两个数字的来源。
