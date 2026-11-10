# NR-GLADNet 权重

来源：research/experiments/Log/checkpoint，整理日期：2026-11-09。

按用户指定的已整理实验目录选取 glam/glam_fusion 权重，保留原始目录结构和文件内容。

| 来源目录 | 数据集 | 条件 | 数量 |
|---|---|---|---:|
| CumtbClean | CUMTB | 干净 | 1 |
| CumtbNoise | CUMTB | 高斯噪声 | 5 |
| CumtbNoise2 | CUMTB | 风扇干扰 | 5 |
| DcaseClean | DCASE | 干净 | 1 |
| DcaseNoise | DCASE | 高斯噪声 | 5 |
| DcaseNoise2 | DCASE | 风扇干扰 | 5 |

共 22 份检查点。推理读取检查点的 model_state，并采用 manifest.json 中对应的 model_config 构造 LiteGLAMFusionNetV2。

每份副本均与新指定来源进行 SHA-256 校验、严格权重加载和有限值检查。此次未比较旧集合、未重新评估准确率；具体权重与特征、测试划分的匹配仍需核实。

源目录保持不变，未上传远程仓库。
