# 权重与测试划分核对

核对日期：2026-11-09。以 research/experiments/Log/log 中同目录运行记录为依据。

| 数据集 | 条件 | 权重数 | 实际划分 | 测试样本 |
|---|---|---:|---|---:|
| CUMTB | clean | 1 | cumtb_source_file_holdout | 1537 |
| CUMTB | gaussian | 5 | cumtb_balanced575_random_train_test | 1557 |
| CUMTB | fan | 5 | cumtb_balanced575_random_train_test | 1557 |
| DCASE | clean | 1 | class_random_train_test | 600 |
| DCASE | gaussian | 5 | class_random_train_test | 600 |
| DCASE | fan | 5 | class_random_train_test | 600 |

## 检查结果

已核对检查点与日志的运行名称、明确记录的特征路径和划分路径、类别顺序、分类头类别数、测试样本数量和混淆矩阵总数。未重新执行测试或计算准确率。

- CUMTB gaussian / glam_fusion_seed43_20260615-203358：logged_feature_directory_missing_in_release。日志要求特征目录 `stft_cumtb_cond1_1rpm_clean9_minus4db_win25ms_nfft2048_hop10ms_1s_len128_remove_dc_peak`。

其余现存特征目录的对应测试样本均齐全。

CUMTB 干净条件的归档运行采用 source_file_holdout，3707/1537；CUMTB 加噪运行采用 balanced575_random_train_test，3618/1557。DCASE 全部为 class_random_train_test，1400/600。划分种子均为 44。

CUMTB -4 dB 高斯权重引用的旧目录名包含 clean9_minus4db。当前已复制的同 SNR 目录名以 snr_neg4db 结尾；仅凭噪声强度和文件名不能确认特征内容完全相同，因此未自动替换对应关系。

原有划分文件保持不变，本次未将划分文件复制到 Github。各权重的完整映射见 REPRODUCTION_CHECK.json。
