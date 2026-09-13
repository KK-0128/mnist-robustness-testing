# Robustness Lab — 模型鲁棒性测试与防御评估工程

[![CI](https://github.com/KK-0128/mnist-robustness-testing/actions/workflows/ci.yml/badge.svg)](https://github.com/KK-0128/mnist-robustness-testing/actions/workflows/ci.yml)

把毕业设计《Robust Design of Deep Neural Networks against Adversarial Attacks》
中的一次性实验脚本，重构为**可复现、可测试、可 CI 集成**的标准 Python 测试工程。

- 攻击：FGSM / PGD / C&W(L2)（统一 `Attack` 接口，可插拔）
- 防御：FGSM-AT / PGD-AT / CW-AT / 高斯增强训练 / 随机平滑（Randomized Smoothing）
- 评估：干净准确率、对抗准确率、平滑准确率、L2 可证明认证半径（Clopper-Pearson 置信界）
- 数据集：MNIST + 标准 CNN（与报告一致）

## 快速开始

```bash
# 1. 安装（CI/本地一致：CPU 版 torch）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# 2. 跑一遍基线基准测试（MNIST 会自动下载到 ./data）
python -m robustness_lab.cli --config experiments/configs/baseline.yaml --tag baseline

# 3. 跑测试套件（冒烟级，CPU 可跑）
pytest -m "smoke" -v

# 4. 本地 lint
ruff check src tests
```

## 目录结构

```
mnist-robustness-testing/
├── pyproject.toml            # 依赖 + pytest/ruff 配置
├── src/robustness_lab/       # 源码包
│   ├── config.py             # 全部超参数（dataclass + YAML 加载）
│   ├── data.py               # MNIST 加载与预处理
│   ├── models.py             # CNN（Net）+ 工厂函数
│   ├── attacks/              # FGSM / PGD / CW，统一 Attack 接口
│   ├── defenses/             # 对抗训练 / 高斯增强 / 随机平滑
│   ├── evaluation/           # metrics（纯函数）/ runner / report
│   └── cli.py                # 命令行入口
├── tests/                    # pytest 套件（conftest + 4 个测试文件）
├── experiments/configs/      # YAML 实验配置
├── experiments/results/      # 结果输出（JSON/CSV/Markdown，gitignore）
└── .github/workflows/ci.yml  # lint + 冒烟测试 + 覆盖率
```

## 实验配置

| 配置文件 | 对应报告章节 | 说明 |
|---|---|---|
| `baseline.yaml` | 4.2 | 标准训练基线 |
| `fgsm_at.yaml` | 4.3.1 | FGSM 对抗训练（ε=0.1） |
| `pgd_at.yaml` | 4.3.2 | PGD 对抗训练（ε=0.1, α=0.01, iters=10） |
| `cw_at.yaml` | 4.3.3 | CW 对抗训练（c=10, k=0, lr=0.01, iter=100） |
| `gaussian_rs.yaml` | 4.4 | 高斯增强训练 + 随机平滑推理 |
| `pgd_at_rs.yaml` | 4.5 | PGD-AT（ε=0.3, iters=40）+ 随机平滑组合 |

> **关于 CW 攻击**：报告附录的 `cw_attack` 使用 `f_val = clamp(other - real + k, 0)`，
> 这等价于"以真实类别为目标的定向攻击"，最小化它会保护正确分类、无法产生对抗样本
> （实测对抗准确率反而上升）。本工程按 C&W 原论文修正为不定向目标
> `f_val = clamp(real - other + k, 0)`（`attacks/cw.py` 中有详细注释），
> 默认参数 c=10.0, k=0.0, lr=0.01, max_iter=100（`tests/` 与实验 YAML 均已同步）。
> 若需严格复现报告附录行为，可自行改回并在实验中观察结果差异。

## 测试分层

- `pytest -m "smoke"`：小数据、CPU 可跑、CI 必跑（攻击有效性、扰动受限、指标正确性、可复现性）
- `pytest -m "full"`：全量数据/完整迭代，本地或 GPU 手动执行

## 已知口径说明

- 报告正文写基线干净准确率 0.9825，对比表写 0.9885，两处不一致；本工程不预设
  结果，以实际运行输出为准。
- 随机平滑认证：σ=0.25, N=30, 置信度 1−α=0.999，仅认证测试集前 100 个样本（可配置）。

## Roadmap

- [ ] 结果对比脚本：读取 `experiments/results/*.json` 生成横向对比表
- [ ] matplotlib 可视化（`pip install -e ".[viz]"` 后接入）
- [ ] AutoAttack 等更强评估基准（报告 Future Work 提及）
