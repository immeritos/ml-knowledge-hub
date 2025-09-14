# Batch Size 对梯度估计的影响 — 知识点提纲

## 1. 基本概念
- **Batch Size**：每次前向传播与反向传播中使用的样本数。
- **梯度估计**：在 SGD 中，用 mini-batch 近似全数据集的真实梯度。

## 2. 梯度估计原理
- 全量梯度：
  \[ g = \nabla_\theta L(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \ell(x_i, \theta) \]
- Mini-batch 梯度估计：
  \[ \hat{g} = \frac{1}{B} \sum_{i=1}^B \nabla_\theta \ell(x_i, \theta) \]
- 期望：\( \mathbb{E}[\hat{g}] = g \)（无偏估计）。
- 方差：\( Var(\hat{g}) \propto \frac{1}{B} \)，Batch 越大，方差越小。

## 3. Batch Size 大小的影响
- **小 Batch**：
  - 梯度估计噪声大，更新方向抖动。
  - 噪声有助于跳出局部最优，提高泛化。
  - 内存占用小，更新频繁。
- **大 Batch**：
  - 梯度估计更精确，更新稳定。
  - 可能陷入 sharp minima，泛化能力差。
  - 占用显存大，单次迭代慢。
- **全量 Batch**：
  - 梯度最精确，但效率低，泛化通常最差。

## 4. 对比总结
| Batch Size 大小 | 梯度估计方差 | 泛化性能 | 收敛速度 | 内存需求 |
|----------------|--------------|----------|----------|----------|
| 小 Batch       | 高           | 较好     | 慢       | 低       |
| 大 Batch       | 低           | 较差     | 稳定     | 高       |
| 全量 Batch     | 0            | 最差     | 最慢     | 最高     |

## 5. 实践建议
- 常用范围：32 ~ 256。
- 大 Batch 时需配合：
  - 学习率调度（如 warmup, cosine decay）。
  - 正则化手段（如 Dropout, 数据增强）。
- 在硬件资源允许下，中等 Batch 是折中选择。

