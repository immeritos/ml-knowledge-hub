# 📑 神经网络 & 深度学习

## 1. 激活函数与初始化

### 激活函数常见选择
- **Sigmoid**
  - 输出范围 $(0,1)$，易梯度消失。  
  - 常用初始化：**Xavier / LeCun**。  

- **Tanh**
  - 输出范围 $(-1,1)$，零均值，梯度稳定性优于 Sigmoid。  
  - 常用初始化：**Xavier**。  

- **ReLU**
  - 非线性强，计算高效，避免梯度消失。  
  - 常用初始化：**He (Kaiming)**。  

- **Leaky ReLU / ELU / GELU**
  - 改进 ReLU，负半轴不过度截断。  
  - 常用初始化：**He (带修正系数)**。  

- **Softmax**
  - 用于输出层，概率分布形式。  
  - 前一层通常使用与隐藏层一致的初始化（Xavier/He）。  

---

## 2. Batch Normalization (BN) vs Layer Normalization (LN)

### Batch Normalization (BN)
- **归一化维度**：按批次，对同一层的每个特征维度做归一化。  
- **优点**：加速收敛，抑制梯度消失。  
- **缺点**：对 batch size 敏感，小 batch 效果差。  
- **应用场景**：CNN 训练常用。  

### Layer Normalization (LN)
- **归一化维度**：按样本，对该样本的所有神经元做归一化。  
- **优点**：与 batch size 无关，适合 RNN/Transformer。  
- **缺点**：在 CNN 中效果通常不如 BN。  
- **应用场景**：RNN、Transformer。  

---

## 3. CNN 感受野 (Receptive Field)

### 定义
- 高层神经元在输入图像中覆盖的区域大小。  
- 表示某层特征“能看到”多少输入信息。  

### 计算公式
- 递推关系：  
  ```math
  R_l = R_{l-1} + (k_l - 1) J_{l-1}, \quad J_l = J_{l-1} \cdot s_l
  ```
  - R: 感受野大小  
  - J: 跳跃步长  
  - k: 卷积核大小  
  - s: 步幅  

### 特性
- 层数越深，感受野越大。  
- stride > 1 会让感受野增大更快。  
- dilation 会进一步扩大感受野。  
- padding 只影响边界覆盖，不改变感受野大小。  

### 应用
- 图像分类：需覆盖全图。  
- 检测/分割：需知道某层特征对应输入的区域大小。  

---

## 4. 一句话总结
- **激活函数 ↔ 初始化**：Sigmoid/Tanh → Xavier，ReLU 系 → He。  
- **BN vs LN**：BN 依赖 batch，适合 CNN；LN 与 batch 无关，适合 RNN/Transformer。  
- **CNN 感受野**：逐层叠加扩大，高层特征“看到”的输入范围更大。  
