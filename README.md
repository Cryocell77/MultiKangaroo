# 🦘 Kangaroo Multi-Layer Cascaded Early Exit Framework

<div align="center">
  <img src="model.jpg" alt="Multi-Layer Kangaroo Architecture" width="800">
  <p><em>多层级联早退机制架构图</em></p>
</div>

## 📋 模型架构概述

本项目基于论文《Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting》的框架基础上，实现了创新的**多层级联早退机制**，通过智能的层间隐藏状态传递和置信度阈值控制，显著提升了推理效率和输出质量。

## 🏗️ 核心架构设计

### 1. 基础模型结构
- **基础模型（Base Model）**: LLaMA-7B的前2-4层作为轻量级草稿模型
- **目标模型（Target Model）**: 完整的LLaMA-7B模型用于最终验证
- **Adapter模块**: 轻量级适配器桥接子网络与完整模型的表示能力差距

### 2. 多层级联早退流程

#### 🔄 **第一阶段：Layer 2 推理**
```
输入序列 → Base Model (Layer 1-2) → Hidden State₂ → Adapter₂ → Token₂ + Confidence₂
```
- **置信度阈值**: 0.6
- **决策逻辑**: 
  - 若 `Confidence₂ ≥ 0.6` → 接受Token₂，进入验证阶段
  - 若 `Confidence₂ < 0.6` → 隐藏状态传递至Layer 3

#### 🔄 **第二阶段：Layer 3 推理**
```
Hidden State₂ → Base Model (Layer 3) → Hidden State₃ → Adapter₃ → Token₃ + Confidence₃
```
- **置信度阈值**: 0.55
- **决策逻辑**:
  - 若 `Confidence₃ ≥ 0.55` → 接受Token₃，进入验证阶段
  - 若 `Confidence₃ < 0.55` → 隐藏状态传递至Layer 4

#### 🔄 **第三阶段：Layer 4 推理**
```
Hidden State₃ → Base Model (Layer 4) → Hidden State₄ → Adapter₄ → Token₄ + Confidence₄
```
- **置信度阈值**: 0.5
- **决策逻辑**:
  - 若 `Confidence₄ ≥ 0.5` → 接受Token₄，进入验证阶段
  - 若 `Confidence₄ < 0.5` → 隐藏状态直接传递至完整模型后续层

#### ✅ **验证阶段：目标模型验证**
```
Accepted Token + Hidden State → Full Model (All Layers) → Verification Result
```
- 使用完整的LLaMA-7B模型对接受的token进行验证
- 确保输出质量与原始模型保持一致

## 🧠 关键技术特性

### 1. **智能隐藏状态传递**
- **层间连续性**: 每层的隐藏状态无缝传递至下一层
- **信息保持**: 确保语义信息在层间传递过程中不丢失
- **动态路由**: 根据置信度动态决定计算路径

### 2. **渐进置信度阈值**
- **Layer 2**: 0.6（最严格，优先早退）
- **Layer 3**: 0.55（中等严格）  
- **Layer 4**: 0.5（相对宽松，最后机会）
- **设计原理**: 越早的层要求越高的置信度，确保早退质量

### 3. **统一KV-Cache管理**
- **共享机制**: 所有adapter层共用一组KV-cache
- **回滚策略**: 若生成的token不被验证接受，自动回滚KV-cache状态
- **内存效率**: 避免多套KV-cache带来的内存开销

### 4. **自然早退机制**
- **Origin风格终止**: 当所有早退层都无法满足置信度要求时，自然终止投机解码
- **无人工限制**: 不依赖硬编码的序列长度限制
- **动态适应**: 根据模型自身的置信度自适应调整生成策略
