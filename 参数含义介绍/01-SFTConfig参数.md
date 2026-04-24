# SFTConfig 参数整理

`SFTConfig` 来自 `trl`，本质上可以理解成：

- 在 `transformers.TrainingArguments` 的基础上
- 增加了一些 `SFT`（监督微调）专用参数


## 1. 先用一句话理解整份配置

一套 `SFTConfig`，主要就是在回答这几个问题：

- 训练要跑多久
- 每次喂多少数据
- 用什么优化器和学习率策略
- 怎么省显存、提速度
- 多久记录日志、评估、保存
- 数据怎么截断、打包、算 loss

## 2. 训练规模与训练时长

这组参数决定“训练跑多大、跑多久”。

### `output_dir`

- **类型**：`str | None`
- 输出目录
- 用来保存模型、checkpoint、日志等内容

### `per_device_train_batch_size`

- **类型**：`int`
- 每张卡每一步喂多少条训练样本
- 如果是单卡，就是单步 batch size
- 如果是多卡，总 batch 还要乘设备数

### `num_train_epochs`

- **类型**：`float`
- 训练多少轮
- 一轮表示把整个训练集完整跑一遍

### `max_steps`

- **类型**：`int`
- 最大训练步数
- 如果设置为正数，通常会优先生效
- 常见理解：`num_train_epochs` 控制“按轮数训练”，`max_steps` 控制“按步数训练”

### `gradient_accumulation_steps`

- **类型**：`int`
- 梯度累积步数
- 显存不够时非常常用
- 比如每步只跑 1 条，但累积 8 步再更新一次，相当于有效 batch 变大了

### `max_grad_norm`

- **类型**：`float`
- 梯度裁剪阈值
- 防止梯度爆炸

### `seed`

- **类型**：`int`
- 训练随机种子
- 控制初始化、shuffle 等随机行为

### `resume_from_checkpoint`

- **类型**：`str | None`
- 从指定 checkpoint 恢复训练


### 有效 batch size 的理解

常见公式：

`有效 batch size = per_device_train_batch_size × 设备数 × gradient_accumulation_steps`

例如：

- 单卡
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`

那么有效 batch size 可以理解为 `8`

实际执行过程中，总的step数计算公式为：

`total_steps = （total_trainable_data_nums + 有效batch size - 1） // 有效batch size`

## 3. 学习率与优化器

这组参数决定“模型参数怎么更新”。

### `learning_rate`

- **类型**：`float`
- 初始学习率
- 是最核心的超参数之一

### `lr_scheduler_type`

- **类型**：`SchedulerType | str`
- 学习率调度器类型
- 常见如：
  - `linear`
  - `cosine`
  - `constant`

### `warmup_steps`

- **类型**：`float`
- 预热步数
- 训练刚开始时让学习率从较小值逐渐升上去

### `warmup_ratio`

- **类型**：`float | None`
- 预热比例
- 和 `warmup_steps` 通常二选一

### `optim`

- **类型**：`OptimizerNames | str`
- 优化器类型
- 比如 `adamw_torch_fused`

### `weight_decay`

- **类型**：`float`
- 权重衰减
- 用来抑制过拟合，属于正则化手段

### `adam_beta1` / `adam_beta2` / `adam_epsilon`

- **类型**：`adam_beta1: float，adam_beta2: float，adam_epsilon: float`
- AdamW 这类优化器内部使用的超参数
- 一般保持默认即可

---

## 4. 精度、显存与性能优化

这一组在实际训练里非常重要，尤其是单卡训练。

### `bf16`

- **类型**：`bool | None`
- 是否启用 `bfloat16`
- 如果显卡支持，通常优先于 `fp16`

### `fp16`

- **类型**：`bool`
- 是否启用 `float16`
- 常见于混合精度训练


### `gradient_checkpointing`

- **类型**：`bool`
- 梯度检查点
- 通过“多算一点”来“少占一些显存”
- 大模型训练里很常见

### `use_liger_kernel`

- **类型**：`bool`
- 是否启用 liger kernel
- 可以提升速度或降低显存占用

### `activation_offloading`

- **类型**：`bool`
- 将部分激活转移到别的位置（例如 CPU）
- 用时间换显存

### 单卡训练最值得先关注的几个

- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `bf16` 或 `fp16`
- `gradient_checkpointing`
- `max_length`

---

## 5. 日志相关

这组参数决定“训练过程怎么观察”。

### `logging_strategy`

- **类型**：`IntervalStrategy | str`
- 什么时候记录日志
- 常见是 `steps`

### `logging_steps`

- **类型**：`float`
- 每多少步记录一次日志

### `logging_first_step`

- **类型**：`bool`
- 第一步是否就记录日志

### `log_on_each_node`

- **类型**：`bool`
- 多机训练时，每个节点是否都记录日志


### `report_to`

- **类型**：`None | str | list[str]`
- 日志汇报到哪里
- 常见有：
  - `none`
  - `tensorboard`
  - `wandb`

### `logging_dir`

- **类型**：`str | None`
- 日志目录

## 6. 评估相关

这组参数决定“什么时候验证模型效果”。

### `eval_strategy`

- **类型**：`IntervalStrategy | str`
- 是否评估，以及按什么节奏评估
- 常见：
  - `no`
  - `steps`
  - `epoch`

### `eval_steps`

- **类型**：`float | None`
- 每多少步评估一次

### `eval_delay`

- **类型**：`float`
- 延迟到多少步之后才开始评估

### `per_device_eval_batch_size`

- **类型**：`int`
- 每张卡的评估 batch size

### `prediction_loss_only`

- **类型**：`bool`
- 评估时是否只返回 loss

### `eval_on_start`

- **类型**：`bool`
- 训练开始前先评估一次

## 7. 保存与最佳模型选择

这组参数决定“多久存一次、保留多少、是否加载最佳模型”。

### `save_strategy`

- **类型**：`SaveStrategy | str`
- 保存策略
- 常见：
  - `steps`
  - `epoch`

### `save_steps`

- **类型**：`float`
- 每多少步保存一次

### `save_total_limit`

- **类型**：`int | None`
- 最多保留多少个 checkpoint
- 超过后会删除更早的 checkpoint

### `load_best_model_at_end`

- **类型**：`bool`
- 训练结束后是否自动加载表现最好的 checkpoint

### `metric_for_best_model`

- **类型**：`str | None`
- 用哪个指标判断“最佳模型”

### `greater_is_better`

- **类型**：`bool | None`
- 指标越大越好还是越小越好
- 比如 accuracy 通常越大越好，loss 通常越小越好

## 8. SFT 专属参数

这一组是 `SFTConfig` 相比普通 `TrainingArguments` 更值得重点关注的部分。

### `model_init_kwargs`

- **类型**：`dict[str, Any] | str | None`
- 模型初始化时传入的额外参数
- 可以在此处传入`{"attn_implementation": "kernels-community/flash-attn2"}`，表示训练过程中使用flash attention来加快训练过程，减少显存占用

### `chat_template_path`

- **类型**：`str | None`
- 聊天模板路径
- 用于把多轮对话格式化成模型输入文本

### `dataset_text_field`

- **类型**：`str`
- 数据集中哪一列存放文本
- 默认通常是 `text`
- 使用对话格式数据时，不需要传入该参数
### `max_length`

- **类型**：`int | None`
- 每条样本允许的最大 token 长度
- 超过就需要截断

### `completion_only_loss`

- **类型**：`bool | None`
- 是否只对 completion 部分计算 loss
- 适用于 prompt-completion 格式的数据

### `assistant_only_loss`

- **类型**：`bool`
- 是否只对 assistant 回复部分计算 loss
- 很适合多轮对话 SFT

### `activation_offloading`

- **类型**：`bool`
- 激活是否卸载到其他设备或内存
- 本质是“用时间换显存”

## 9. 最值得优先理解的一小撮参数

如果是做单卡 SFT，不要试图一次记住全部参数。先抓下面这些最关键：

- `output_dir`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `num_train_epochs` 或 `max_steps`
- `learning_rate`
- `lr_scheduler_type`
- `warmup_steps` 或 `warmup_ratio`
- `bf16` 或 `fp16`
- `gradient_checkpointing`
- `logging_steps`
- `save_steps`
- `eval_strategy`
- `max_length`
- `assistant_only_loss` / `completion_only_loss`

如果这几个已经理解得比较清楚，基本就能开始配置一次正常的 SFT 训练了。

## 10. 记忆总框架

### 第一步：训练怎么跑

- `batch size`
- `epoch / steps`
- `gradient accumulation`

### 第二步：参数怎么更新

- `learning rate`
- `optimizer`
- `scheduler`

### 第三步：显存和速度怎么平衡

- `bf16/fp16`
- `gradient checkpointing`
- `max_length`

### 第四步：过程怎么管理

- `logging`
- `eval`
- `save`

### 第五步：数据怎么喂给模型

- `dataset_text_field`
- `chat_template_path`
- `packing`
- `truncation`
- `assistant_only_loss` / `completion_only_loss`

