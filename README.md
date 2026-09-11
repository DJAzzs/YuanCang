# 远苍 YuanCang - 数学推理大模型训练项目

## 项目简介
远苍是一个专注于数学推理与逻辑分析的AI助手，基于Qwen2.5-Math-7B全量微调训练。

## GPU配置
- 双卡 NVIDIA RTX PRO 6000 Blackwell MAX-Q (总显存 192GB)
- 算力: ~3.5 PFLOPS NVFP4 稠密模式（不可用时自动回退 BF16）
- CUDA: 13.0

## 训练策略
### 精度 / FSDP
- **NVFP4 稠密 QAT**（torchao `NVFP4FakeQuantizedLinear`），若硬件/框架不支持自动回退 BF16
- FlashAttention-2（不可用回退 sdpa）
- FSDP2 (full_shard, optimizer+gradient sharding)，通信重叠，activation checkpointing（省~30%激活显存）
- 启动前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 超参数
| 项 | 值 |
|---|---|
| Epochs | 1 (早停：验证loss连续3次不降则终止) |
| Global batch | 32 = per_device4 × grad_accum4 × 2卡 |
| Max seq len | 8192 |
| Peak LR | 1.5e-5 |
| Scheduler | Cosine with Warmup → min_lr 1.5e-7 (峰值1%) |
| Warmup ratio | 10% (Trainer按 len(train_dataloader) 动态计算) |
| Optimizer | AdamW betas=0.9,0.999 eps=1e-8 |
| Weight decay | 0.01 |
| Grad clip | 1.0 |

### 数据
- ChatML `<|im_start|>`; Loss masking: system+user=-100，仅 assistant 参与
- 验证集: `Dataset/验证集/*.parquet`（low/high/top, 375条）
- 验证频率每2000步；存档每200步保留最近3个；seed=42

## 使用流程

### 1. (可选)数据预处理 - 生成训练分片
```bash
cd /home/dja/桌面/远苍
source venv/bin/activate
# 先小样校验 schema（不落盘）
python prepare_data.py --dry
# 全量流式生成 train_part_*.jsonl + eval.jsonl (防爆内存)
nohup python prepare_data.py > /tmp/prep.log 2>&1 &
tail -f /tmp/prep.log
```

### 2. 单卡冒烟测试 (BF16, 验证全流程)
```bash
# 先手动清理显存！两块 GPU 需空闲
cd /home/dja/桌面/远苍
source venv/bin/activate
nvidia-smi   # 确认两卡都空
bash launch.sh            # = torchrun --nproc_per_node=1 train.py --mode test --data train_100.jsonl
tail -f /tmp/train_smoke.log
```

### 3. 双卡正式训练 (NVFP4 QAT，自动回退BF16)
```bash
cd /home/dja/桌面/远苍
source venv/bin/activate
nvidia-smi   # 确认两卡都空
bash launch_full_sft.sh             # NVFP4, global batch32, seq8192
# 或强制 BF16:
bash launch_full_sft.sh --quant bf16
tail -f /tmp/train_full.log
```

## 文件结构
- `prepare_data.py`: 数据预处理（全流式，seed=42可复现）
- `train.py`: BF16 FSDP SFT 主训练脚本
- `train_nvfp4.py`: NVFP4 QAT 正式训练（复用 train.py 参数/数据集逻辑 + BF16保底）
- `launch.sh`: 单卡冒烟启动
- `launch_full_sft.sh`: 双卡正式启动
- `eval.py`: 分层评估 (L1留出集/L2 GSM8K+MATH/L3全checkpoint)

# ================= SOTA 冲击路线图 =================
目标门槛: GSM8K 90%+ / MATH-500 80%+ / AIME 2024&2025 25%+ → 后续RL冲击50%+

Phase 1 - SFT基座(当前, ~4天双卡)
  数据: Math2(全量) + Math1(30%) + Nemotron-SFT-V4(全量) + Numina-CoT(全量)
        + high(acc<0.3) + medium(acc<0.5) + Math/* + math1/* + identity + 通用闲散~2%
  [已排除 OpenMathReasoning: 其parquet仅含expected_answer无CoT, 会教模型跳过推理]
  产出: saves/yuancang-full-sft

Phase 2 - 三层评测定位(随训进行)
  L1 每200步: 留出验证集loss/acc  → 早停依据(连续3次不降)
  L2 每2000步: GSM8K全量 + MATH-500抽100
  L3 训后: 全部checkpoint跑完整GSM8K/MATH-500/AIME24&25, 选最佳

Phase 3 - 难例增强第二轮(~1-2天)
  从L3失败样例 + Dataset高难度档(top/high, acc<0.15)构造增强集
  对最佳checkpoint做低LR(5e-6)继续SFT 1 epoch

Phase 4 - RL冲真SOTA(AIME 50%+ 的关键, GRPO路线)
  题库: dapo-math-17k(prompt+ground_truth) + 验证集 + Math1/Numina题面
  奖励: 答案可验证(\boxed{}匹配ground_truth) + 格式奖励
  参考: DeepScaleR / DeepSeekMath-GRPO 配方, 1.5e-6, batch 512 prompts
  预算: 2卡约5-8天

Phase 5 - 终评发布
  全量GSM8K/MATH-500/AIME24/AIME25 + Chat格式人工抽检 → 开源发布

# ================= 评测脚本用法 (eval.py) =================
# 环境变量: HF_ENDPOINT=https://hf-mirror.com (必需,直连HF会超时)
# 数据源已验证: GSM8K 1319 / MATH-500 500 / AIME2024 90(去重后) / AIME2025 30

# L2 (训练中每2000步, 或checkpoint间隙; 需~15GB空闲显存):
HF_ENDPOINT=https://hf-mirror.com python3 eval.py \
  --checkpoint saves/yuancang-full-sft/checkpoint-2000 --level 2 --batch-size 16

# L3 (训后全量选最佳; 双卡并行开两个终端各分一半):
HF_ENDPOINT=https://hf-mirror.com python3 eval.py --level 3 \
  --all-checkpoints saves/yuancang-full-sft --shard 0 --num-shards 2 --gpu 0 &
HF_ENDPOINT=https://hf-mirror.com python3 eval.py --level 3 \
  --all-checkpoints saves/yuancang-full-sft --shard 1 --num-shards 2 --gpu 1 &

# 结果JSON: eval_results/<checkpoint>_L<级别>_s<分片>.json

# ================= 验证集说明(更正) =================
# 验证集完整四档各125题共500题: low/medium/high/top
# medium.parquet已于Aug20补齐内容(此前0字节为旧状态)
# 500题全部进入eval.jsonl作为L1评测集(每200步), 与train.jsonl零泄漏(已验证0/126339重叠)
# 用途: L1实时监控 -> Phase2 GRPO分层pass@1曲线(low/high/top) -> Phase3难例增强种子

# ================= 最终路线图 v2 (含基模决策) =================
SFT(进行中,~1.9天) → RFT自蒸馏二次微调 → GRPO RL → MCTS+BoN最终跑分
                     ↑_____________↓
                    MCTS轨迹+难题蒸馏回流

决策矩阵:
  [step 4000复测] MATH-500 >= 55% → U谷确认, 现配方继续
                  MATH-500 <  45% → 配方问题: MATH风格语料加权2倍, 重训(用resume+新数据分片)
  [SFT完成L3]     全面达标       → 按流程走, 基模不换
                  MATH/AIME不达标 → 评估换基模: MiMoV2Flash已被tfs5.15支持(实测通过)
                                     MiMo-7B初代支持需实测; 换基模=管线重验(tokenizer/TRL), 1-2天成本
RL门槛: 策略在RL题库上pass率须20~70%才有密集奖励信号
        (ckpt-2000的AIME 2%太低 → RL题库选中难题, GSM8K/中MATH为主)

# ================= 教师决策(锁定) =================
# 外部教师 = Qwen3.8-27B-NVFP4 (本地/桌面/Models/, 22GB混合量化, Qwen3_5架构)
#           transformers 5.15.0原生支持Qwen3_5 ✓ (AutoConfig已验证)
# QwQ-32B放弃(用户决策)。R1-Distill-32B备选未启用。
#
# ⚠ 教师资质门禁(使用前必测): 教师必须在目标能力上显著强于学生
#   测试命令(5000步对话暂停窗口内, 双卡均空闲时):
#   HF_ENDPOINT=https://hf-mirror.com python3 eval.py \
#     --checkpoint /home/dja/桌面/Models/Qwen3.8-27B-NVFP4 \
#     --level 2 --quick 300 --batch-size 16 --gpu 0
#   判定: GSM8K/MATH-500零样本 >= 远苍SFT最终checkpoint → 教师合格
#         否则 → 教师无意义, 退回纯自蒸馏+GRPO路线
#
# ⚠ NVFP4量化推理加载未实测(torchao stable版兼容性风险), 首次加载失败
#   则fallback: 自蒸馏+GRPO, 或下载bf16版教师
#
# 蒸馏策略(锁定): 两轮制
#   第一轮: RFT普通蒸馏(全题库, K=8采样验证) → 微调v2
#   第二轮: MCTS轨迹蒸馏(第一轮失败集+top难题, 本地Qwen3.8出题解) → 微调v3
#   混合比例: 外教:自蒸馏 ≈ 3:7起步, 每轮L2/L3实测调整
#   GRPO在蒸馏轮之后(策略最强时推上限)

# ================= Phase 2.5: SQLM自提问数据引擎 (已落地) =================
# 脚本: sqlm_gen.py (arXiv 2508.03682 务实版: 拒绝采样近似RL双循环)
# 机制: 提问者(远苍按15个数学主题出新题) + 解题者(每题G=8采样)
#       + SQLM难度门控(pass_rate 0.125~0.875, 太简单/太难均剔除)
#       + 多数投票答案中取PRM最优一致解作为SFT目标
# 产物: sqlm_data.jsonl (并入RFT/MCTS二次微调语料)
# 定位: 解决dapo题库有限问题, 且题目难度自动瞄准远苍"半会不会"区(学习信号最密)
# 运行时机: SFT完成后, 与RFT/MCTS蒸馏共用GPU窗口
#   python3 sqlm_gen.py --rounds 3 --per-topic 12 --g 8 --gpu 0

# ================= 错误驱动定向出题闭环 (已落地) =================
# 你的提案: 跑分统计失分考点 → SQLM定向多出薄弱类型题
#
# 闭环链路:
#   L3评测(eval.py, 自动携带逐题明细: 考点/难度/对错/预测/真值)
#     → weakness_report.py <结果JSON>
#     → 考点错误率排名 + weights_for_sqlm.json (薄弱主题加权)
#     → sqlm_gen.py --weights weights_for_sqlm.json
#     → 薄弱考点定向出题(出题量∝失分占比×4) → 解题验证 → 二次微调语料
#
# MATH-500考点映射: Algebra/IntermediateAlgebra/Prealgebra/Geometry/
#   NumberTheory/Counting&Probability/Precalculus → 15个SQLM主题

## License

Apache License 2.0 · Copyright (c) 2025 DJAzzs
