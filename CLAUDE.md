# FedSchNet-ReorgEnergy — Claude Code 项目说明

> 你正在协助开展一项联邦学习预测分子重组能(reorganization energy)的研究,
> 目标是 SCI 投稿(JCIM 风格)。读完此文件再开始任何工作。
> 关键规则一次性看完:不擅自重新预处理数据 / 不在 GPU 机做 CPU 任务 / 长任务必须 tmux / 改动核心策略前先 plan。

---

## 1. 双机架构(最重要,先看这个)

本项目跨两台同账号(`lengcan`,同密码)的机器协作:

| 角色 | 主机 | 何时用 | 不该做的事 |
|---|---|---|---|
| **GPU box** | `dell-PowerEdge-R940xa`(本机)<br>4× Xeon Platinum 8260 / 2× NVIDIA A30 24GB / CUDA 12.4 | 模型训练、SSL 预训练、联邦聚合、LOOCV、消融实验 | 别在这做 matplotlib 大批渲染、别 pip 装数据预处理依赖 |
| **CPU box** | `10.98.137.100`(SSH 同凭据)<br>双路 Xeon(Silver 4314 + Platinum,含 AVX-512 VNNI/BITALG) | 数据预处理(rdkit 批量构图)、特征工程、绘图(plot_paper_figures.py)、xlsx/docx/pdf 文档生成、经典 ML baseline(XGBoost/RF/GP)、SHAP 分析 | 别尝试 torch.cuda(无 GPU) |

**判定规则(给 Claude Code):**
- 任务涉及 `import torch` + `.cuda()` / `cuda_a` / `cuda_b` → **必须在 GPU box 跑**
- 任务只用 numpy / pandas / sklearn / rdkit / matplotlib / python-docx / openpyxl → **优先去 CPU box**
- 不确定时停下来问用户,**不要默认在当前机跑**

**切换到 CPU box 的标准做法:**
```bash
# 在 GPU box 上发起
ssh lengcan@10.98.137.100
cd ~/<项目同步路径>     # 见 §3 路径同步
tmux new -s cpubox
```

> ⚠️ 用户的家目录是否 NFS 共享于两机之间未确认。**第一次切换前必须 `ls ~` 双边对比**,
> 若不共享则用 `git push` + `git pull` 同步代码,数据用 `rsync -avz` 单向同步。

---

## 2. 项目核心事实

**研究目标:** 用联邦学习把 Client C(TADF 私有,49–53 分子)的重组能预测 R² 从负值推到 ≥ 0.4。

**四个客户端:**
| Client | 数据 | 规模 | 标签 | 分配 GPU |
|---|---|---|---|---|
| A | QM9 公开,非芳香 | ~6k | reorg_energy_eV | cuda:0 |
| B | QM9 公开,≥1 芳香环 | ~9k | reorg_energy_eV | cuda:1 |
| C | TADF 私有 | 49(triplet)/ 53(hole) | lambda_hole_eV / lambda_T_total_eV | cuda:1 |
| D | Atahan-Evrenk 2019 | 5,876 | reorg_eV(hole) | cuda:0 |

**当前最好成绩(README,务必更新):**
- Client A/B 5-fold:E14 FedPer GIN+KAN, MAE_A=0.1425 (R²=0.777), MAE_B=0.0780 (R²=0.865)
- Client C LOOCV:E52 SchNet+KAN+FedPer, MAE=0.2877, **R²=0.139** ← 主要瓶颈
- Client D LOOCV:E51 SchNet+KAN+FedAvg, R²≈0.181

**核心组件:**
- 编码器:GIN / SchNet(`src/models.py`)
- 回归头:KAN / MLP / Physics-KAN(`src/models.py`)
- 联邦策略:FedAvg / FedPer / FedBN / FedProx(`src/federated.py`)
- SSL:AtomMask / EdgePred / GraphCL(`src/ssl_pretrain.py`)
- 训练入口:`experiments/run_all.py`(实验由 `experiments/configs.py` 注册)

---

## 3. 路径与环境

```
GPU box 项目根:  ~/cleng/Function_calling/test/0-ground_state_structures/0503/reorganization/FedSchNet-ReorgEnergy
CPU box 项目根:  <待用户填写,首次同步时记录>
Conda 环境:      <待用户填写,例如 fedreorg>;两机环境名应一致
数据:           ./data/{client_a_b, client_c, client_d}/  已就绪,不要重新下载或预处理 SMILES
checkpoints:    ./checkpoints/(SSL 预训练)
results:        ./results/{tables, figures, json}/
```

**第一次会话开始时 Claude 必须做的事:**
1. `pwd && hostname` 确认在哪台机
2. `nvidia-smi` 或 `lscpu | head -5` 确认硬件可见
3. `git status && git log --oneline -5` 确认代码版本
4. **不要**自动 `pip install` 任何东西。需要新依赖时先列出来给用户确认

---

## 4. 训练 / 评估协议(改实验前必看)

- **Client A/B/D**:5-fold CV(KFold,seed 固定)
- **Client C**:LOOCV(53/49 折)。**每折独立标准化**(`norm_stats` 参数),预测后反标准化
- 报告三个指标:MAE / RMSE / R²;Client C 必须报 R²
- 数据增强(物理特征):节点级广播 HOMO/LUMO/gap/dipole(4 维),Client A/B/D 用零填充 → in_dim 从 11 变 15
- SSL 预训练 → 下游微调时,GINConv 第一层权重必须用 `_adapt_encoder_state_dict` 零填充适配维度

**联邦设备分配(代码已硬编码,不要改):**
```python
device_a = torch.device('cuda:0')   # Client A / D
device_b = torch.device('cuda:1')   # Client B / C
# 聚合在 CPU
```

**FedPer / FedBN 排除规则**(`_get_exclude_keys`):key 中包含 `head` / `bn` / `norm` 的张量不参与聚合。新增策略时遵守同样接口。

---

## 5. 长任务执行规范(强制)

任何预计 > 5 分钟的任务都必须:

```bash
# GPU 训练
tmux new -s e60                                    # 用实验编号命名
nohup python experiments/run_all.py --exp E60 \
    > logs/E60_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $!                                            # 记录 PID
# Ctrl-b d 脱离

# 监控
tail -f logs/E60_*.log
nvidia-smi -l 5                                    # 5 秒刷新一次
```

**Claude Code 不可前台阻塞执行长训练**。正确流程:启动 → 立即返回 → 让用户 attach tmux 看进度。后台启动后只需告诉用户"已在 tmux:e60 启动,日志 logs/E60_xxx.log"。

---

## 6. 代码风格约束

- 新增联邦策略:加到 `src/federated.py`,通过 `get_aggregation_fn` 暴露,**不**绕过该接口
- 新增 SSL 方法:加到 `src/ssl_pretrain.py`,实现 `compute_loss(data) -> Tensor` 和 `parameters() -> list`
- 新增实验:在 `experiments/configs.py` 注册 `E编号`,然后 `run_all.py --exp E编号` 能直接跑
- Docstring 风格:NumPy 风格,英文(全仓库已统一)
- 不引入新的随机源:固定 seed=42(数据划分)、torch.manual_seed 在每个实验开头
- 改 `src/data_utils.py` 中 SMILES → Graph 的逻辑要极其谨慎,会重置所有已有结果

---

## 7. 已知坑(踩过的)

1. SSL 预训练 `in_dim=11`,物理特征增强后 `in_dim=15` —— 加载预训练权重必须零填充第一层
2. KAN 头的 `grid_size` 改了之后 state_dict shape 变,断点续训会失败
3. Client C LOOCV 单折 batch=1,梯度噪声大;不要把 `batch_size_c` 改大(数据量根本不够)
4. `get_aggregation_fn('fedbn')` 与 `'fedper'` 当前排除集相同(BN 已被 fedper 排除),修改时两者一起改
5. `plot_paper_figures.py` 用了相对路径读 `results/json/*.json`,从其他目录运行会找不到文件

---

## 8. 当前研究优先级

按价值降序:

1. **Client C R² 攻坚**(最高优先级)
   - 候选:FedRep / Ditto / pFedMe(个性化联邦)、Client D → C 知识蒸馏、meta-learning(MAML 把 A/B/D 当 meta-train)
   - 目标:R² ≥ 0.4
2. **新增 baseline**(投稿必备):Morgan FP + RF/XGBoost/GP、D-MPNN、ChemProp
3. **消融完善**:KAN grid 已做,补做 SSL 方法 × 联邦策略 的 3×4 网格
4. **可解释性**:SHAP / Integrated Gradients,关联 HOMO/LUMO 与 hole-λ 的 Marcus 理论
5. **论文图表 + 正文**:见 §10

---

## 9. GPT-MCP 交叉审核工作流

用户配置了 GPT MCP server,**用作独立第二意见**。触发时机:

- 改了 `src/federated.py` 或 `src/train_eval.py` 核心逻辑 → 让 GPT review diff
- 写完一段论文(Method/Results)→ 让 GPT 挑技术错误
- 设计新实验前 → 让 GPT 估算可行性 + 找潜在 confound

**调用模板(由 Claude Code 主动发起):**
```
# 改完 federated.py 后
"我刚在 src/federated.py 加了 FedRep 实现(diff 见下),
请你以审稿人视角检查:(1) 与已有 fedper/fedbn 接口是否一致;
(2) 客户端个性化层的梯度更新是否正确;(3) 是否会破坏 _get_exclude_keys 共用逻辑。
列出最多 5 个具体问题,按严重性排序。"
```

**重要:Claude Code 不应盲信 GPT 反馈**。每条反馈要么自己验证(跑测试),要么明确告诉用户"GPT 提出 X 但我不同意,因为...",由用户裁决。

---

## 10. 论文写作流程(skills 驱动)

**前置:首次会话执行**
```bash
/plugin install document-skills@anthropic-agent-skills
```
这会装上 docx / pdf / pptx / xlsx 四个官方 skill。

### 流水线

| 阶段 | 工具 / Skill | 在哪台机 | 产出 |
|---|---|---|---|
| ① 跑实验出数据 | run_all.py | GPU box | results/json/*.json + tables/*.csv |
| ② 渲染论文图 | plot_paper_figures.py(matplotlib) | CPU box | results/figures/fig*.{pdf,png} |
| ③ 制作正文表 | xlsx skill + pandas | CPU box | tables/main_results.xlsx, supp_*.xlsx |
| ④ 撰写正文 | docx skill | CPU box | manuscript.docx(JCIM 模板) |
| ⑤ 投稿 PDF 校对 | pdf skill | CPU box | manuscript_review.pdf 标注 |
| ⑥ 答辩 / 组会 slides | pptx skill | CPU box | defense.pptx |
| ⑦ 补充材料网页 | frontend-design skill | CPU box | supp_website/index.html |

### 推荐建立的项目级自定义 skills

放在 `.claude/skills/` 下,版本控制随仓库走:

```
.claude/skills/
├── run-experiment/SKILL.md      # 封装"注册 E 编号 → 跑 → 写入 results"全流程
├── update-results-table/SKILL.md # 自动把新实验追加到 all_experiments_summary.csv
├── jcim-figure/SKILL.md         # JCIM 配色 / 字号 / 双栏宽度规范(从 plot_paper_figures.py 抽出)
├── manuscript-section/SKILL.md  # 写一个论文章节的标准流程(数据→大纲→草稿→自审)
└── reviewer-response/SKILL.md   # 审稿意见逐点回复的模板
```

**`jcim-figure/SKILL.md` 骨架示例:**
```markdown
---
name: jcim-figure
description: Generate publication-quality figures matching JCIM style guide (single column 3.33", double column 7", DPI 600, sans-serif Arial 8pt, V_BLUE/V_GREEN/V_RED palette). Use when creating any figure for the FedSchNet-ReorgEnergy paper.
---

# JCIM Figure Standards

## Constraints
- DPI = 600 for raster, prefer vector (PDF) when possible
- Font: Arial / Helvetica, 8pt body, 9pt axis, 10pt panel labels
- Colors: see `plot_paper_figures.py` for V_BLUE / V_GREEN / V_RED hex codes; never invent new colors
- Error bars: capsize=2, linewidth=0.8
- Panel labels (a)/(b)/(c) in bold with white STROKE outline

## Workflow
1. Read `plot_paper_figures.py` for existing patterns
2. Add new figure as `plot_figN_<topic>()` function in same file
3. Save via `savefig(fig, 'figN_topic')` (handles PDF + PNG)
4. Update `main()` to include the new function
```

### 论文章节写作三段式(Claude Code 应遵守)

每写一个章节(Method / Results / Discussion)按这个顺序:

1. **检索现有素材** —— 用 conversation_search 翻看过往讨论,读 results/tables/*.csv 拿数字
2. **写 200 字大纲** —— 列要点,等用户确认后再展开
3. **生成 docx**(用 docx skill,不要直接写 markdown 然后转换),嵌入图表引用

---

## 11. 不要做的事(总结清单)

- ❌ 在 GPU box 跑批量绘图、文档生成
- ❌ 不开 tmux 直接前台跑 LOOCV(53 折要几小时)
- ❌ 重新做 SMILES → Graph 预处理(已缓存,会破坏复现性)
- ❌ 修改 `device_a=cuda:0 / device_b=cuda:1` 硬编码
- ❌ 在 `_get_exclude_keys` 里加 'head'/'bn'/'norm' 之外的关键字而不在 federated.py 顶部留 comment
- ❌ 不经用户确认就 `pip install` 新包
- ❌ 不经用户确认就 `git push`、删 checkpoints、改 results 已存在的文件
- ❌ 引用 GPT MCP 的反馈时不验证就执行

---

## 12. 常用命令速查

```bash
# 数据健康检查
python -c "from src.data_utils import prepare_all_clients; d=prepare_all_clients('hole', include_d=True); print({k: len(v) for k,v in d.items() if isinstance(v,list)})"

# 单实验
python experiments/run_all.py --exp E14

# 一组实验
python experiments/run_all.py --batch2

# 消融
python experiments/ablation.py --all

# 重生成所有论文图(在 CPU box)
python plot_paper_figures.py

# 查看实验汇总
column -t -s, results/tables/all_experiments_summary.csv | less -S
```

---

## 13. 会话开头 checklist(Claude Code 自检)

新会话第一条 user message 后,在动手前确认:

- [ ] 我在哪台机?(`hostname`)
- [ ] 任务该不该在这台机做?(对照 §1 表格)
- [ ] 是否需要 plan 模式?(改核心代码 / 设计新实验 → 必须先 plan)
- [ ] 是否长任务?(→ 准备 tmux + nohup 模板)
- [ ] 是否需要 GPT MCP 二审?(改了 federated/train_eval/SSL 核心 → 是)

不确定时,**直接问用户**,不要猜。
