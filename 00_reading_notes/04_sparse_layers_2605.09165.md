# 04 · Sparse Layers are Critical to Scaling Looped Language Models（arXiv 2605.09165 v2）

> 读法说明：本笔记依据 v2（2026-06-30）的 HTML 与 PDF 全文（含附录）；表/图编号 HTML 与 PDF 一致（Table 1–7，Figure 1–10）。标注"图读"者为只能从图中读出的值；标注"数字化"者为本人从 arXiv HTML 的原始 PNG 数字化得到的近似值（方法与误差见文末注）。

## 基本信息
- 作者与机构：Ryan Lee（USC Information Sciences Institute；通讯 ryantlee@usc.edu）、Jacob Biloki（Netflix）、Edward J. Hu（Independent Researcher）、Jonathan May（USC ISI）。
- 日期/版本：v1 2026-05-09，v2 2026-06-30（本笔记用 v2；两版摘要相同）；cs.LG + cs.CL；CC BY 4.0；PDF 15 页（正文 9 页 + 参考文献 + Appendix A）。
- 发表状态：预印本，未标注会议；致谢 DARPA Agreement No. HR00112590089、NAIRR Pilot、TACC、Jetstream2（NSF-OAC 2005506）。
- 代码与权重：正文与附录**无任何 GitHub/HuggingFace 链接**。
- 算力（Appendix A）：NVIDIA H100 80GB；isoFLOP 扫描 ≈1,200 GPU-h、μP 验证 ≈200、early-exit/loop-depth ≈300、分析 ≈100，合计 ≈2,000 GPU-h；训练数据 <2TB。

## 他们匹配了什么
- **匹配轴**：训练算力 C = 6·N_active·D（isoFLOP）。四种架构在同一宽度下 N_active 相同（"By design, all four architectures have the same N_active parameters (neglecting the small router weights) and therefore the same compute budget at matched token counts."），有效深度固定 16 层，故同预算同宽度下 token 数 D 也相同。**不匹配**存储参数 N_unique（Looped 的非嵌入部分约为 Base 的一半；Looped-MoE、MoE 更多，Table 3/5）。KV cache 原文未讨论；同宽度、同 16 个执行注意力层、MHSA，按构造相等（SMELT Table 1 亦将其标为 ✓）。
- **参数量定义**（§3.2）：N_active = "the total parameters used in a single forward pass, counting repeated (looped) layers at each invocation and counting only the k active experts in MoE layers"，并且**包含 embedding 与 unembedding**："We include embedding and unembedding parameters in our N_active count for compute calculations, an additional (2·V·d_model) parameters". N_unique = 存储参数。Table 3（非嵌入部分；L 为有效层数，A/F 为每层注意力/FFN 参数，F_expert = F/k）：Looped N_active = L(A+F) > N_unique = (L/R)(A+F)；Base 二者相等 = L(A+F)；Looped-MoE N_active = L(A+F) < N_unique = (L/R)(A + E·F_expert)；MoE N_active = L(A+F) ≪ N_unique = L(A + E·F_expert)。
- **FLOP 公式**：C = 6ND，N = N_active（§3.2，引 Kaplan/Hoffmann）；**无注意力项、无序列长度项**。
- **批评原话**：§1："Re-using parameters through compute depth … is an appealing alternative, however such architectures have been found to under-perform baselines when compared on equal training compute [Kaplan et al.]." §7 评 Ouro："they loop dense FFNs, which we have shown are sub-optimal when truly compared to dense baselines on fixed compute. In their study, they compare performance by the number of unique parameters, but giving their models much more compute (4x the effective depth) than the models in their comparisons. In contrast, we conduct a comparative study in the isoFLOP setting: understanding instead when compute is fixed, which model architectures scale better."

## 模型与训练设定
- **骨干**（§2，Figure 8）：decoder-only；causal MHSA + RoPE；SwiGLU FFN；pre-RMSNorm（"RMS Norm without a learnable gain"）；标准残差流。四种配置（Table 1）：Looped（dense FFN，8×2）、Base（dense，16）、Looped-MoE（sparse，8×2）、MoE（sparse，16）。
- **循环方式**（§2.1，Figure 1 左）：L=8 个独立层作为整体重复 R=2 次，有效深度 16（token embeds → loop 1 → loop 2 → to vocab）。**无 prelude/coda、无输入注入、无残差缩放、无循环专用归一化**；初始状态即 embedding 输出。
- **MoE**（§2.2）：token-choice top-k；**k=2 / E=8**（"following Mixtral"）；expert 是更小的 SwiGLU（F_expert = F/k，故激活 FFN 参数与 dense 相同）；辅助损失 load balancing L_LB = E·Σᵢ fᵢ·p̄ᵢ 与 router z-loss L_RZ = (1/B)·Σⱼ(log Σᵢ exp h(xⱼ)ᵢ)²；损失系数未给。
- **μP**（§3.1，Table 2）：d_base = 128；除 embedding 外所有矩阵 init 方差 σ²_base/w_ratio、LR η_base/w_ratio（w_ratio = d_model/d_base），并扩展到 W_unembed、W_router、W_experts，去掉原始 μP 的输入/输出乘子；循环层不做修改："weight-tied layers receive gradient contributions from multiple positions but their scale is unchanged"。验证：4 层有效深度（Base 4、Looped 2×2、MoE 4、Looped-MoE 2×2）、d ∈ {128, 256, 512, 1024}、LR 扫 10⁻³–10⁻¹（Figure 2，图读），最优区带 ≈1×10⁻²–2×10⁻²（图读；**正文未给峰值 LR 数值**），各架构最优 LR 处损失变化 0.8%/0.2%/0.5%/0.3%（图内标注），"at most 0.8% loss difference"。
- **训练**（§4.1）：FineWeb-Edu 的 10B-token 样本；GPT-2 tokenizer，V = 50,257；AdamW β₁=0.9、β₂=0.999、ε=10⁻⁸、independent weight decay 1.0×10⁻⁴；WSD，sqrt-decay cooldown 至峰值的 5%，占最后 10% 步；峰值 LR 由 d_base=128 代理模型 μTransfer 得到。**未报告**：序列长度、batch size、warmup 长度、grad clip、训练 seed 数（看起来每点一次）、dropout。
- **循环次数**：固定（R=2；§6.3 另训 4×4、2×8）；**反传**：全展开（未提 truncation）。
- **规模网格**（Table 5）：

  | d_model | d_ff | heads | w_ratio | Active (M) | Stored Looped-MoE (M) | Stored MoE (M) |
  |---|---|---|---|---|---|---|
  | 128 | 384 | 2 | 1.0 | 16 | 18 | 23 |
  | 256 | 704 | 4 | 2.0 | 39 | 45 | 65 |
  | 384 | 1024 | 6 | 3.0 | 67 | 81 | 124 |
  | 512 | 1408 | 8 | 4.0 | 103 | 129 | 207 |
  | 640 | 1728 | 10 | 5.0 | 144 | 184 | 303 |
  | 768 | 2048 | 12 | 6.0 | 190 | 247 | 417 |
  | 896 | 2432 | 14 | 7.0 | 246 | 325 | 560 |
  | 1024 | 2752 | 16 | 8.0 | 305 | 407 | 711 |

  d_ff 向上取 64 的倍数（≈2.7×d_model）；head dim 64；"Some isoFLOP runs use intermediate widths"。预算 C ∈ {5×10¹⁶, 2×10¹⁷, 5×10¹⁷, 10¹⁸} FLOPs，每预算 4–7 个宽度（图读）。注意：d=128 时 embedding+unembedding = 2×50,257×128 ≈ 12.9M，占 N_active（16M）的 ≈80%；d=704 时仍 ≈43%。
- **loop-depth 变体**（§6.3）：8×2、4×4、2×8，同宽 d_model = 704（10¹⁸ 处 compute-optimal 8×2 的宽度），16 有效层，10¹⁸ FLOPs。
- **评测**：test loss（held-out）；分析与早退用 800K test tokens；OLMES Core 9（ARC-E、ARC-C、BoolQ、CSQA、HellaSwag、OBQA、PIQA、SIQA、WinoGrande）；早退用输出分布熵阈值 τ，循环模型只在循环边界退出，非循环模型任意层退出。

## 函数形式与拟合方法
- **两步法**（§4.1）：每个预算内对各宽度的 test loss 拟合二次曲线（横轴 N_active 对数刻度），取最小值为该预算的 compute-optimal 点；再对 4 个预算的 4 个点拟合 **L ∝ N^{−α}**（纯幂律：无不可约项 E、无 D 项、无置信区间）。图 3/9/10 的纵轴刻度间距非均匀，经数字化确认为 **log-loss**，即 log-log 拟合。
- **拟合出的 α**：Base **0.076**、Looped-MoE **0.077**（§5.1、Figure 3）；Looped **0.084**、MoE **0.072**（**仅在 Figure 9 图例中**，正文未提）。本人数字化 Figure 10 四条拟合线得 0.0763/0.0789/0.0859/0.0731，与图例一致（校准有效）。
- **与 Kaplan 的对比**："consistent with the 0.076 scaling exponent reported by Kaplan et al."——注意 Kaplan 的 α_N 是数据充足极限下 loss-vs-N 的指数，不是 compute-optimal 前沿上的 loss-vs-N* 指数，二者不同义。
- **拟合质量**（本人数字化各 compute-optimal 星点相对拟合线的残差，nats，按预算 5e16→1e18）：Looped +0.044/−0.031/−0.064/+0.053；Base +0.015/−0.039/+0.009/+0.011；Looped-MoE +0.072/−0.072/−0.041/+0.044；MoE +0.042/−0.044/−0.032/+0.036。即 4 点幂律的残差达 ±0.04–0.07 nats，α 之间 0.072–0.084 的差异不显著；作者未报告残差或 CI。
- **作者对形式的说明**（Figure 10 图注）："All architectures have roughly the same scaling slope, but are offset differently. MoE scales the best, followed by Looped-MoE, then Base and lastly Looped. Looping the same architecture seems thus to strictly reduce expressivity. However, in this paper we focus on how adding looping and MoE improves upon dense Base scaling and also Looped scaling."

## 主要结论
1. **排序**（§5.1、Figure 10）：test loss 由低到高 MoE < Looped-MoE < Base < Looped。原话："We find that Looped-MoE scales better than Base, while Looped models scale strictly worse (Figure 1, middle)"；"Looping alone is detrimental, with MoE scaling better than Looped-MoE and Base better than Looped (Figure 10), consistent with prior work [Kaplan; Parcae]".
2. **每个预算的 compute-optimal 点**（数字化 Figure 3/9 的星点质心；精度约 ±0.005 nats、±3% N；括号后为与同预算 Base 的损失差）：

  | C (FLOPs) | Base (N*, L*) | Looped | Looped-MoE | MoE |
  |---|---|---|---|---|
  | 5×10¹⁶ | (31.7M, 4.086) | (38.3M, 4.182) **+0.096** | (37.0M, 4.062) −0.024 | (31.2M, 4.028) −0.058 |
  | 2×10¹⁷ | (72.5M, 3.785) | (92.0M, 3.812) **+0.027** | (57.1M, 3.785) 0.000 | (58.4M, 3.767) −0.018 |
  | 5×10¹⁷ | (143.4M, 3.641) | (124.6M, 3.683) **+0.042** | (96.2M, 3.664) +0.023 | (96.2M, 3.645) +0.004 |
  | 10¹⁸ | (204.2M, 3.548) | (244.9M, 3.594) **+0.046** | (176.4M, 3.580) +0.032 | (170.8M, 3.565) +0.017 |

  交叉验证：Table 7 "Full depth" 困惑度（10¹⁸ compute-optimal 模型，800K test tokens）Base 34.8、MoE 34.8、Looped 36.2、Looped-MoE 35.9 → 损失 3.550/3.550/3.589/3.581，与数字化值差 0.005–0.015。**要点**：(i) 稠密 Looped 在**每个预算**都比 Base 差 0.03–0.10 nats（≥2×10¹⁷ 时 0.03–0.05，约 1% 损失，Table 7 在 10¹⁸ 给 +4% 困惑度）；(ii) Looped-MoE / MoE 只在 ≤2×10¹⁷ 优于 Base，在 5×10¹⁷ 与 10¹⁸ **反而比 Base 高 0.02–0.03 nats**——"Looped-MoE scales better than Base"来自 4 点幂律拟合线的平均偏移，并不被最大两个预算的原始点支持（Table 7 的 35.9 vs 34.8 亦然），作者未讨论。
3. **拟合线偏移**（数字化 Figure 10，同 N_active）：Looped − Base = +0.141（30M）/ +0.116（50M）/ +0.085（100M）/ +0.056（200M）；Looped-MoE − Base = −0.030/−0.034/−0.039/−0.044；MoE − Base = −0.089/−0.079/−0.067/−0.056 nats。
4. **循环中间层 vs 整栈**：**无数据**——本文所有循环模型均为整栈（8×2、4×4、2×8），无 prelude/coda 对照。
5. **循环 2 次 vs 4/8 次**（§6.3、Table 7；仅 10¹⁸、d=704、同宽同 L_eff=16）：全深度困惑度 Looped-MoE 8×2 **35.9**、4×4 **35.4**、2×8 **37.3**（Base 34.8、MoE 34.8、稠密 Looped 8×2 36.2；稠密 4×4/2×8 只有 Figure 7 左的曲线，无数值）。作者只讨论早退折衷（"For Looped-MoE, more loops yield a strictly better compute-quality tradeoff"；稠密 Looped "though not strictly at all savings levels"），并明说"we leave the impact on scaling for future work"。早退 Table 7（省 5%/10%/20%/30% FLOPs 时困惑度）：Base 43.1/55.4/112.6/272.6；MoE 51.9/75.7/167.5/369.2；Looped 40.9/50.2/84.3/156.0；Looped-MoE 41.1/51.0/89.7/182.6；Looped-MoE 4×4 38.2/44.3/71.0/160.1；2×8 38.8/42.0/57.6/153.5。
6. **稠密循环 vs 循环 MoE**：见 1–3。机制解释为 routing divergence（§6.1，Figure 5，10¹⁸ 的 Looped-MoE，k=2/E=8）：层 1–6 与 8 上 25–53% 的 token 两次访问 expert 集完全不同、仅 4–14% 完全相同；层 7 例外（37% 完全相同、3% 完全不同），"possibly stabilizing embeddings before vocabulary projection at the loop boundary"。
7. **下游**（Table 4/6，10¹⁸ compute-optimal，OLMES Core 9 平均）：Looped 37.4（168M 存储）、Base 38.7（246M）、Looped-MoE 39.6（216M）、MoE 36.4（366M）。逐任务（ARC-E/ARC-C/BoolQ/CSQA/HellaSwag/OBQA/PIQA/SIQA/WinoG）：Looped 39.1/24.6/49.6/27.1/25.6/24.4/57.1/38.2/51.2；Base 39.3/25.3/51.4/29.3/26.2/27.6/57.8/40.5/50.5；Looped-MoE 40.4/24.3/63.9/30.9/25.4/26.8/55.2/38.0/51.7；MoE 38.9/23.0/39.0/30.0/26.7/25.2/56.1/39.2/49.7。Looped-MoE 的领先几乎全来自 BoolQ（63.9 vs 51.4），其余任务与 Base 互有胜负；此规模下多数任务接近随机。
8. **早退机制**（§6.2）：JSD<0.5 的 token 比例在循环边界跳升，"By the end of the first loop iteration, the majority of tokens have already reached near-final output distributions"（Figure 6，无数值）。

## 失败模式与警告
- 4 个预算（20× 范围）、每预算单次训练、4 点无 E 项幂律；星点残差 ±0.04–0.07 nats；未给任何 CI。
- N_active 含 embedding，小宽度下占 40–80%，使 N 轴与真实非嵌入容量脱节；也使 Looped 的"参数减半"实际只减少非嵌入部分。
- 未报告 seq len、batch、warmup、seed；绝对数值不可复现。
- μP 不跨深度迁移（§8），只扫宽度、深度固定 16 层，深宽比随规模变化未控制。
- "Looped-MoE 优于 Base"在最大两个预算处反号（主要结论 2），作者未讨论。
- MoE 在 test loss 最好但 OLMES 最差（36.4）："MoE scores lowest on average (36.4) despite achieving the best test loss."（Table 6 注）——损失与下游脱节。
- 早退只算理论 FLOPs："We do not measure end-to-end inference throughput."（§4.2）

## 对我们实验的直接启示
**应该照搬（具体到数值）**
- iso-FLOP 协议本身：同宽度下 N_active（= 每 token FLOPs）相同、有效深度相同（16）、D = C/(6·N_active)；这与 iso-depth 设定一致，是稠密模型可行的匹配方式。
- μP（d_base=128，init 方差与 LR 按 1/w_ratio，扩展到 unembedding）+ 在 4 个宽度（128/256/512/1024）上做 LR 迁移验证（LR 扫 10⁻³–10⁻¹，容忍 <1% 损失差）；循环层不需要额外 μP 修正。
- 每预算 ≥5 个宽度、二次曲线取最小值；预算范围至少 20×（5×10¹⁶–10¹⁸ 在 4×RTX 4090 上单次 10¹⁸ 约数小时量级，可行）。
- 用 Table 7 式"全深度困惑度/损失"逐点交叉核对拟合线。
- AdamW β=(0.9, 0.999)、ε=10⁻⁸、weight decay 10⁻⁴（independent）、WSD + 最后 10% 步 sqrt-decay 到 5%，可作为起点。
**应该避免或改进**
- 不用纯幂律 L ∝ N^{−α}：加 E 项并拟合 L(N,D) 曲面或 Schwethelm 式联合律 L = E + A·(N_once + r^φ·N_rec)^{−α} + B·D^{−β}；对每个预算/架构给残差与 bootstrap CI。
- N 只计非嵌入参数（SMELT/Schwethelm 口径），embedding 单独报告；FLOPs 计入 unembedding 与注意力项（Kaplan 式 6·N_nonemb + 6·L_eff·ctx·d）。
- 报告并固定 seq len、batch、warmup、seed；最小档跑多 seed。
- 不把"拟合线偏移"当结论：必须检查每个预算的原始 compute-optimal 点是否同号（本文在 5×10¹⁷/10¹⁸ 处反号）。
- 不只用整栈循环：必须加 prelude/coda 臂才能回答 placement 问题。
- Kaplan 的 0.076 不是合适的参照。
**(a) "循环中间块优于整栈"的证据具体是什么、在什么规模**：**无**。本文全部循环模型为整栈（L=8×R=2，或 4×4、2×8），没有 prelude/coda 对照。唯一可借用的是 Figure 5 的层 7 例外（循环边界前最后一层的路由高度一致，被解释为"stabilizing embeddings before vocabulary projection"）——它暗示整栈循环时末层被迫兼任 coda，是把末层设为独立 coda 的间接动机。
**(b) 稠密循环结果与 iso-depth 的 φ=0.46 是否一致**：设定可比（同 F、同 L_eff=16、r=2、N_once=0、N_unique 的非嵌入部分减半），这正是 iso-depth 协议。**方向一致**：稠密 Looped 在全部 4 个预算都比 Base 差（+0.027 到 +0.096 nats）。**幅度**：φ=0.46 意味着 N_eff = 2^0.46·N_rec = 0.688·N_active(非嵌入)，即少 31% 的等效非嵌入参数。用 Base 前沿斜率 α≈0.0755 粗略换算，观测到的 +0.027/+0.042/+0.046 nats（≥2×10¹⁷）对应 N_eff/N_active ≈ 0.91/0.86/0.84（φ ≈ 0.86/0.78/0.75），5×10¹⁶ 处 +0.096 对应 0.74（φ ≈ 0.56）；若改用"同损失所需 N* 之比"换算，φ 在 0.2–0.9 间跳动（星点残差太大）。结论：本文与 φ<1 的方向一致，但**无法区分 0.46 与 0.7–0.8**；其 N 含 embedding（占 40–80%）会系统性压低稠密循环的相对损失，且 16 层、≤10¹⁸ FLOPs、无 E 项。只能作为"稠密整栈循环在此设定下损失高 1–2.5%"的量级参考，不能用来校准 φ。
**(c) 匹配口径如何在稠密小模型上复制**：可直接复制——同宽度、同 L_eff、C = F·D（F 改用实测含注意力）；每预算扫 5–7 个宽度并二次拟合取最优；同一数据流与 tokenizer；KV 与 FLOPs 均按执行层数计（同 L_eff 下与 Base 自动相等）。在此之上补：(i) 非嵌入 N 口径；(ii) 每个 (C, 架构) 的原始最优点表（不是只给拟合线）；(iii) 联合 φ 拟合；(iv) prelude/coda 臂与截断反传臂。

## 留白（§8 Limitations 及文中明示的未做项，逐条引用）
1. "μP transfer does not hold when model depth changes, so we hold depth constant and scale width. Future work could validate with depth-scaling extensions such as CompleteP [31]."
2. "Due to compute constraints, we did not scale our four architectures beyond 305M/711M (active/stored) parameters, but rely on the principle that compute-optimal scaling laws fitted at smaller scales predict larger model performance. We aim to validate this with extended pretraining at 1B and 7B scales."
3. "Finally, our early-exit results demonstrate theoretical compute savings via a layer exit criterion, but we have not yet measured end-to-end throughput gains with optimized inference engine."
4. §6："For Looped-MoE, we find that more looping further improves early-exit trade-offs (Sec 6.3); we leave the impact on scaling for future work."
5. §6.1："We speculate this loop-invariance reflects a structural role for layer 7, possibly stabilizing embeddings before vocabulary projection at the loop boundary."
6. Table 6 注："We hypothesize this reflects narrower per-token expert access: each token consults only k=2 experts per layer in a single pass, whereas Looped-MoE tokens access 3–4 unique experts per physical layer across loops due to routing divergence (Section 6.1)".
7. §4.2："We do not measure end-to-end inference throughput."

> 数字化方法注：取 arXiv HTML 的原始 PNG（2605.09165v2/figures/isoflops_fit_curves_{base,looped,looped-moe,moe,combined}_active.png，1728×1361）；用轴框与主刻度线定位坐标（x 为 log 轴，279.1 px/倍频程；y 为 log 轴，主刻度 3.60/3.80/4.00/4.20 回代误差 <0.001 nats）；compute-optimal 星形标记取同色连通域质心（与拟合线粘连的星点按线宽突起识别）；拟合线取像素列中心后在 log-log 下最小二乘。四条拟合线的斜率复原为 0.0763/0.0789/0.0859/0.0731，与图例 0.076/0.077/0.084/0.072 一致；10¹⁸ 处的星点损失与 Table 7 的全深度困惑度一致到 0.005–0.015 nats。
