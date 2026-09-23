# 阅读笔记 05 — Chen, Vegesna, Dahal & Wilson, "How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents" (arXiv:2609.19107v2)

> ID 核对：arXiv 2609.19107 确为本文，无差异。阅读版本 v2（2026-09-17，最新）。来源：arxiv.org/html/2609.19107v2 全文 + arxiv.org/pdf/2609.19107v2（表格与图内嵌文字用 PDF 核对）。全文（正文 §1–7、附录 A–D）均已读到；无缺失章节。标注"图读"的数字只能从图中读出。

## 基本信息
- 作者/机构：Zixi Chen（New York University；† "Work done as an intern at Q Labs"）、Akshay Vegesna（Q Labs）、Samip Dahal（Q Labs）、Andrew Gordon Wilson（NYU & Q Labs）。
- 日期/版本：v1 2026-09-16，v2 2026-09-17（arXiv 提交记录，两版均 1,275 KB）；44 页；cs.LG。
- 发表状态：arXiv 预印本，ICLR 风格模板，未注明会议/期刊。
- 代码：https://github.com/qlabs-eng/scaling-exponents（首页脚注与 arXiv comments）。

## 他们匹配了什么
- 单 epoch 主结果**匹配训练 FLOP**："we compare at equal compute throughout"（§4.1）。每条 ladder 的每个点是一个 compute-optimal 模型（TPP 固定），在同一 compute 下比 loss；compute multiplier 由 loss 匹配反推 compute。
- 参数量的两个定义：
  - **stored parameters N**：实际存储的参数，tied core 只算一次；用于模型大小、TPP = T/N 和 LR 规则（"the number of stored blocks, which we use for model size throughout"，§3）。
  - **compute-active parameters N_c,K**："the prelude, coda, and output-head matrix parameters counted once, plus the core matrix parameters counted K times, even when shared. This count excludes the input embedding lookup."（App. A.2.2）
- FLOP 公式原文："Plotted compute comes from the model FLOP estimator, which includes matrix multiplications and attention computation; expressions of the form 6TN_eff below are leading-order allocation identities."（App. A.1）；分配恒等式 C = 6 D N_c（App. A.3；Eq. 9：C = 6 D_{K_ρ} N_{c,K_ρ}）。
- 三组受控比较（Fig. 1）：Vanilla vs Operator-1（参数与 FLOP 都匹配，只差边界算子）；Loop-2 vs Untied-2（FLOP 与执行深度匹配、参数不匹配：d8 时 Untied-2 存 11 块 vs 8 块，Table 2）；Loop-Grow vs Untied-Grow。
- 多 epoch（§5）：固定 100M unique FineWeb tokens × 10 epochs = 1B tokens，在同一 compute 下比较"加参数"与"加循环"。

## 模型与训练设定
**架构**（§3、App. A.1、Table 1/2）
- prelude–core–coda（Geiping et al. 2025 形式，Eq. 2）：e = P(s)；h_k = R_{θ_k}(φ(h_{k−1}, e))，k = 1…K；y = C(ρ(h_K, e))。执行深度 ℓ = P + K·C + D。
- 边界算子 BO(h, e) = Norm(h) + α·e（Eq. 3），用于每次 core pass 之前**和 coda 之前**（φ = ρ = BO；Algorithm 1 第 16、19 行用 RMSNorm）。α = α_emb 可调：Operator-1 = 1，Loop-2 = 0.707，Untied-2 = 1（Table 5）。作者强调 coda 前的注入是他们相对以往 looped transformer 的新增，且重要（§3；Table 4）。
- 初始状态 h_0 = 0；K 训练时固定，不采样；"gradients flow through every pass"（§3）/"Backpropagate through all K passes"（Alg. 1 line 22）→ 全展开反传，**无截断**。
- 块分配：三段尽量均分，余数先给 core 再给 coda（App. A.2.1）。Table 2：d6 = 2/2/2，d8 = 2/3/3，d10 = 3/4/3，d12 = 4/4/4，d14 = 4/5/5，d16 = 5/6/5，d18 = 6/6/6，d20 = 6/7/7，d26 = 8/9/9。宽度 = 128 × 名义深度（d8 → 1024）。
- Transformer 细节（App. A.1）：pre-norm decoder-only，RoPE，SwiGLU，QK-norm；无 bias、无可学习 norm 增益；token embedding 后与 lm_head 前各加一个 RMSNorm；attention/MLP 输出投影与 lm_head **零初始化**；embedding 正态初始化（std = WTE init）；Q/K/V 与 MLP 输入矩阵均匀初始化（scale = UIS）。
- 残差缩放：**没有 1/N 一类随循环次数变化的缩放**。用常数 residual multiplier RM（"the shared weight multiplier to the MLP down projection and attention output projection"，App. A.1）和 output multiplier OM；tuned RM：Vanilla 0.25，Deep Vanilla 0.5，Operator-1 0.5，Loop-2 0.25，Untied-2 0.25（Table 5）。跨 pass 的稳定性靠 BO 中的 RMSNorm。
- Untied：每次 pass 独立 core，同一计算图、同一 FLOP；Growth：2→4 passes（tied 直接多跑；untied 复制已训练 core 后独立训练，Alg. 1）。

**训练**（Table 1、Table 3/5/9、App. A.4）
- 优化器：Muon（矩阵参数）+ AdamW（embedding 与 lm_head）；AdamW β1 = 0.8，β2 ∈ {0.95, 0.98, 0.99}，ε ∈ {1e-10, 1e-8}（Table 5）。Muon 自身超参未列出。
- 数据：FineWeb（GPT-2 tokenizer，50,257 padded to 50,304）；corpus-transfer 与外推用 FineWeb-Edu（App. C）。context 2048；global batch 524,288 tokens；单 epoch。
- Token 数 T = TPP × N（stored）：Vanilla 5，Deep Vanilla / Operator-1 / Loop-2 / Untied-2 6，Loop Grow t_ref = 7，Untied Grow t_ref = 8，Deep Vanilla Grow 6（Table 9；未取整均值 5.33 / 5.66 / 5.79 / 6.39 / 6.08，Table 7）。
- 学习率规则（Eq. 13）：GLR(N) = GLR_d8 · (N/N_d8)^β，N = 初始 stored 参数量；GLR_d8 = 0.04（Deep Vanilla Grow 0.036）；β：Vanilla −0.8，Deep Vanilla −0.7，Operator-1 −0.6，Loop-2 −0.6，Untied-2 −0.6，Deep Vanilla Grow −0.8，Untied Grow −0.6，Loop Grow −0.5（Table 9）。证据：一维扫描 "The preferred GLR falls from 0.04 at d8–d10 to 0.02 at d11–d12"（App. A.4.4）；Table 8 constant-recipe 漂移指数 Vanilla −0.78±0.06，Operator-1 −0.62±0.02，Loop-2 −0.58±0.08，Untied-2 −0.66±0.02；施加规则后残余漂移 ≈ 0（Vanilla −0.00±0.01，Operator-1 −0.11±0.02，Loop-2 +0.03±0.00，Untied-2 −0.04±0.07）。其余 HP 不随规模变：加 OM/WD 缩放只改 multiplier ≤ 3%（Fig. 16(a)）。GLR 是最敏感的 HP（2× 变化的 regret 12–19×10⁻³，Table 8）。
- 其他 base HP（Table 5，d8、1B tokens、两轮链式调参，采纳阈值 1e-3）：ELRM/HLRM（embedding/head LR 乘子）Vanilla 0.453/0.113，Deep Vanilla 0.16/0.057，Operator-1 0.905/0.08，Loop-2 0.32/0.113，Untied-2 0.16/0.16；OM Vanilla 0.5，其余 1；WD Vanilla 0.071，Deep Vanilla 0.1，Operator-1 0.05，Loop-2 0.05，Untied-2 0.071；WTE init 0.007 / 0.005 / 0.113 / 0.02 / 0.01；UIS 0.063 / 0.5 / 0.354 / 0.044 / 0.354。
- 调度：warmup 步数 WU（网格 {0,5,10,20}，初值 40）：Vanilla 40，Deep Vanilla 5，Operator-1 0，Loop-2 40，Untied-2 40；warmdown ratio WDR（线性衰减占比，网格 {.2,.6,.8,1}）：Vanilla 0.6，Deep Vanilla 0.8，Operator-1 0.8，Loop-2 1，Untied-2 1（Table 3/5）。（d8 在 1B tokens ≈ 1,900 步，warmup 40 步 ≈ 2%。）
- 规模网格：Vanilla d6–d20 偶数（120M–1.8B）；looped 变体 d6–d18 偶数（tied 120M–1.4B，untied 130M–1.8B）；每条 ladder 终点 ≈ 1e20 FLOPs（App. A.4.5）；TPP 拟合用五个预算 d8–d12（App. A.4.2）。外推点：Untied-Grow d26，7.4B（起始 5B），1.225e21 FLOPs（8× 最大拟合 compute）。
- 硬件：单节点 8×H100；d26 用两节点。d8 上 1B tokens 约 6.5 min（Table 4）。
- 种子：训练种子数**未说明**（Table 6 迁移探针 "trained once"；采纳阈值 1e-3 "reflects run-to-run noise"）；下游评测 seeds 0/1/2（App. C.2）。
- 多 epoch 设定（§5、App. D）：Operator-1 family，K ∈ {1,2,3,4,6,8,12}，120M–1.4B，100M × 10 epochs，其余 HP 固定、只重调 WD，网格 {0.05, 0.2, 0.4, 0.8, 1.2, 1.6}（App. D.2）；K=1 选出的 WD：d6 0.4，d8–d10 0.8，d12–d14 1.2，d16–d18 1.6（App. D.3）；预算范围 0.85e18–6.7e18 FLOPs（Fig. 5 第一栏，图读）。对照：fresh ≈1B、250M×4、100M×10（Fig. 21）。

## 函数形式与拟合方法
- Scaling law（Eq. 1）：L(C) = E + A·(C/C_0)^{−γ}，C_0 = Vanilla 调参处的 compute。E 只在 Vanilla 上用 Huber loss 拟合，其余 arm 共享 E，做 log–log 仿射拟合 log(L − E) = −γ·log(C/C_0) + log A（§4.2）。斜率的回归标准误 < 1e-3（Fig. 3）。
- **Compute multiplier 定义**（§2）："Let Ĉ_A(ℓ) denote the compute at which architecture A's fitted law reaches loss ℓ. The compute multiplier of architecture B over A at loss ℓ is then π(ℓ) = Ĉ_A(ℓ)/Ĉ_B(ℓ)"；A = Vanilla，B = 变体；按 Vanilla 的预算索引 π(C) = π(L_A(C))。实际计算：在 log-loss/log-compute 空间中于最近两个测点间线性插值，**不外推**（§4.2）。"A flat curve is a constant improvement and a rising curve is an exponent improvement."（Fig. 3）
- 拟合到的 γ（Fig. 3 图例，PDF 内嵌文字）：Vanilla 0.111；Deep Vanilla 0.111；Deep Vanilla Grow 0.114；Operator-1 0.114；Loop-2 0.114；Loop-Grow 0.116；Untied-2 0.114；Untied-Grow 0.117。四位小数（App. C.1）：FineWeb Untied-Grow 0.1168 vs Vanilla 0.1113；FineWeb-Edu 0.1146 vs 0.1095。log A 落在 0.46–0.49（Fig. 3 右，图读）。
- 循环次数选择（App. A.2.2，Eq. 4/8）：固定 anchor、固定 compute 时最优 K* 随预算上升；E_{K*} = 0.82·TPP_{K1}^{0.15}（tied），0.83·TPP_{K1}^{0.15}（untied）；TPP_{K*} = 1.19·TPP_{K1}^{0.85}（tied），1.34·TPP_{K1}^{0.74}（untied）。固定 token 的 ladder 上最优 K 在 1–2 之间（Fig. 8）。TPP 关系式 TPP_K = TPP_{K1}/(E_K S_K)（Eq. 7），近均分时 E_K → (K+2)/3。
- 生长时机（App. A.2.3，Eq. 5）：TPP_{K_ρ*} = a·TPP_{K2} + b，(a, b) = (0.902, 0.216) Loop-Grow，(0.883, 0.103) Untied-Grow，R² > 0.9999；生长后 token 占比 ρ 的拟合均值 Loop Grow 0.170、Untied Grow 0.271、Deep Vanilla Grow 0.544→0.5（Fig. 13）；用常数 ρ = 0.30 替代 fitted ρ 改变 loss −0.0009 到 +0.0019，复用 TPP 6 改变 compute < 5%（App. A.2.4）。
- 多 epoch：不拟合幂律；在匹配 compute 的切片上以 log-loss vs log-K 二次拟合取最优 K（Fig. 5）。
- 作者报告的不稳定/异常：未报告发散。但 (i) 把 Vanilla 配方迁移到 Operator-1 多 7.8e-3 loss 并抹平指数增益（Table 6、Fig. 16(b)）；(ii) 固定 prelude/coda 只扩 core 时 multiplier 在最大预算跌破 1（App. B.1）；(iii) Muon 相对 Adam 的增益在宽深耦合缩放时随规模缩小（App. B.5）；(iv) 随机循环数训练略差于固定（App. B.4）。

## 主要结论
1. **单 epoch，tied vs untied 差距**：Untied-2 高于 Loop-2 1.08×（1e18）→ 1.06×（1e20）；Untied-Grow 高于 Loop-Grow 1.16× → 1.14×（§4.2 "Untying improves the constant"）。"Weight sharing therefore costs a fixed factor of compute at every scale, rather than a factor that grows with the budget." γ 相同：Loop-2 0.114 = Untied-2 0.114；Loop-Grow 0.116 vs Untied-Grow 0.117。d8、1B tokens 的 tuned loss：Loop-2 3.2704 vs Untied-2 3.2563（Table 4）。
2. **谁改指数**：BO（Operator-1 1.12× → 1.25×，γ 0.111 → 0.114）；growth（Untied-Grow 1.30× → 1.55×；growth 相对 Untied-2 1.09× → 1.16×；Loop-Grow 1.36× at 1e20，§1）。Deep Vanilla 常数 1.08×（更小宽深比）；1:64 不再改善（App. B.3）。
3. **外推**：7.4B Untied-Grow 落在预测曲线之下 0.03；CORE 0.3865±0.0015 vs 预测 0.3837；≈ GPT-3 13B 的 0.3852（2.31e22 FLOPs）→ "roughly 20×"（§4.3，作者称 "indicative rather than a controlled comparison"）；FineWeb-Edu 上 multiplier 1.6×（1e20）→ 1.8×（1.23e21）→ 2.7×（1e25）；下游 2.5–3.5×。
4. **多 epoch，最优循环数随算力变化**：最优 K 从 1.4 升到 6.7（预算 0.85e18–6.7e18 FLOPs，Fig. 5 第一栏）；与 Operator-1 的 loss 差距 0.00 → 0.11；扩循环比扩模型（WD 已调）省 2.2× compute（Operator-1 调 WD 只到 3.40 loss at 7.5e18）；循环之上再调 WD 最多再降 0.02。Fresh 与 4-epoch 下最优 K 仍在 1–2（App. D.1）；WD 越弱越偏好高 K，但调 WD 也不能消除（App. D.2）。
5. **"过拟合追踪存储参数"的证据**：(a) 最优 WD 随深度上升、随 K 几乎不变（Fig. 5 第四栏；App. D.3）；(b) 循环的边际收益从 d8 起各深度间差 ≤ 0.006（Fig. 5 第二栏）；(c) untied 循环加同样深度但不胜 Operator-1（Fig. 24）；(d) KL 有效深度 18 vs 14（Fig. 6 右）。
6. **指数是否改变**：单 epoch 下 tying 不改指数（只改常数）；多 epoch 下作者未拟合指数，只报 multiplier。
7. 其他：tied 模型的 GLR/WD/OM/RM/α_emb 最优值在 K = 2,4,8 间 "broadly similar"，untied 偏移更大（App. D.3、Fig. 23(b)）；KL 有效深度（阈值 2 nats）Untied-Grow 36 vs Untied-2 24 vs Deep Vanilla 20（1e20，Fig. 6 左）；looped 变体偏好稍高 TPP（Table 7）。

## 失败模式与警告
- 必须按架构单独调参并给出 LR 随规模的规则，否则指数改善会被判成常数改善：constant recipe 下 multiplier 更"好看"（Fig. 15(c)），Vanilla 配方直接迁移让 Operator-1 的指数增益变常数（Fig. 16(b)）。
- "on fresh data, weight sharing is not compute-optimal, nor is increasing the loop count across scales"（§5）。
- 去掉 coda 前的注入：Loop-2 loss 3.2704 → 3.2912；Untied-2 multiplier 1.34× → 1.17–1.19×（Table 4、App. B.1）。
- prelude/coda 必须随规模等比放大，固定 2/C/3 或 1/C/2 会使 multiplier < 1（App. B.1）。
- 固定 K 训练的模型测试时多跑 pass 会变差：k = 8 vs 4 增加 0.008–0.042 loss（Fig. 17）。
- Muon 在深模型上优势缩小（App. B.5）；batch size 固定、未与规模联合优化（App. C.5）。
- 文内不一致：App. A.4.1 写迁移代价 "7–26×10⁻³"，Table 6 最大值为 13.3×10⁻³。

## 对我们实验的直接启示
**应该照搬**
- 边界算子：每次循环前 RMSNorm(h) + α·e，且**输出头之前同样做**；α 从 {0.707, 1} 起扫；h_0 = 0；K 固定；全反传臂与他们一致。
- 8 块的分配用 2/3/3（他们的 d8 分配）；不要把 prelude/coda 固定成 1 块。
- 参数量报两套：stored N 与 compute-active N_c（core 按 r 次计、排除 embedding lookup）；compute 用含 attention 的估计，正文注明 6·T·N_c 只是主阶恒等式。
- 拟合：E 只在 r=1 基线上 Huber 拟合并共享；其余臂 log–log 仿射；multiplier 用最近两点插值不外推；报告斜率标准误。φ 的参考点：K=2 时 tying 只损失 1.06–1.08× compute（匹配 FLOP、非匹配参数）。
- LR 随规模缩小：GLR ∝ N^β，β ≈ −0.6（looped）/ −0.8（vanilla）；只缩放 LR，其余 HP 固定（≤3%）。
- WD 在 r=1 上调好后跨 r 复用（Fig. 5 第四栏、App. D.3）。
- 若加多 epoch：100M × 10 epochs，WD 网格 {0.05, 0.2, 0.4, 0.8, 1.2, 1.6}。
**应该避免/改进**
- 他们用 Muon 且宽深耦合（w = 128·depth），LR 指数 β 混合了宽度与深度效应，不能直接套到我们"固定 8 层、只扩宽度、AdamW"的网格。
- 他们的 LR 规则按 stored N 索引，tied 模型不同 r 的 N 相同 → LR 不随 r 变；我们若把 φ 表述为"等效参数量"，要明确 LR 是否按等效参数缩放。
- 单种子（未说明）；我们在 20M–160M 上至少 2–3 种子估计噪声（他们的采纳阈值 1e-3）。
- 无截断反传、无整栈循环（Loop-K 始终带 prelude/coda）——这两项都是我们要补的空白。
**三问**
- (a) 本文没有 ε = 1/N 规则；它用常数 RM（0.25/0.5）加跨 pass RMSNorm。对 8 层循环 r 次，本文能给的对应物是"每次 pass 前归一化 + 注入"，与 N 的定义无关。
- (b) **不支持只在最小尺寸扫 LR 再固定迁移**：最优 GLR 随规模下降（0.04 → 0.02，d8 → d11/12），constant recipe 会扭曲 multiplier 趋势（Fig. 15(c)、16(a)）。支持的是：在小尺寸拟合一个 GLR ∝ N^β 规则后迁移；且 tied 模型的 LR 跨 K = 2,4,8 大致可复用（Fig. 23(b)）。
- (c) **是**：多 epoch 下"一次循环值多少参数"从"常数因子劣势"变成"随算力增长的优势"（最优 K 1.4 → 6.7），φ 明显依赖数据 regime。至少加一个数据受限臂（固定 unique-token 池 × 10 epochs、只重调 WD），否则应把 φ 明确限定为单 epoch 结论。

## 留白
（本文无独立 Limitations 节；以下摘自 §7 Discussion、§4.3、§6 与附录。）
- §7："Context length, the number of experts in a mixture-of-experts model, and width all grow with compute, yet all are fixed before training begins, so growing them on the schedule the network needs may change the exponent."
- §7："Within depth itself, staged schedules (two passes, then four, then six) may increase the exponent further still."
- §7："the distinction between constant and exponent improvements deserves to be a standard part of how new architectures and training recipes are evaluated"
- §4.3："The two models were trained on different data, and the GPT-3 reference is an estimate from a separate evaluation pipeline, so the roughly 20× gap is indicative rather than a controlled comparison."
- §6："KL effective depth is only a proxy for computational depth, since falling within this threshold does not mean later blocks stop changing the prediction, and unused blocks in the middle of the network go undetected."
- App. B.4："In this setting, random recurrence chiefly reduces the penalty for extra passes rather than providing sustained test-time scaling."
- App. B.5："This discrepancy between width and joint scaling motivates investigation on the interaction of architectures and optimizers."
- App. C.4："these projections assume that the small-scale relationships continue beyond the measured ladders."
- App. C.5："The ladders use a fixed batch size and common hardware, so their fitted recipes do not address joint optimization of batch size and model scale."
- §2（对 Wang et al. 2026c 的评论）："In our setting, by contrast, we find that weight sharing alone is not enough to improve the exponent."
