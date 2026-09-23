# 阅读笔记 06 — Wang, Li, Zhang, Huang, Yan & Li, "On the Residual Scaling of Looped Transformers: Stability and Transferability" (arXiv:2606.18524v2)

> ID 核对：arXiv 2606.18524 确为本文，无差异。版本：v1 2026-06-16，v2 2026-09-11（PDF 内页日期 "September 14, 2026"）。阅读 v2 PDF 全文（19 页，正文 §1–6、Limitations、Appendix A 证明、Appendix B 实验设定，均已读到）。v2 无 HTML 渲染，图片取自 v1 HTML；核对后 v1/v2 的图注与所有引用数字一致，v2 的改动是附录重编号（v1 的 §7/§8 → Appendix A/B）并在相关工作新增 Chen et al. 2026 "Training-free looped transformers"（[4]）。标注"图读"的数字只能从图中读出。

## 基本信息
- 作者/机构：Shaowen Wang（1,2,3）、Bingrui Li（1,2,3）、Ge Zhang（2,3）、Wenhao Huang（2）、Shen Yan（2）、Jian Li（1）；1 Tsinghua University，2 ByteDance Seed，3 M-A-P。通讯：wangsw23@mails.tsinghua.edu.cn、zhangge.eli@bytedance.com。
- 19 pages, 9 figures；cs.LG；发表状态未注明（预印本）。
- 代码：**无**（arXiv 页与全文均无链接；"We do not release new model weights, datasets, or downstream applications"，Ethical Considerations）。资助 NSFC Grant 04130200126。

## 他们匹配了什么
- 匹配的是 **token 数与 unique 参数量**：所有 (L, N) 组合都训练 20,000 步 × 0.5M tokens = 10B tokens（Table 2）；同一 L 下参数量与 N 无关（"Params (tied)" 183.5M / 268.4M / 438.3M）。
- **不匹配训练 FLOP，也不匹配部署 FLOP**：N = 8 每 token 计算量约为 N = 1 的 8 倍，论文没有任何 compute-matched 比较，也没有 FLOP 公式（全文无 6ND 类表达）。
- 参数量定义：Table 2 的 "Params (tied)" = 与 LM head 绑定的 embedding（128,256 × 768）+ L 个 unique block；embedding/LM head 在循环外。

## 模型与训练设定
**架构**（§5 "Model structure"、Table 2、Table 3）
- Decoder-only Llama-style pre-norm，RMSNorm，SwiGLU（Limitations 提到），Llama 3 tokenizer（vocab 128,256），d_model 768，12 attention heads / 12 KV heads（无 GQA），head dim 64，MLP dim 2048，tied embedding/LM head。
- **整栈循环**：embedding 与 LM head 在循环外，"the entire L-block sequence is repeated for N passes"。无 prelude/coda，无输入注入（全文无 "injection"），无循环间归一化；循环初值即 embedding 输出（Fig. 7 的 loop step 0 = post-embedding）。N = 1 就是标准 L 层 pre-norm 模型。
- 残差缩放精确公式（Table 3）：
  - attention 分支：x_{n,ℓ} + λ·N^{−1}·m_L^{−1/2}·Attn(LN(x_{n,ℓ}))
  - MLP 分支：z_{n,ℓ} + λ·N^{−1}·m_L^{−1/2}·MLP(LN(z_{n,ℓ}))
  - m_L = L/12（参考深度 L_ref = 12），λ = 1（Table 4）。即 ε = λ/(N√L) 归一到参考深度后为 ε = N^{−1}·(L/12)^{−1/2}；每个 unique block 的两条分支都乘同一 ε（"the constant factor of two per block is absorbed into λ"，§5）。
- 训练时 N 固定（每个 N 单独训练）。反传：未提截断，按全展开反传理解（"truncat" 仅出现在 truncated normal 初始化）。
- 参数化（Table 3、App. B.2）："Width scaling terms are set to one"——宽度轴为 **SP**（所有模型 d = 768，无 μP 宽度规则）。深度轴：hidden 权重 LR η0·m_L^{−1/2}，pre-LN LR η0·m_L^{−1/2}，hidden bias LR η0·m_L^{−1/2}，AdamW ε for residual blocks ε0·m_L^{−1/2}；embedding / final-LN / unembedding LR η0，AdamW ε 为 ε0；所有初始化方差 σ0²（σ0 = 0.02，截断正态）；WD ω0 = 0.1 不缩放。循环轴：只有残差乘子 N^{−1}，LR 不随 N 变。
- 优化器与调度（Table 2/4）：AdamW (β1, β2) = (0.9, 0.95)，ε0 = 1e-8，WD 0.1；warmup-stable-linear-decay（WSD，Wen et al. 2024）：500 步 warmup（2.5%）、末尾 1,000 步线性衰减（5%）；global batch 0.5M tokens；20,000 步；10B tokens；单 epoch；**序列长度未说明**（只有初始化诊断实验写了 seq len 128、batch 1、10 步）。
- 基础 LR η0 = 1.25e-3（L = 12）；LR 网格 {5, 7.5, 10, 12.5, 15, 20, 30, 40}×1e-4；发散判据：最终验证 loss > 4 的 run 剔除。
- 规模网格：L ∈ {12, 24, 48} × N ∈ {1, 2, 4, 8}（Table 2）；sqrt vs linear 的对照只在 L = 12（Fig. 5a,b）。
- 种子：语言模型实验**单种子**（seed 0，Table 4；Limitations）；初始化诊断 10 seeds。
- 数据：FineWeb-Edu，10B tokens；验证集 FineWeb-Edu held-out，取训练结束时的 loss。

## 函数形式与拟合方法
- 单层 setup（Eq. 1）：h_{n+1} = h_n + ε·W·φ(h_n)，ε = N^{−α}，W_ij ~ N(0, 1/d)，φ = ReLU，‖h_0‖²/d = Θ(1)。
- 推导要点（§3.2、Theorem 1、App. A.2）：R_N² = R_0² + B_N + C_N，B_N = O(εN)，C_N = (ε²/d)‖Σ_n r_n‖² = (ε²/d)‖W·Σ_n u_n‖²（W 共享可提出）。ReLU 非负 ⇒ Σ u_n 在正锥内，Lemma 4：m₋N ≤ d^{−1/2}‖U_N‖ ≤ √q₊·N；Gaussian W 在正锥上 a_γσ_W‖x‖ ≤ ‖Wx‖ ≤ 3σ_W‖x‖（Lemma 5，Gordon escape-through-a-mesh）⇒ (1/d)‖Σ r_n‖² = Θ(N²) ⇒ 需 εN = O(1)，即 α ≥ 1。对照非共享深栈 E[R_L²] ≈ exp(c·L^{1−2α})·R_0² ⇒ α ≥ 1/2（App. A.1）。
- LR 规则（Theorem 2、App. A.3）：Adam 式更新 ΔW_ij = (η/√d)·S_ij；d^{−1/2}‖Δh_N‖ ≤ C_Δ·Q₊·e^M·ηεN ⇒ ηεN = O(1) 充分；sharpness 条件 ζ_N = Θ(1) 下 ε = N^{−α} 给 η ∝ N^{α−1}；α = 1 ⇒ η = Θ(1)，与 N 无关。
- 多层（Eq. 2、Theorem 3/6）：h_{n,ℓ+1} = h_{n,ℓ} + ε·W_ℓ·φ(h_{n,ℓ})，h_{n+1,0} = h_{n,L}。假设 (7)–(9)：a₋ ≤ E[d^{−1}‖Y_ℓ‖²] ≤ a₊；条件均值 E[d^{−1}‖B_ℓ‖²] ≤ C_0β²；单层替换 ≤ C_1β²（β = εN，"local influence condition"）⇒ E[d^{−1}‖Σ_ℓ G_ℓ‖²] ≤ C·N²(L + β²L²)；取 ε = λ/(N√L) ⇒ ε²·E[…] = O(λ² + λ⁴)，与 N、L 无关；小 λ 时 unscaled 方差 Θ(LN²)。证明用 Hoeffding–ANOVA + Efron–Stein。
- 多层 LR（Proposition 7、App. A.5）：d^{−1/2}‖Δh_out^lin‖ ≤ C_J·C_Δ·ηεNL = C_J·C_Δ·ηλ√L ⇒ η = O((λ√L)^{−1})；相干条件 (15) 下为 Θ((λ√L)^{−1})。汇总为 Table 1："Residual branch × λ N^{−1} m_L^{−1/2}；Looped stack LR η0 m_L^{−1/2}；Embed/Unembed LR η0"。
- 拟合方法：无曲线拟合；LR 扫描取最小 loss（星号）。
- 报告的不稳定情形：ε = 1 与 1/√N 时残差范数随 N 快速增长，"can explode during early optimization"（Fig. 2d,e；N = 64 时 R ≈ 60 与 ≈ 8，图读）；L = 48、N = 8 时 η0 > 2e-3 发散（§5.3）；1/N 下的一步更新量在 N = 1–64 内约 4× 范围、N ≈ 32 饱和（Fig. 4/9）——上界非紧。

## 主要结论
1. **ε = 1/N 的效果**：初始化时 R = d^{−1/2}‖h‖ 在 N ∈ {1,…,64} 内保持 ≈ 1（Fig. 2f），1/√N 增至 ≈ 8、无缩放增至 ≈ 60（图读）；训练后不同 N 的最终残差范数在同一 L 内收敛到近似同值（Fig. 7）；L = 12、N = 8 时 linear 比 sqrt 的最优 loss 低 **0.025 nats**（§5.2）。
2. **最优 LR 只取决于 L 的证据**：L = 12、linear 下 N = 1, 2, 4, 8 的最优 η0 全部 ≈ 1.25e-3（Fig. 5b，图读；对应 loss ≈ 3.005 / 2.97 / 2.945 / 2.91）；sqrt 下最优随 N 右移（N = 1, 2 ≈ 1.25e-3，N = 4 ≈ 1.5e-3，N = 8 ≈ 2e-3，Fig. 5a，图读）；L = 24、48 施加 m_L^{−1/2} 后最优 base η0 仍 ≈ 1.25e-3 对所有 N（Fig. 5c,d，图读）："a single base learning rate remains near-optimal across all three depths"。
3. **高循环数稳定性**：1/N 让 N = 8 可训且更优；但 L = 48、N = 8 在 η0 > 2e-3 仍发散——"consistent with the tighter stability margin at large effective depth"（§5.3）。
4. **机制证据**：初始化后 10 步，循环增量的 pairwise cosine 在 [0.027, 0.995]，非共享栈为 [−0.037, 0.034]（Fig. 3，L = 12、N = 64、d = 768）；训练 20,000 步后仍为正，早/中步 0.3–0.6，末步近零或负（Fig. 6/8）。
5. **与其他方案比较**：**只比较了 ε ∈ {1, 1/√N, 1/N}**；DeepNet/DeepNorm、Fixup、ReZero 仅在相关工作提及，未做实验；Post-Norm 留作未来工作；sandwich norm 全文未提。

## 失败模式与警告
- 1/√N 对循环网络不够（Fig. 2e）；反过来 ε = 1/L 对非共享深栈过强，R 随 L 衰减到 0.2–0.5（Fig. 2c，图读）——缩放过头也有代价。
- Theorem 2 只在 εN = O(1) 前提下成立；sqrt 缩放违反该前提，理论不预测其最优 LR 偏移的方向与大小（§5.2）。
- L = 48、N = 8 在 η0 > 2e-3 发散（§5.3）；Fig. 5 已剔除最终 loss > 4 的 run。
- 理论是 ReLU 单层 MLP，实验是 SwiGLU + attention；未考虑优化器状态、归一化变体、特征学习（Limitations）。
- 单种子；宽度轴未缩放；未匹配 FLOP。

## 对我们实验的直接启示
**应该照搬**
- 残差缩放：每条分支乘 ε = λ·r^{−1}·(L/L_ref)^{−1/2}，λ = 1。我们固定 L = 8，取 L_ref = 8 ⇒ **ε = 1/r**；r = 1 时 ε = 1，即标准 pre-norm 模型，基线不受影响。
- LR 协议：在 r = 1 上扫 LR（他们的网格是基准 ×{0.4, 0.6, 0.8, 1, 1.2, 1.6, 2.4, 3.2}），原样迁移到 r ∈ {2, 4, 8}；他们的锚点 η0 = 1.25e-3 是 d = 768、L = 12、AdamW 的值，仅供参考。
- AdamW (0.9, 0.95)、ε 1e-8、WD 0.1；WSD 500 步 warmup / 末尾 1,000 步线性衰减（2.5% / 5%）——与我们的 WSD 计划一致；发散判据 loss > 4。
- 诊断：记录每个 loop step 的残差范数轨迹与 N×N 增量 cosine 矩阵（Fig. 6/7 方法），用于验证 1/r 在我们的注入架构下是否仍需要。
**应该避免/改进**
- 他们不匹配 FLOP；φ 必须同时给 compute-matched 与 param-matched 两种视角。
- 单种子、无宽度缩放、无 prelude/coda、无输入注入、无截断反传；这些都是我们要覆盖而本文无证据的轴。
- 1/N 下输出更新量并非严格常数（约 4× 范围），LR 迁移在 r ≤ 8 可用但非精确；建议在最大宽度、r = 8 上再做 3 点 LR 验证。
- 我们"每次循环注入输入"改变了残差累积机制（注入相当于每次循环重新加入 h_0），本文理论未覆盖；Theorem 1 的 Θ(N²) 累积来自共享权重的相干性，注入不消除它，所以 1/r 仍应保留，但 λ 可能需重调。
**三问**
- (a) **N = 循环次数，L = 独立层数，都不是展开层数**。原文："N is the loop count (the number of times the shared layer is applied)"（§3.1）；"L unique layers looped N times"（摘要）；"1/N controls the within-layer loop correlation, and 1/√L controls the across-layer variance"。对 8 个独立层循环 r 次：N = r，L = 8。若把 N 取成展开层数 8r，会把独立层之间的随机游走（应为 1/√L）当作相干累加处理，r = 1 时也会把 ε 压到 1/8，而 Fig. 2c 显示过强缩放会让残差流萎缩。对"中间块循环 + prelude/coda"，原文明确未覆盖（"heterogeneous multi-stage looped architectures remains future work"）；一个与原文精神一致的扩展是：只对循环块内的分支乘 1/r，prelude/coda 分支不乘，√L 项用总独立层数——这是我们的推断，需实验验证。
- (b) 支持**跨 r 迁移**（Fig. 5，N = 1 → 8 最优 LR 不变）和**跨 L 迁移**（m_L^{−1/2}，L = 12 → 48）；**不支持跨宽度迁移**——所有模型 d = 768，"Width scaling terms are set to one"。我们从 20M 到 160M 的变化几乎全在宽度（448 → 1280，L 固定），本文对此没有证据；要么加 μP 宽度规则，要么在每个宽度上各扫一次 r = 1 的 LR（便宜）再按本文规则跨 r 迁移。
- (c) 本文无多 epoch 内容，不能回答。

## 留白
- Conclusion："Natural next steps include scaling up to production-sized models and extending the theory to Post-Norm architectures [26], where the interaction between normalization placement and weight sharing may yield a different scaling regime."
- Limitations："Our theoretical analysis models the looped block as a shared-weight MLP, abstracting away the multi-head attention mechanism present in Transformers."
- Limitations："The formal results further assume ReLU activations, whereas our experiments use SwiGLU; the predicted Θ(N²) growth and 1/N scaling nonetheless hold empirically (Figures 3, 6)."
- Limitations："More broadly, the analysis does not account for optimizer state dynamics, normalization variants, or data-dependent feature learning."
- Limitations："On the empirical side, our experiments cover a finite set of model sizes, loop counts, and training budgets."
- Limitations："Validation on larger models, longer training horizons, and heterogeneous multi-stage looped architectures remains future work."
- Limitations："In addition, each reported language-modeling result uses a single random seed; multi-seed replicas of the full configuration sweep were not conducted due to computational constraints."
- App. A.5："If the layer contributions are less coherent than (15), the upper bound can be loose; this is the sense in which less coherent regimes can tolerate larger learning rates."
- §5.1："consistent with the theorem providing an O(·) upper bound rather than a tight scaling"
