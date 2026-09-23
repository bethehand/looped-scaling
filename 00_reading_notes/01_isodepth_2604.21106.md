# 阅读笔记 01 — How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models（arXiv 2604.21106, v3）

## 基本信息
- 作者/机构：Kristian Schwethelm（TUM, Chair for AI in Healthcare and Medicine）；Daniel Rückert（TUM；Imperial College London；MCML）；Georgios Kaissis（Hasso Plattner Institute, University of Potsdam）。
- 日期/版本：v1 2026-04-22；v2 2026-04-27（arXiv comments："added case studies on truncated-BPTT and hyperconnections"）；v3 2026-05-07（"substantially refined framing + minor corrections"）。本笔记基于 v3。
- 发表状态：PDF 首页标注 "Preprint."；cs.LG / cs.CL；CC BY 4.0。未见任何会议或期刊信息。
- 代码与权重：**全文未给出代码或权重链接**（正文与附录中唯一的 GitHub 链接是参考文献里的 nanochat 仓库 github.com/karpathy/nanochat 以及 FineWeb-Edu 数据集页）。实现 "builds on nanochat [17]"（Sec. 4.1）。
- 算力：A100-80GB 与 H100-80GB 混合，整个项目（含失败与探索性运行）约 5,000 GPU-hours（Appendix B）。

## 他们匹配了什么
- 主匹配量：**训练 FLOP（iso-FLOPs）+ 有效深度（iso-depth）**。四种架构 r∈{1,2,4,8} 都执行 20 层前向（ℓ_eff = ℓ_prelude + r·ℓ_recur + ℓ_coda = 20），同宽度下每 token 训练与推理 FLOP 相同（仅差注入层开销），唯一参数量随 r 降到 ∼61%/∼41%/∼31%（Sec. 3, Sec. 3.2）。每个预算 C 下扫宽度找 compute-optimum，比较的是 compute-optimal 前沿 L*_r(C)（Sec. 4.2）。因 FLOP/token 相同，固定 C 下各架构训练 token 近似相同（"every variant trains on approximately the same token count D ≈ C/F_train"，Sec. 3.2 脚注 2）；looped 模型因注入层开销 token 略少（Table 6 说明）。部署 FLOP 在同宽度下也相同，这是 iso-depth 设计的副产品。
- 参数定义：N = "the transformer's total non-embedding parameters"（Sec. 3.2），**不含 embedding 与 unembedding**；分成 N_once = (ℓ_prelude + ℓ_coda)·n_b 与 N_rec = ℓ_recur·n_b + n_i（**注入层计入 N_rec**）。
- FLOP 公式（Sec. 3.2）：n_b = 12d² 为单层参数量（4 个 d×d attention 投影 + d→4d→d MLP），n_i = 2d² 为注入层，d ≡ d_model。
  - Eq. (1)：F_fwd(r) = 2(N_once + r·N_rec) ≈ F_fwd(1) = 2·ℓ_eff·n_b = 2N；训练 F_train(r) = 3·F_fwd(r)（脚注 2）。
  - 注入层开销 = 2r·n_i/(2ℓ_eff·n_b) = r/120，即 r=2/4/8 时 1.7%/3.3%/6.7%（脚注 2）。
  - 报告的 FLOP 还包含无参数的 attention matmul（训练时 12·h·q·T，h 头数、q 头维、T 序列长）与 unembedding matmul（训练时 6·d·V），"both fixed across architectures at matched width. For simplicity we exclude them from the equations below."
  - 脚注 1：N(r) = (4 + 16/r)·n_b + n_i = (48 + 192/r + 2)·d²，r∈{2,4,8} 时为 {146, 98, 74}·d²，r=1 为 240·d²；s=10 时 N ∈ {98.3, 59.8, 40.2, 30.3} M。
  - 截断反传 Eq. (6)（Appendix G.1）：F_train^trunc(r) = (2(r − r_bwd) + 6·r_bwd)(ℓ_recur·n_b + n_i) + 6(ℓ_prelude + ℓ_coda)·n_b；节省约 30% 训练 FLOP，节省的算力换成更多 token，经验中位数 D_trunc/D_full = 1.315。

## 模型与训练设定
- 架构（Sec. 3.1, Appendix C.1, Table 4）：nanochat 风格 decoder-only；ℓ_eff = 20；(ℓ_prelude, ℓ_coda) = (2, 2)（r>1）；ℓ_recur = 16/r ∈ {8, 4, 2}；宽度 d_model = 64·s；d_head = 128，n_head = d_model/128；RMSNorm pre-norm（ε = 1e−6，可学习 scale 初始化为 1）；RoPE base θ = 10,000；QK-norm（functional RMSNorm on q, k）；squared-ReLU MLP，hidden = 4·d_model；无 bias；无 dropout；**untied** wte / lm_head；Llama 2 tokenizer，vocab 32,008 pad 到 32,064；logit softcap z = 15·tanh(logits/15)（fp32）；全因果注意力，无滑窗；FlashAttention-2（A100）/-3（H100）。
- 模型级 RMSNorm 三处：token embedding 之后、**每次循环迭代结束时**（"so the state handed to the next iteration or to the coda has controlled scale"）、lm_head 之前（Appendix C.1）。除此之外**未提任何残差缩放**。
- 输入注入 Eq. (5)：u^(t) = W_inject·[e ∥ h^(t)]，W_inject ∈ R^{d×2d}，初始化为 [I ∥ 0]（使 u^(0) ≈ e）；e 为 prelude 输出（循环中不变）；**初始状态 h^(0) = e**（既非零也非随机）。注入消融（Appendix C.2, Table 5；s=10, r=4, C=1e18）：Linear 955M tokens → 2.793 nats；Additive（u = h + e，h^(0) = 0）973M → 2.797；Passthrough（无注入）973M → **7.400（训练失败）**；Hyperconnections K=2 973M → 2.757。
- 初始化（Appendix C.1）：token embedding N(0,1) 后 cast 到 bf16；LM head N(0, 1e−3)；attention/MLP 权重 U(−a, a)，a = √3/√d_model；mlp.c_proj 用 a = √3/√(4·d_model)；注入层 [I ∥ 0]；RMSNorm scale = 1。
- 循环次数：**架构固定 r，不采样**。反传：主网格 **full BPTT**；截断反传只作为 case study（Sec. 5.1, Appendix G.1）：r_bwd = ⌈r/2⌉ ∈ {1, 2, 4}，第 i 次循环后对所有 i < r − r_bwd 的状态 detach。
- 优化（Sec. 4.1, Appendix D）：矩阵参数用 MuonH，embedding/unembedding/norm 用 AdamW；weight decay = 0（"first-order no-op under MuonH's Frobenius-sphere constraint"）；base MuonH LR η_base = 0.014；AdamW base LR 0.3（embedding）、0.004（unembedding）、0.005（norm）；LR 经 muP + HyperP 跨宽度/batch/训练长度迁移（HyperP 数据修正项 D^−0.32）；调度 "each LR linearly decayed to 10% of its peak"；**全文未提 warmup，也未提 WSD/cosine**。LR sweep（Appendix D.1, Figure 6）：s=10、tokens/param = 10（∼1B tokens）、B = 262,144 tokens，每架构 8 个 LR ∈ [0.008, 0.024]，两架构共同最优 η* ≈ 0.014；宽度迁移 s∈{8,10,14} 最大 regret 0.004 nats（s=8 looped）；s=18 looped 0.014 vs 0.012 为 2.473 vs 2.476 nats；数据迁移 ratio {10,20,40} regret < 0.005 nats（D.2）。
- Batch：B = 262,144 tokens（256K；在 {256K, 512K, 1M} 中"uniformly lower loss for both architectures"，D.1）；s=34 外推实验改为 524,288（Appendix I）。序列长 2,048（打包成 2,049，Sec. 4.1）。
- 数据：FineWeb-Edu 子集，四架构看到完全相同的数据流；验证集 held-out FineWeb-Edu（nats）。
- 规模网格（Sec. 4, Appendix E Table 6）：6 个预算 C ∈ {4.64e17, 1.00e18, 2.15e18, 4.64e18, 1.00e19, 2.15e19} FLOPs（∼50×）；宽度 s ∈ {6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 34}（d_model 384–2,176）；每架构 29 runs，共 116 runs。唯一非嵌入参数 N(s, r)（M，顺序 r=1/2/4/8）：s=6: 35.4/21.5/14.5/10.9；s=8: 62.9/38.3/25.7/19.4；s=10: 98.3/59.8/40.2/30.3；s=12: 141.6/86.1/57.8/43.7；s=14: 192.7/117.2/78.7/59.4；s=16: 251.7/153.1/102.8/77.6；s=18: 318.6/193.8/130.1/98.2；s=20: 393.3/239.2/160.6/121.3；s=24: 566.3/344.5/231.2/174.6；s=28: 770.8/468.9/314.7/237.7；s=34: 1136.5/691.4/464.1/350.4。r=1 各 (s, 预算) 的训练 token（B，按预算从小到大）：s=6: 0.98, 2.10；s=8: 0.64, 1.36, 2.95；s=10: 0.45, 0.97, 2.08, 4.49；s=12: 0.34, 0.72, 1.55, 3.34, 7.13；s=14: 0.56, 1.20, 2.59；s=16: 0.96, 2.07, 4.43；s=18: 1.70, 3.62, 7.78；s=20: 1.42, 3.05；s=24: 2.20, 4.71；s=28: 3.58；s=34: 2.52。
- Case-study 网格（Sec. 5）：r∈{2,4,8} 在四个较低预算（4.64e17–4.64e18）重跑，复用不变的 r=1；hyperconnections 联合拟合共 83 runs（vs 主拟合 116，Appendix G.2）。
- 种子：**全文未提及种子或重复运行**；116 = 29 × 4 表明每个 (预算, s, r) 只跑一次。

## 函数形式与拟合方法
- Chinchilla 基线 Eq. (2)：L(N, D) = E + A·N^−α + B·D^−β。
- 联合律 Eq. (3)：L(N_once, N_rec, D, r) = E + A·(N_once + r^φ·N_rec)^−α + B·D^−β。符号：L 验证损失（nats）；E 不可约损失；N_once 仅用一次的 prelude+coda 参数；N_rec 循环块参数（含注入层）；r 循环次数；φ ≥ 0 为 recurrence-equivalence exponent，r^φ ≥ 1 放大循环参数的贡献；N_eff ≡ N_once + r^φ·N_rec；D 训练 token；A, B, α, β 拟合常数。参考点：φ = 1 完全等价（同训练 FLOP 下与非循环同损失），φ = 0 无增益；全循环模型对应 N_once = 0。作者强调 "the law is fit on iso-compute runs, so C enters only implicitly through the data. The non-looped reference matches loss at the same D, not at the same compute."（Sec. 3.3）
- 拟合目标 Eq. (4)：log 空间参数化（a = log A, b = log B, e = log E），最小化 Σ_i Huber_δ(LSE(a − α·log N_i, b − β·log D_i, e) − log L_i)，δ = 1e−3；L-BFGS-B，**500 个随机起点取最优**；box 约束 a, b ∈ [−5, 35]，α, β ∈ [0, 2.5]，e ∈ [−3, 2]，联合拟合另有 φ ∈ [−3, 3]；每次最多 10,000 次迭代（Appendix F "Optimisation details"）。全部 116 runs 参与联合拟合，**未见排除任何点**。
- 置信区间（Appendix F.2）：block bootstrap，对 (预算, 架构) 6×4 个 cell 有放回重采样 200 次，每次重拟合，取 φ 的 2.5/97.5 百分位；不对受限形式做 bootstrap。
- 每架构 Chinchilla 拟合（Table 1；A, B 只保留 2 位有效数字，因 "only loosely identified under iso-compute designs [33]"）：r=1: A=58, α=0.251, B=150, β=0.267, E=1.56, Huber 5.84e−5, R²=0.9979；r=2: 33, 0.216, 910, 0.365, 1.60, 5.27e−5, 0.9983；r=4: 23, 0.191, 1300, 0.388, 1.56, 6.81e−5, 0.9976；r=8: 41, 0.235, 780, 0.362, 1.69, 5.33e−5, 0.9980。
- 联合拟合（Table 2；**A, B 未报告**）：φ 自由：α=0.199, β=0.369, E=1.57, φ=0.459 [0.41, 0.53], R²=0.9972；受限 φ=0：α=0.227, β=0.390, E=1.71, R²=0.9858；受限 φ=1：α=0.218, β=0.410, E=1.66, R²=0.9552。
- 残差（Appendix F.1, Table 7）：max |resid| = 0.036 nats，pooled RMSE = 0.010 nats，各 r 的 RMSE 0.009–0.011，均值在 ±0.006 内（r=1: −0.001/0.018/0.009；r=2: +0.004/0.036/0.011；r=4: −0.006/0.023/0.011；r=8: +0.001/0.029/0.010，格式 mean/max|resid|/RMSE）。半区重拟合（F.3）：低预算半区（C ≤ 2.15e18，n=56）φ=0.44，高半区（C ≥ 4.64e18，n=60）φ=0.49。
- Case-study 重拟合（Table 3，四个低预算，同目标与约束）：Baseline（full BPTT, linear injection）α=0.266, β=0.457, E=1.85, φ=0.453, R²=0.9959；Truncated BPTT（含 r=2）0.255, 0.484, 1.87, **0.380**, 0.9827；Truncated BPTT（不含 r=2）0.265, 0.492, 1.89, 0.373, 0.9958；Hyperconnections 0.308, 0.464, 1.93, **0.646**, 0.9900。
- 作者自述的形式失效 / 拟合不稳定情形（原话）：
  - "The R² values in Table 2 look uniformly high because most variance across the runs comes from the compute-budget axis (cross-architecture loss spans only ∼0.1 nats). On that scale, small drops from 0.997 (free φ) are substantial."（Sec. 4.3）
  - "iso-FLOPs sampling places each architecture in a different region of the (N,D) plane, where α and β are weakly identified [33]."（Sec. 4.2）
  - "The r^φ form is a pre-saturation local approximation valid in this range and does not include the architectural ceiling."（Sec. 6）
  - "When some r outperform the baseline and others fall below it, no single φ captures both directions and the fit degrades."（Sec. 6）
  - "Most of the residual mass sits at r=2 (r_bwd=1), where the joint law systematically under-predicts loss."（Appendix G.2；去掉 r=2 后 R² 0.983 → 0.996，φ 不变）
  - "φ obtained with different baselines are not comparable."（Sec. 6）

## 主要结论
- φ = 0.46（点估计 0.459，95% CI [0.41, 0.53]，"no resample reaching φ=0 or φ=1"）；r=4 时循环块贡献 4^0.46 ≈ 1.89 倍其唯一参数（约 47% 的完全等价），r=8 时 ≈ 2.6 倍（Sec. 4.3, Sec. 6）。等价规模（Appendix F.4）：N(r=4)/N(r=1) = 8/20 = 0.40，N_eff 比 ≈ (4 + 4^φ·4)/20 ≈ 0.58 → "a 410M r=4 model performs on par with a 580M non-looped model, but incurs the training cost of a 1B non-looped one"。
- Compute-optimal 差距（Sec. 4.2）：r=2 落后 [0.03, 0.06] nats，r=4 [0.05, 0.08]，r=8 [0.09, 0.12]，随 r 单调增大；低预算差距更大，最大两个预算之间变化 Δ ≤ 0.006 nats。
- Compute-optimal 分配（Figure 3 图例，**仅从图中可读**）：N* ∝ C^0.57（r=1）、C^0.70（r=2）、C^0.76（r=4）、C^0.67（r=8）；D* ∝ C^0.53、C^0.42、C^0.37、C^0.44。looped 最优点更宽（但唯一参数仍更少）、token 更少。
- 截断反传（Sec. 5.1, Table 3）：所有 run 验证损失都下降，但 φ 从 0.45 → 0.38（去掉 r=2 为 0.37）；归因于早期循环拿不到准确梯度；结果是 compute-optimal 宽度更宽、每 token 推理 FLOP 更高（Appendix G.3）："the joint law attributes the validation-loss improvement to a larger token budget and a wider compute-optimal model that offsets the per-loop capacity loss but raises per-token inference cost."
- 循环位置：**未变化**（固定 prelude/coda = 2/2 的中间循环），列为 limitation。
- Hyperconnections（K=2 lanes，full BPTT，Sec. 5.2）：φ 0.45 → 0.65；最优宽度变窄、推理 FLOP 降低；r=2 在部分预算追平或超过 r=1。
- 训练 FLOP 匹配 vs 参数匹配：全部结论在训练 FLOP 匹配（iso-FLOP + iso-depth）下得出；φ < 1 直接意味着 compute-matched 下 looped 落后（Sec. 3.3）。参数匹配的问题被归给 Parcae（Appendix A）。
- 外推（Sec. 4.2, Appendix I, Table 10）：s=34（d_model = 2,176）、47B tokens、∼4e20 FLOPs（∼20× 网格顶部）、B = 524,288、匹配 token（非 FLOP，looped 多 ∼3% FLOP）：r=1 2.047 vs r=4 2.108 nats，差 +0.061，落在 [0.05, 0.08] 内。
- 测试时循环外推：**未做实验**。Appendix A 引用 Parcae 的饱和指数 L(T) = L_∞ + Z·exp(−zT/μ_rec) 与 Ouro 的结论，认为 "effective inference depth in trained looped LMs concentrates near the training depth distribution rather than extrapolating freely past it. We therefore treat r as an architectural, not a test-time, scaling axis."
- 下游（Appendix H，continuation loss，五个轴）：parametric knowledge 跟随验证损失排序，r=8 落后 r=1 达 0.28 nats；reading comprehension r∈{2,4} 与 r=1 持平，r=8 落后 0.05–0.18 nats；compositional symbolic 大致持平（looped 在 BigBench Dyck 领先：C=2.15e19 时 r=8 3.264 vs r=1 3.902，Table 9）；reasoning primitives 与 math word problems "unresolvable at our scale"（math 各 r 压缩在 ∼0.1 nats 内）。外推点差距（Table 10，r=4 − r=1）：parametric +0.108，reading +0.050，math +0.028，reasoning primitives +0.095，compositional −0.004。

## 失败模式与警告
- Passthrough（无输入注入）在 s=10, r=4, 1e18 FLOPs 训练失败（7.400 nats）："some form of injection is essential at this scale"（Appendix C.2）。
- 截断反传下 r=2（r_bwd=1）系统性欠拟合，是最大残差来源："the shared block receives a direct gradient only from the second recurrence... this indirect signal is evidently too weak to train the looping mechanism"（Sec. 5.1）。
- 截断反传的损失下降是 token 侧收益（D_trunc/D_full 中位数 1.315）+ 更宽模型，而非循环机制变好："A method that substantially lowers φ should therefore be treated cautiously"（Sec. 6）。
- 评测陷阱：小规模下 accuracy 接近随机、双峰，改用 continuation loss（Appendix H.5）；CoQA 因 prompt 超 2,048 上下文降到 1-shot；reasoning 类任务 "cannot drive architectural decisions at our scale"（H.2）。
- 拟合陷阱：A, B 与 α, β 在 iso-compute 设计下识别性差；跨架构差异仅 ∼0.1 nats 时 R² 无区分力，须看 Huber/RMSE 与 Figure 1（右）；hyperconnections 若也加到 r=1 baseline，"would likely shift the calibration of φ downward"（Sec. 5.2）。
- 外推点因 s=34 离 r=4 的最优点比离 r=1 的更远，差距混入宽度次优惩罚（脚注 3）。
- 未提及训练发散或 loss spike；未提 warmup；无种子重复，训练噪声水平不可知。

## 对我们实验的直接启示
**应该照搬**
- 架构与注入：prelude/coda 各 2 层的中间循环模板；线性拼接注入 W_inject ∈ R^{d×2d} 初始化 [I ∥ 0]、h^(0) = e；每次循环迭代末尾一个 RMSNorm；embedding 后与 lm_head 前各一个 RMSNorm；QK-norm、RoPE θ=10,000、squared-ReLU、无 bias、logit softcap 15、untied embeddings。注入层参数计入 N_rec、FLOP 计入 r·n_i。
- FLOP 记账：Eq. (1)/(6)，并把 6·d·V（unembedding）与 12·d·T（attention）加进报告的 FLOP。
- 拟合协议：log 空间 Huber（δ = 1e−3）+ LSE 参数化；L-BFGS-B 500 起点；box 约束 a, b ∈ [−5, 35]，α, β ∈ [0, 2.5]，e ∈ [−3, 2]，φ ∈ [−3, 3]；10,000 迭代上限；同时报告 φ 自由 / φ=0 / φ=1 三种拟合的 Huber 与 R²；对 (预算, r) cell 做 200 次 block bootstrap 报 95% CI；低/高预算半区重拟合检验漂移；报告每个 r 的残差 RMSE。
- 超参流程：先在参考宽度对每个 r 分别扫 LR，确认最优一致（regret < 0.005 nats）后共用一套。他们的数值：MuonH η_base = 0.014；AdamW 0.3 / 0.004 / 0.005（embedding / unembedding / norm）；WD = 0；线性衰减到峰值的 10%（无 warmup）；B = 256K tokens（{256K, 512K, 1M} 中最优）；seq 2,048；LR sweep 用 tokens/param = 10。若我们用 AdamW 而非 MuonH，数值不可直接搬，但流程可搬。
- 规模：作者称每架构 "∼20 runs across our four lower budgets" 足以测 Δφ；每预算 ≥ 4–5 个宽度。
- 评测用 continuation loss 而非 accuracy。

**应该避免或改进**
- 不要把截断反传的"节省 FLOP → 更多 token"与"截断本身"混在一起：若目标是量化截断对 φ 的影响，应同时跑 iso-token（同 D）与 iso-FLOP（D 放大 ∼1.3×）两组，否则 Δφ 与 token 增益纠缠。
- 避免 r=2 + k=1 作为主要数据点（系统性欠拟合）；若扫 k，至少覆盖 k ≥ 2。
- 不要只报 R²；跨架构差异只有 ∼0.1 nats，必须报 bootstrap CI 与残差。
- 不要用无注入（passthrough）循环块。
- 不要期望 20M–160M 规模在 reasoning 下游任务上看到 r 的信号。
- 他们没有种子重复；我们在更小的网格里应至少对若干 cell 跑 2–3 个种子估计噪声，否则 φ 的 CI 只反映 cell 采样而非训练噪声。

**(a) φ 的测量条件**：r ∈ {1, 2, 4, 8}（r=1 为非循环基线；r 固定不采样），ℓ_eff = 20，prelude/coda = 2/2，d_model 384–2,176（s = 6–34），C ∈ [4.64e17, 2.15e19] FLOPs，**full BPTT**。截断反传的 φ = 0.38 只在四个低预算（4.64e17–4.64e18）、r_bwd = ⌈r/2⌉ ∈ {1,2,4} 下测得，且节省的 FLOP 已换成 ∼1.3× token。作者自述该设计 r 上限 r_max = 16。
**(b) 网格里缺的、我们要补的**：① 全栈循环（N_once = 0）vs 中间循环——他们只有 (2,2) 一种；② 截断反传的 k 扫描（他们只有 k = ⌈r/2⌉）与 iso-token 对照；③ 更小的模型（他们最小 10.9M 唯一参数、预算 ≥ 4.64e17 FLOPs；我们 20M–160M 大致对应他们 s = 6–12 的下半段）；④ 种子重复；⑤ 固定 r vs 采样 r；⑥ 不同 prelude/coda 大小；⑦ 测试时 r 外推。
**(c) FLOP 公式能否沿用**：可以直接沿用 Eq. (1)/(6)；全栈循环时令 N_once = 0（若保留注入层则 N_rec 含 n_i）。注意在我们的规模（d ≈ 256–768，V ≈ 32K）6·d·V 的 unembedding 项与 6N 同量级（例如 d = 512、V = 32,000 时 6dV ≈ 9.8e7 FLOPs/token，而 N = 20M 时 6N = 1.2e8，本笔记推算），不能像他们那样只在公式中省略，须显式加入且不随 r 变化；若注入方式不是 d×2d 线性层（如 Parcae 式对角注入），n_i 需改为 O(d)。

## 留白
- "We fix a single architecture configuration: 20 effective layers with (ℓ_prelude, ℓ_coda)=(2,2) following the prelude-recur-coda template of Geiping et al. [4]. Different depth allocations or prelude/coda sizes may shift φ, which we leave to future work."（Sec. 6）
- "Our iso-depth setup also caps recurrences at r_max=16. The r^φ form is a pre-saturation local approximation valid in this range and does not include the architectural ceiling."（Sec. 6）
- "The joint law also assumes that each additional recurrence must either consistently help (φ>1) or consistently hurt (φ<1) compared to the non-looped baseline... Additionally, φ obtained with different baselines are not comparable."（Sec. 6）
- "The link between φ and reasoning quality at scale is therefore empirically untested... A reasoning-heavy pretraining mix [13] might surface architectural differences on reasoning tasks at our budgets and offer a specialised Δφ axis."（Sec. 6）
- "Other methods worth quantifying include shrinking the shared fraction (larger prelude/coda), per-token adaptive compute [6, 7, 13, 14], retrofitting pretrained non-looped models [8, 9], and training with a diffusion objective in place of unrolling the loops [34]."（Sec. 6）
- "We apply our framework to truncated backpropagation and hyperconnections, leaving the others to future work."（Sec. 2）
- "Their diagonal-injection layer remains untested in our framework."；"A scaling law that matches parameters, FLOPs, and memory simultaneously remains an open direction."（Appendix A）
- "Until per-token adaptive compute delivers wall-clock gains at inference, it does not raise the worth of a recurrence beyond what φ already captures at training time."；"Until depth extrapolation works, it likewise does not raise the worth of a recurrence beyond what φ captures at training time."（Appendix A）
