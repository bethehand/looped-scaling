# 08 · Hoffmann et al. — Training Compute-Optimal Large Language Models（Chinchilla，arXiv 2203.15556）

> 阅读版本：arXiv v1（2022-03-29，唯一版本），全文（正文 + Appendix A–J）来自 arxiv.org/html/2203.15556v1 与 PDF。文末附一条外部复现（Besiroglu et al. 2024，Porian 的参考文献 [9]）的对照，已明确标注为非原文。

## 基本信息
- 作者 / 机构：Jordan Hoffmann*、Sebastian Borgeaud*、Arthur Mensch*、Elena Buchatskaya、Trevor Cai、Eliza Rutherford、Diego de Las Casas、Lisa Anne Hendricks、Johannes Welbl、Aidan Clark、Tom Hennigan、Eric Noland、Katie Millican、George van den Driessche、Bogdan Damoc、Aurelia Guy、Simon Osindero、Karen Simonyan、Erich Elsen、Jack W. Rae、Oriol Vinyals、Laurent Sifre*（DeepMind；* 同等贡献）。
- 日期 / 版本：v1 2022-03-29（无后续版本）；DeepMind 报告编号 001。
- 发表状态：NeurIPS 2022（Hägele 引为 "Advances in Neural Information Processing Systems, 35:30016–30030"；Porian 引为 "An empirical analysis of compute-optimal large language model training", NeurIPS 2022）。
- 代码 / 模型：无代码发布；模型不公开（Table A8："We will not make this model available publicly"）。

## 方法要点
- 问题：在 FLOPs(N,D) = C 约束下最小化 L(N,D)（Eq. 1），得 N_opt(C)、D_opt(C)。用 >400 个模型（"from under 70M to over 16B parameters, and trained on 5B to over 400B tokens"），每个配置训练多个长度。
- **Approach 1（固定 N、变 D）**：70M–10B+，每个 N 训 4 个长度，"decaying the learning rate by a factor of 10× over a horizon (measured in number of training tokens) that ranges by a factor of 16×"；对每条曲线平滑（高斯窗 10 步，App. D.1）并插值；在 1500 个对数等距 FLOP 值上取最低 loss 的运行 → 包络；拟合 N_opt ∝ C^a、D_opt ∝ C^b：a = 0.50, b = 0.50。脚注 4："all selected points are within the last 15% of training. This suggests that when training a model over D tokens, we should pick a cosine cycle length that decays 10× over approximately D tokens"。
- **Approach 2（IsoFLOP）**：9 个 FLOP 预算 6×10^18 – 3×10^21，模型最大 16B，cosine 周期与目标 token 数匹配（脚注 7），对每条 IsoFLOP 曲线（最终平滑 loss vs N）拟抛物线取最小值 → a = 0.49, b = 0.51。要求"trained a diverse enough set of model sizes to see a clear minimum"。
- **Approach 3（参数律）**：对 Approach 1 & 2 的**全部最终 loss** 拟合 L̂(N,D) = E + A/N^α + B/D^β（Eq. 2/5）→ a = 0.46, b = 0.54。
- **cosine 周期长度（App. B, Fig. A1）**：周期取目标步数的 1、1.1、1.25、1.5、2、5 倍；"overestimating the number of training steps beyond 25% leads to clear drops in performance"。衰减倍数：10× 与衰到 0 差别小，10× 略好；5× "clearly worse"（脚注 9）。
- **20 tokens/param**：论文未写成规则，而由 Table 3（Approach 1）读出：400M → 8.0B tokens、1B → 20.2B、10B → 205.1B、67B → 1.5T、175B → 3.7T、280B → 5.9T、520B → 11.0T、1T → 21.2T、10T → 216.2T（ratio ≈ 20）。Approach 2/3 见 Table A3：1B → 20.0B / 27.1B，10B → 219.5B / 410.1B，67B → 1.7T / 4.1T。Chinchilla 本身 70B / 1.4T = 20。
- 置信区间：Table 2 括号内为 10th/90th 百分位，"estimated via bootstrapping data (80% of the dataset is sampled 100 times)"。
- 补充数据集（App. C, Table A2，IsoFLOP 法）：C4 a=0.50, b=0.50；GitHub a=0.53, b=0.47。

## 实验设定
- 规模：Table A9 列 50 个配置 44M–16,183M（d_model 512–5120，ffw = 4·d_model，kv 64（≤306M）或 128，heads 8–40，层数 8–49），"Many models have been trained multiple times, for a different number of training steps"。
- 数据：MassiveText（Table A1：MassiveWeb 45%、Books 30%、C4 10%、News 10%、GitHub 4%、Wikipedia 1%），全部 <1 epoch（Wikipedia 与 MassiveWeb 在 1.4T 时 >1 epoch）。词表：SentencePiece 32,000，不做 NFKC（Table A8）。
- Loss：**平滑后的训练 loss**（脚注 2："an unbiased estimate of the test loss, as we are in the infinite data regime"）。
- 优化器：Chinchilla 用 AdamW（Gopher 用 Adam）；App. G 显示 "independent of the learning rate schedule, AdamW trained models outperform models trained with Adam"；脚注 8：AdamW 约在 cosine 周期 80% 处才超过 Adam。400 模型扫描本身的优化器未明说。
- 学习率 / 调度：Approach 1 最大 LR 从 2×10^-4（最小模型）到 1.25×10^-4（最大模型），cosine 10× 衰减，周期 ≈ 训练步数（App. D.1）。D.4 对照实验：batch 0.5M tokens、LR 1.5×10^-4、10× 衰减。Chinchilla 70B：80 层、64 头、kv 128、d 8192、LR 1×10^-4、batch 1.5M→3M（中途翻倍）；Gopher 280B：LR 4×10^-5、batch 3M→6M（Table 4）。
- **未报告**：warmup 长度（全文无 "warmup"）、扫描运行的 batch size（D.2 仅称 "the number of steps to perform depends on the gradient batch size, for which we use well-tested heuristics"）、序列长度（Hägele 估算时假设 1024）。
- 硬件：TPUv3/v4，JAX + Haiku；bf16 计算、fp32 权重副本。
- **参数计法**（App. F）："We include all training FLOPs, including those contributed to by the embedding matrices, in our analysis. Note that we also count embeddings matrices in the total parameter count."
- **FLOP 计法**（App. F，前向；反向 = 2×前向；乘加计 2）：Embeddings 2·seq_len·vocab_size·d_model；Attention 每层：QKV 投影 2·3·seq·d_model·(key_size·num_heads)，Key@Query logits 2·seq²·(key_size·num_heads)，Softmax 3·num_heads·seq²，Softmax@query 2·seq²·(key_size·num_heads)，Final Linear 2·seq·(key_size·num_heads)·d_model；Dense 每层 2·seq·(d_model·ffw_size + d_model·ffw_size)；Final Logits 2·seq·d_model·vocab_size；总前向 = embeddings + num_layers·(attention + dense) + logits。Table A4 与 6ND 之比：73M 1.03、305M 1.10、552M 1.08、1.1B 1.04、1.6B 1.03、6.8B 0.99（"differences ... very small and they do not impact our analysis"）。Gopher 按此法 6.3×10^23（Rae et al. 报 5.76×10^23）。Sect. 3.3 求前沿时则用 FLOPs ≈ 6ND。

## 函数形式与拟合方法
- 形式（Eq. 2）：L̂(N,D) ≜ E + A/N^α + B/D^β。E = 数据分布熵（Bayes risk）；A/N^α = 函数逼近项；B/D^β = 单 epoch 有限步优化项（App. D.2 Eq. 9 的三项分解；假设第三项只依赖 D）。
- 目标（Eq. 3）：min_{A,B,E,α,β} Σ_runs Huber_δ( log L̂(N_i,D_i) − log L_i )；实际实现（Eq. 11）：min_{a,b,e,α,β} Σ Huber_δ( LSE(a − α log N_i, b − β log D_i, e) − log L_i )，A,B,E = exp(a),exp(b),exp(e)。
- 优化器：L-BFGS；**初始化网格**：α ∈ {0, 0.5, …, 2}，β ∈ {0, 0.5, …, 2}，e ∈ {−1, −0.5, …, 1}，a ∈ {0, 5, …, 25}，b ∈ {0, 5, …, 25}（5×5×5×6×6 = 4500 个起点），取最优；"We find that the optimal initialisation is not on the boundary of our initialisation sweep."
- **δ = 10^-3**："We find that using larger values of δ pushes the model to overfit the small compute regime and poorly predict held-out data from larger runs. We find that using a δ smaller than 10^-3 does not impact the resulting predictions."
- **排除点**：无显式排除，拟合用 Approach 1&2 的全部最终 loss；但 Sect. 3.4："the observed points (L,N,D) for low training FLOPs (C ⩽ 1e21) have larger residuals ... The fitted model places increased weight on the points with more FLOPs—automatically considering the low-computational budget points as outliers due to the Huber loss."
- **拟合值（Eq. 10）**：α = 0.34，β = 0.28，E = 1.69，A = 406.4，B = 410.7；a = 0.46 (0.454, 0.455)，b = 0.54 (0.542, 0.543)（Table 2）。
- **前沿闭式（Eq. 4）**：N_opt(C) = G·(C/6)^a，D_opt(C) = G^-1·(C/6)^b，G = (αA/(βB))^{1/(α+β)}，a = β/(α+β)，b = α/(α+β)。
- 作者报告的敏感性 / 不稳定（原话）："We account for possible local minima by selecting the best fit from a grid of initialisations."；"The third approach predicts even smaller models being optimal at larger compute budgets."；"As a consequence of the empirically observed negative curvature in the frontier C → N_opt (see Appendix E), this results in predicting a lower N_opt than the two other approaches."；"While there is significant uncertainty extrapolating out many orders of magnitude..."
- **外部对照（非原文）**：Besiroglu et al., "Chinchilla Scaling: A replication attempt", arXiv 2404.10102 (v2 2024-05-15)，从图中重建 240 个数据点后用同一流程（δ=10^-3、同一初始化网格）重拟得 A = 482.01 (±124.58)、B = 2085.43 (±1293.23)、E = 1.8172 (±0.03)、α = 0.3478 (±0.02)、β = 0.3658 (±0.02)；指出论文报告的 (0.454,0.455) 区间 "implausibly narrow"；Chinchilla 作者随后确认原因是 "averaging the Huber loss values over different data points instead of summing them ... the high loss scale of their optimizer caused their optimization to terminate early"，且 β 由 TeX 源码中的 0.2849 四舍五入成 0.28 亦带来约 13% 偏差。

## 主要结论
1. 三种方法一致：a ≈ b ≈ 0.5（Table 2：0.50/0.50、0.49/0.51、0.46/0.54），Kaplan 为 0.73/0.27；"for every doubling of model size the number of training tokens should also be doubled"。
2. Gopher 预算（5.76×10^23）下最优模型 40–70B（Approach 3 给 40B，Fig. 4）；Chinchilla 70B/1.4T 全面超过 Gopher 280B/300B：MMLU 5-shot 67.6% vs 60.0%（Table 6）、BIG-bench 65.1% vs 54.4%、Wikitext103 ppl 7.16 vs 7.75、LAMBADA 77.4 vs 74.5、RACE-h 82.3 vs 71.6。
3. cosine 周期必须与训练长度匹配，超出 >25% 即明显变差（Fig. A1）；Kaplan 用固定 130B 周期导致中途 loss 高估，"eventually contributes to the conclusion that model size should increase faster than training data size"（Sect. 2）。
4. 175B 模型的最优预算 4.41×10^24 FLOPs、>4.2T tokens；280B 约 10^25、6.8T；1T 参数需 10^26（Sect. 3.4）。
5. C4、GitHub 上 IsoFLOP 结论相同（0.50/0.50、0.53/0.47），"independent of the dataset as long as one does not train for more than one epoch"（App. C）。
6. 前沿存在负曲率（App. E, Fig. A5），意味着大预算下最优模型可能更小。
7. AdamW 优于 Adam、fp32 权重副本有益（App. G, Fig. A6/A7）。

## 失败模式与警告
- **所有预算共用一个 cosine 周期**（Kaplan 做法）→ 中途 loss 系统性高估 → 参数指数被抬高（Sect. 2）。
- **周期超调 >25%** 明显掉点（App. B）；**衰减不足（5×）**明显更差（脚注 9）。
- **Huber δ 过大**（>10^-3）→ 过拟合小算力区、对大运行外推差（App. D.2）；δ = 10^-3 下低 FLOP（≤1e21）点被当作离群值自动降权（Sect. 3.4）——即拟合实际上由大预算点主导。
- **局部极小**：必须用初始化网格（App. D.2）。
- **幂律假设可能失效**：log N_opt 在高算力处呈凹形（App. E），"we may still be overestimating the optimal size of large models"（Sect. 5）。
- **可比大规模运行仅两条**（Chinchilla、Gopher），无中间规模验证（Sect. 5）。
- **分解假设**：第三项 "only depends on D"（App. D.2）——对循环深度模型，优化难度可能随有效深度变化，需检验。
- **（外部）实现细节会毁掉拟合**：Huber 求平均而非求和 + L-BFGS 早停 → 参数错误且 CI 假窄（Besiroglu et al. 2024）；报告 β 时保留 ≥4 位小数。

## 对我们实验的直接启示
**应该照搬（含数值）**
- 形式 L = E + A/N^α + B/D^β；在 log 空间用 Huber（δ = 10^-3）**求和**；L-BFGS；初始化网格 α,β ∈ {0,…,2 步 0.5}，e ∈ {−1,…,1 步 0.5}，a,b ∈ {0,…,25 步 5}，取最优并检查最优起点不在网格边界；用 LSE 参数化（Eq. 11）。
- Bootstrap 给 CI：80% 子样本 × 100 次（Table 2），更稳妥可做 ≥1000 次全量有放回重采样；若 CI 窄到 ±0.001 应怀疑优化早停。
- 每个 N 的 D 跨度 ≥16×（Approach 1），并混入 IsoFLOP 式的点（多个 N 在同一 C 上），避免 N、D 完全共线。
- cosine 对照：周期 = 训练长度，衰减 10×（不要 5×）。
- 若用训练 loss，先做窗口 10 步高斯平滑；最好直接用固定验证集。
- 报告 FLOPs 时同时给 App. F 逐项值与 6ND 之比（Table A4 式）。

**应该避免**
- 只在 ratio ≈ 20 附近取点（β 不可辨识）；D 跨度 <4×。
- Huber 求平均、用默认收敛阈值不检查梯度范数；把两位小数的 (α, β) 直接拿去算 a = β/(α+β)。
- 忽略低 FLOP 点残差偏大的现象——在 20M–160M 这种全是"低 FLOP"的区间，δ = 10^-3 会让最小模型/最短预算的点被当离群值；应报告残差图，并做 δ ∈ {10^-4, 10^-3, 10^-2} 敏感性。
- 把嵌入参数排除/包含混用：Chinchilla 的 N 与 FLOPs 均含嵌入，若沿用其拟合值作对照必须用同一口径。

**(d) 嵌入与输出头的计法**：Chinchilla：N 含嵌入矩阵，FLOPs 含 embedding（2·seq·vocab·d）与 final logits（2·seq·d·vocab）及 attention 各项（App. F），并说明大模型下差异 ≤10%（Table A4）。Porian：N = 全部线性层（**不含输入嵌入、含输出头**），FLOPs = 6ND，且指出不计输出头会在小模型上把 FLOPs 低估 90%（其 Table 2）。两者共同点：**输出头必须计入 FLOPs**；分歧只在输入嵌入（Chinchilla 计、Porian 不计，理由是查表不产生矩阵乘 FLOPs）。对 20M–160M、词表 5 万的模型，嵌入矩阵（d=512 时 25.7M）可与非嵌入参数同量级，建议：(i) 拟合律中的 N 采用 Porian 口径（非嵌入 + 输出头），并同时记录含嵌入的 N 以便与 Chinchilla 表对照；(ii) FLOPs 不用 6ND，用 Chinchilla App. F / Hägele Fig. 14 逐项公式（含 head 与 attention），对循环模型把 num_layers 换成实际执行层数，并另外记录"唯一参数 N_unique"与"计算等效参数 N_compute = 共享块参数 × 循环次数 + 非共享 + head"两套 N，分别拟合——这正是循环深度研究要回答的问题。

**(e) 每个规模需要几个 token 预算点**：本文对每个 N 用 4 个长度、跨度 16×，再叠加 9 条 IsoFLOP 曲线，总计 >400 点拟 5 个参数；β 的可辨识性来自 D 的跨度而非点数。10/20/40 只有 3 点、4× 跨度，5 参数拟合下 β 将与 E、B 强相关。建议每个规模 ≥5 个预算、跨度 ≥16×（例如 5/10/20/40/80 tokens/param；下限 5 保证 D ≥ 5× warmup，见 Porian Fig. 2），WSD 下每增加一个分支只多付 20% 的该预算，5 个预算总量 80N + (1+2+4+8)N = 95N，仍比三条 cosine（70N）只多 36%，比五条 cosine（155N）省 39%。拟合后用 bootstrap 报告 β 的 10/90 分位，并用留一规模（如 160M 全部点）做外推检验。
- (a)(b)(c) 见 07/09 笔记。

## 留白
- Sect. 5："Due to the cost of training large models, we only have two comparable training runs at large scale (Chinchilla and Gopher), and we do not have additional tests at intermediate scales."
- Sect. 5："Furthermore, we assume that the efficient computational frontier can be described by a power-law relationship between the compute budget, model size, and number of training tokens. However, we observe some concavity in log(N_opt) at high compute budgets (see Appendix E). This suggests that we may still be overestimating the optimal size of large models."
- Sect. 5："Finally, the training runs for our analysis have all been trained on less than an epoch of data; future work may consider the multiple epoch regime."
- Sect. 3："We assume a power-law relationship between compute and model size as done in Clark et al. (2022); Kaplan et al. (2020), though future work may want to include potential curvature in this relationship for large model sizes."
- App. E："In this work, we do not take this in to account and we leave this as interesting future work as it suggests that even smaller models may be optimal for large FLOP budgets."
- App. D.2："We note that the parameter/data coefficients are both lower than 1/2; this is expected for the data-efficiency coefficient (but far from the known lower-bound). Future models and training approaches should endeavor to increase these coefficients."
- Sect. 5："Speculatively, we expect that scaling to larger and larger datasets is only beneficial when the data is high-quality."；"Better understanding how performance of large language models and toxicity interact is an important future research question."
- Sect. 5："While we have applied our methodology towards the training of auto-regressive language models, we expect that there is a similar trade-off between model size and the amount of data in other modalities."
