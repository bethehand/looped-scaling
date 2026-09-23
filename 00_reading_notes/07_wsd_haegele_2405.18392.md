# 07 · Hägele et al. — Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (arXiv 2405.18392)

> 阅读版本：arXiv v3（2024-10-17），全文（正文 + Appendix A–B）来自 arxiv.org/html/2405.18392v3 与 PDF。凡只能从图中读出的数值均已注明"仅图可读"。

## 基本信息
- 作者 / 机构：Alexander Hägele (EPFL)、Elie Bakouch (Hugging Face)、Atli Kosson (EPFL)、Loubna Ben Allal (Hugging Face)、Leandro Von Werra (Hugging Face)、Martin Jaggi (EPFL)。
- 日期 / 版本：v1 2024-05-28；v2 2024-05-29；v3 2024-10-17（本笔记依据 v3）。
- 发表状态：NeurIPS 2024 Spotlight（arXiv comments 字段）。许可 CC BY-SA 4.0。
- 代码：https://github.com/epfml/schedules-and-scaling/ （基于 nanoGPT 扩展；1B/8B 实验用 nanotron）。

## 方法要点
1. **调度定义（Eq. 1, Sect. 3）**：η(n) = (n/N_warmup)·η_max 若 n < N_warmup；η_max 若 N_warmup < n ≤ N − N_decay；f(n, N, N_decay)·η_max 若 n > N − N_decay。N 为总步数，N_decay 为冷却步数，f 单调递减、终点为 0（"The cooldown phase typically has the LR go to zero"）。作者称之为 constant LR + cooldown（即 Zhai et al. 2022 的 trapezoidal / Hu et al. 2024 的 WSD）。
2. **冷却形状**：线性（"linearly going to zero"，Sect. 3.2）与 **(1-sqrt)**：f = 1 − sqrt((n − (N − N_decay))/N_decay)（Eq. (1-sqrt)）。Fig. 4（200K 步 ≈ 20B tokens，每 20K 步做一次 20% 冷却）："the longer the training duration, the more linear cooldown is outperformed by the (1-sqrt) cooldown"。App. B.1 另测 cosine、mirror cosine、1-square，顺序一致，(1-sqrt) 最好（Fig. 16/17，LR 1e-3 与 5e-4、10% 与 20% 均成立）；对 (1 − x^a) 扫 a < 0.5："Apart from 0.1 and 0.2, which perform noticeably worse because the learning rate is too low for many steps, the other exponents only show a marginal difference to a = 0.5, which still comes out on top"（Fig. 18）。
3. **冷却长度（按总步数的比例 N_decay/N）**：Sect. 3.2 / Fig. 5（124M）："the benefits of extended cooldown periods plateau at around 20%, which we select for the remaining experiments"；Fig. 5 caption："the cooldown surpasses cosine between 10-20% of steps (left), but largely stops improving when done over a majority of training. This also holds when sweeping the LR (right)"。Fig. 6：200k 步长程训练里 **10k 步（5%）的 (1-sqrt) 冷却 "almost perfectly match cosine"**；"the required duration of cooldown to match the cosine loss decreases with longer training"。Fig. 19/20（210M）："Parabola shape of the relationship between cooldown length and final perplexity"（具体曲线值仅图可读）。
4. **学习率**：常数段最优 LR 约为 cosine 最优峰值 LR 的一半（Fig. 3 caption："the optimum lies slightly below at half of the optimal cosine maximum LR"）；扩展实验中明确 "we set the constant LR to be half the maximum LR for cosine and perform 20% (linear) cooldown steps"（Sect. 5）。最优 LR 对不同冷却长度可迁移（Fig. 21）。cosine 基线默认衰减到峰值的 10%。
5. **cosine 衰减到 0 vs 10%**（Fig. 22, Sect. 3.2）：衰减到 0 时 loss 可追平最好的 (1-sqrt) 冷却，但下游指标变差（Table 5：Cosine to 10% 聚合分 46.26，Cosine to 0 为 45.88）；结论 "the maximum and final LR should be set (and ideally swept over) independently"。
6. **用单次训练构造 scaling law 的配方（Sect. 5）**："first, a model sweep with a single sufficiently long training run for each model size in the family; then, using the model checkpoints to perform a cooldown or averaging. This reduces scaling law experiments to only the model scaling axis, effectively dividing the number of necessary training runs by one order of magnitude"。具体做法："for cooldown, we similarly take checkpoints along the constant LR trajectory and perform three annealing periods to match the token counts"——即分支冷却的**终点**落在目标 token 数上，冷却占该目标总步数的 20%。
7. **与 cosine 的匹配程度**：Fig. 3（210M，20% 线性冷却）："an almost perfect match between the performance of the best cosine and cooldown schedule even for different training durations, all while exhibiting slightly less sensitivity to variations in the LR"。Fig. 12 右（33M–360M × 3 个长度）："each model's performance lies almost perfectly on the diagonal"；SWA 点落在对角线下方（"the reciprocal (y-offset) is negative"）。逐点最终验证困惑度见 Fig. 23（仅图可读）。下游：1B/100B tokens 聚合分 Cosine-10% 46.26、Linear-20% 46.20、(1-sqrt)-20% 46.23（Table 5）；1B/460B：Cosine-to-0 48.03、1-sqrt-5% 47.91、Linear-5% 47.84、Linear-10% 47.98、Linear-20% 47.92（Table 6）。8B（Llama-3 架构，12B tokens FineWeb-Edu，20k 步）"matching loss values ... no instability"（Fig. 9，数值仅图可读）。
8. **算力节省**：ratio 10/20/30 的三点设计下 "it saves half the time and FLOPs"（Fig. 13a；逐模型见 Fig. 24/25，"a factor of 1/2 across all models"）。对 Chinchilla 套件按 10% 冷却、seq 1024、batch 0.5M、M = D/N ∈ {10,15,20,25} 估算：5.59×10^23 → 2.36×10^23 FLOPs（Sect. 5, Fig. 13）。"The more training runs are performed per model size (e.g. 4 for Chinchilla), the larger the difference becomes."
9. **替代方案**：SWA（窗口 h = 500 步，≤2500 步 ≈ 256M tokens 最优；EMA 更差）在常数 LR 上大幅降 loss 但 "does not reach the loss values of explicit cooldowns"（Sect. 4.1）。Schedule-Free（SFO）对 (β1,β2) 敏感：(0.95,0.99) 很好、(0.9,0.95) 明显差，二者均被冷却调度追平或超过（Sect. 4.2, Fig. 11）。
10. **冷却期机理**：loss 骤降与 LR 下降同步（Fig. 15/16）；冷却前后权重线性插值也有同样的平滑下降（Fig. 7），提示"the model directly moves within a connected basin"。

## 实验设定
- 架构：decoder-only，SwiGLU、RoPE、RMSNorm、LLaMA/Noam 风格；PyTorch + FlashAttention，bf16 混合精度（App. A.1）。
- 优化器：AdamW (β1,β2) = (0.9, 0.95)，weight decay 0.1（decoupled），grad clip 1.0。
- Warmup：**固定 300 步**（多数运行；>100k 步的长程运行用 1000–3000 步）；所有 scaling 实验 300 步、seq 512（Table 2 说明）。
- Batch：200 条 × 512 ≈ 0.1M tokens（360M 用 0.2M）；词表 GPT-2 tokenizer，50304。
- 数据：SlimPajama 6B 子集（huggingface DKYoon/SlimPajama-6B），验证集约 3M tokens；训练中用固定 32 个 batch（512 长）算验证 loss 曲线，结束后算全验证集困惑度。补充：OpenWebText2（60M/93M/166M，App. B.4）。
- 规模：Table 2 — 33M(d384,L8)、53M(512,8)、60M(512,10)、93M(640,12)、124M(768,12)、151M(768,16)、210M(768,24)、360M(1024,24)；kv_size 64。Table 3 — 每个规模三个 token 数：33M [0.3B,0.7B,1.0B] ratio [9.2,21.4,30.6]；53M [0.4,0.8,1.2B]/[7.2,14.5,21.7]；60M [0.8,1.3,1.8B]/[12.8,21.4,30.0]；93M [1.0,1.8,2.6B]/[11.0,19.2,27.5]；124M [1.5,2.6,3.6B]/[12.4,20.7,29.0]；151M [2.6,3.8,5.1B]/[16.9,25.3,33.7]；210M [3.8,5.1,6.4B]/[18.4,24.6,30.7]；360M [5.1,7.7,10.2B]/[14.2,21.3,28.5]。LR（cosine/常数）：33–93M 为 (2e-3, 1e-3)，124–360M 为 (1e-3, 5e-4)。
- 大模型（Table 4）：1B — d 1792、24 层、ffw 4864、vocab 49152、seq 2048、warmup 2000、batch (1.8M, 2M)、总步 (55k, 220k) ⇒ 100B/460B tokens FineWeb、峰值 LR 8e-4（按 DeepSeek 律取）；8B — Llama-3 架构、vocab 128256、seq 4096、warmup 1000、batch 0.6M、20k 步、LR 3e-4，12 节点 × 4 GH200，TP=4。
- 硬件：多数为 2×A100 数据并行；总耗时约 2500–3000 GPU hours。
- 参数计法：原文未明说是否含嵌入。按 "124M = d768/L12"（与 GPT-2 small 含嵌入 124M 一致；非嵌入约 85M）推断 **N 含嵌入**；Table 3 的 tokens/param 亦按此 N 计算（我的推断，非原文）。
- FLOP 计法（App. A.3, Fig. 14）：不用 6ND，逐项计算再 ×3（反向 = 2×前向）：embedding 2·seq·vocab·d；attention = 2·3·seq·d·(k·h) + 2·seq²·(k·h) + 3·h·seq² + 2·seq²·(k·h) + 2·seq·(k·h)·d；dense(SwiGLU) 2·seq·(3·d·ffw)；final_logits 2·seq·d·vocab；总 = embedding + n_layers·(attention + dense) + final_logits。这与 Chinchilla App. F 公式相同，仅把 dense 换成 SwiGLU 的 3 个矩阵。

## 函数形式与拟合方法
- 原文只写出标准形式 L(N,D) = A/N^α + B/D^β + E（Sect. 5，引 Kaplan/Hoffmann），**没有对任何数据拟合参数律**，因此无拟合值、无 Huber/初始化等细节。Sect. 5 的证据形式是：(i) loss 包络线 vs FLOPs（Fig. 12 左）；(ii) 同一 (N,D) 下 cosine 困惑度（y）对 cooldown/SWA 困惑度（x）的散点是否落在对角线上（Fig. 12 右）。
- 作者报告的敏感性："we find the LR sensitivity (right) to be similar for both schedules, albeit less for cooldown"（Fig. 3）；冷却长度—困惑度呈抛物线（Fig. 19）；"Note that the order might change for substantially different cooldown lengths; we focus on 10% and 20% because they are practically relevant"（App. B.1）。

## 主要结论
1. 常数 LR + 20% 线性冷却在 210M/SlimPajama 上与最优 cosine "almost perfect match"，且 LR 敏感性略低（Fig. 3）。
2. 冷却 ≥10–20% 即超过 cosine，20% 后收益趋平（Fig. 5）；训练越长所需冷却比例越小，200k 步时 5% (1-sqrt) 即可匹配 cosine（Fig. 6）。
3. (1-sqrt) 冷却稳定优于线性，训练越长差距越大（Fig. 4, 16, 17）。
4. 一次长跑 + 多次分支冷却可复现 cosine 从头训练的逐点 loss（Fig. 12 右，33M–360M，ratio ≈10/20/30），FLOPs 与 GPU 时省一半（Fig. 13a, 24, 25）；Chinchilla 套件估计 5.59e23 → 2.36e23。
5. 1B/100B、1B/460B 下游分数 cosine 与 cooldown 一致（46.26 vs 46.20/46.23；48.03 vs 47.84–47.98），8B 短跑 loss 一致（Fig. 8, 9, Table 5, 6）。
6. SWA 免费提升常数 LR 段的 loss 但不及显式冷却；SFO 需调 (β1,β2) 且不及冷却（Sect. 4）。
7. Chinchilla 的观察被复现：cosine 只有周期长度 = 训练长度时最优，中途 loss 高估、周期后延续训练困难（Fig. 1）。

## 失败模式与警告
- **cosine 周期不匹配**：中途 loss 被高估；把 cosine 曲线外推到周期之外是错误的（"one could easily be mistaken to extrapolate a loss curve of a cosine schedule beyond the end of the cycle"）；重新升温（rewarming）会产生 loss 尖峰（Sect. 2, Fig. 1）。
- **常数 LR 不冷却直接当作最终 loss**："the actual performance is then suboptimal, and a cooldown schedule as suggested in our work is needed to properly estimate model performance, in particular for downstream tasks"（Sect. 7，评 Porian/Pearce）。
- **冷却太短**：<10% 在 124M 短跑上不及 cosine（Fig. 5）；(1−x^a) 的 a ≤ 0.2 明显差（Fig. 18）。**冷却太长**：占多数训练后不再改善、呈抛物线（Fig. 5, 19）；下游指标上 5%→20% 不一定更好（Table 6）。
- **LR 不能照搬 cosine 峰值**：常数段应取 cosine 最优峰值的一半（Fig. 3, Sect. 5），否则比较不公平。
- **cosine 终值**：衰到 0 降 loss 却伤下游（"too early saturation"，Fig. 28/Table 5）。
- **SWA/SFO 不能替代冷却**用于取最优点（Sect. 4）。
- **大规模稳定性**：高 LR 长期保持可能不稳，需 QK-norm 等（Sect. 3.3, 6）。
- 本文**没有**证明"用分支冷却点拟合出的 (α, β) 与用 cosine 点拟合出的相同"，只证明逐点 loss 相同；指数层面的一致性需自行用 Chinchilla 流程验证。

## 对我们实验的直接启示
**应该照搬（含数值）**
- 主干常数 LR = 0.5 × cosine 最优峰值 LR（Sect. 5）；AdamW (0.9, 0.95)、wd 0.1、clip 1.0；cosine 对照衰减到 10%。
- 冷却占**分支总步数**的 20%，线性到 0（Sect. 3.2/5）；若追求更低 loss 可全部改用 (1-sqrt)，但所有规模/预算须用同一形状。
- 分支检查点取在 0.8·D_target，使冷却终点恰在 D_target；用固定验证 batch 报告 loss 曲线，用全验证集报告最终值。
- 每个规模至少留 1 条真正的 cosine 从头训练作对角线校验（Fig. 12 右的做法），确认分支点落在对角线上后再用分支点拟合。
- FLOPs 用 Fig. 14 的逐项公式（含 embedding、attention、head），对循环模型把 n_layers 换成**实际执行层数**（共享块 × 循环次数）。
- 主干额外可存 SWA（h = 500 步）作为免费的中间估计（不用于拟合）。

**应该避免**
- 用主干常数 LR 段的 loss 直接当 scaling-law 点；用 SWA 点代替冷却点拟合。
- 冷却比例 <10%（在 10 tokens/param 这样的短分支上尤其）；每个分支用不同的冷却形状或比例。
- 常数段沿用 cosine 峰值 LR；warmup 按固定 300 步照搬到所有规模（见 Porian 笔记：对 160M、batch 0.1M 只等于 0.19N tokens，短于 Porian 最优区间）。

**(a) 冷却长度**：Hägele 的比例定义是 N_decay / N_total（该分支的总步数），推荐 20%（10–20% 即超过 cosine，20% 处收益趋平，Fig. 5）。你们的"分支点的 15%"若指 N_decay = 0.15 × N_branch（即分支后再训 15%，总长 1.15 N_branch，占总长 13%），低于建议；应改为：目标 10/20/40 tokens/param 时，分支检查点取 8/16/32 tokens/param，冷却各 2/4/8 tokens/param（占总长 20%），线性（或 1-sqrt）到 0。40 tokens/param 分支理论上 10% 亦够（Fig. 6：长程训练所需比例下降），但为避免跨预算引入系统偏差建议统一 20%。此设计每个规模总训练量 = 40N + 2N + 4N = 46N tokens，对比三条 cosine 70N，省 34%（Hägele 用 10/20/30 得到 ~50%）。
- (b)(c)(d)(e) 见 08/09 笔记；本文相关的只有：warmup 300 步固定（不推荐照搬）、参数律未拟合、N 含嵌入（推断）。

## 留白
- Sect. 6 Limitations："We conduct our experiments on models of up to 8B parameters, with long training runs of a 1B model on multiple hundred tokens. The trends we find are consistent across all scales, but training behavior can be more brittle at modern scales and extremely long training (Wei et al., 2022; Tay et al., 2021). However, instabilities arising from a high learning rate for a large part of training can be alleviated (Wortsman et al., 2024)."
- Sect. 3.1："It remains an interesting question if a single cooldown schedule is absolutely optimal given a total compute budget for LLM training."
- Sect. 3.1："understanding the curriculum aspect of the separate phases is explored in the concurrent work of Blakeney et al. (2024) and remains open for further research."
- Sect. 3.3："However, not all benchmarks follow this trend, which requires further research."；App. B.5："This observation is an interesting direction for further research."（部分基准在冷却时上翘、部分不）。
- App. B.1："We aim to further investigate the impact of the functional form of the cooldown in future work."
- Sect. 8："Importantly, we do not claim to have established the best learning rate schedule — instead, we investigate and demonstrate how an arguably simple recipe can match the performance of the current best practice of cosine."
- Sect. 3.2："Combining these two arguments implies that the maximum and final LR should be set (and ideally swept over) independently."
