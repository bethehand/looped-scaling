# looped-scaling

受控小规模 scaling 研究：循环 / 递归深度 Transformer 的循环等价指数 φ。
实验设定以 `../01_预注册文档.md`（v1.0，已冻结）为准。

## 布局
- `looped/`  模型、FLOP 计数、数据流、学习率调度、训练循环、评测
- `scripts/` 数据准备、配置生成、队列运行、吞吐测量
- `fit/`     scaling law 拟合（F1 / F2 / F3、Huber + L-BFGS-B、bootstrap、留一档外推）
- `tests/`   开跑前检查（截断反传梯度、r=1 等价、FLOP 计数、参数计数）
- `configs/` 自动生成的运行配置与 manifest

## 环境
    uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
    python -m pytest tests -q

## 使用顺序（第 1 到 3 周）
    python -m pytest tests -q                                   # 五项检查中的 1、3 与训练冒烟
    python scripts/prepare_data.py download --n-files 12        # 在 4090 机器上执行
    python scripts/prepare_data.py tokenizer && python scripts/prepare_data.py tokenize && python scripts/prepare_data.py valsets
    python scripts/measure_throughput.py --device cuda          # 检查 4：实测吞吐 -> configs/throughput.json
    python scripts/make_configs.py --sweep                      # 第 3 周：学习率扫描配置
    python scripts/run_queue.py --manifest configs/manifest_sweep.csv --gpus 0,1,2,3
    python fit/collect_results.py --manifest configs/manifest_sweep.csv --out results/sweep.csv
    python scripts/pick_lr.py --results results/sweep.csv     # -> configs/lr_table.json
    python scripts/make_configs.py --lr-table configs/lr_table.json   # 主网格 111 个配置 + manifest.csv
    python scripts/run_queue.py --manifest configs/manifest.csv --gpus 0,1,2,3
    python fit/collect_results.py && python fit/fit_laws.py     # 第 8 到 9 周
