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
    SRC=data/raw/fineweb-edu/sample/10BT                        # 训练数据：FineWeb-Edu sample-10BT，见 ../03_偏离记录.md
    python scripts/prepare_data.py download --skip-fwe          # 只下三个小验证集
    python scripts/prepare_data.py --src $SRC tokenizer --chars 2e9
    python scripts/prepare_data.py --src $SRC tokenize --workers 32
    python scripts/prepare_data.py --src $SRC valsets --workers 16
    python scripts/measure_throughput.py --device cuda          # 检查 4：实测吞吐 -> configs/throughput.json
    python scripts/make_configs.py --sweep                      # 第 3 周：学习率扫描配置
    python scripts/run_queue.py --manifest configs/manifest_sweep.csv --gpus 0,1,2,3
    python fit/collect_results.py --manifest configs/manifest_sweep.csv --out results/sweep.csv
    python scripts/pick_lr.py --results results/sweep.csv     # -> configs/lr_table.json
    python scripts/make_configs.py --lr-table configs/lr_table.json   # 主网格 111 个配置 + manifest.csv
    python scripts/run_queue.py --manifest configs/manifest.csv --gpus 0,1,2,3
    python fit/collect_results.py && python fit/fit_laws.py     # 第 8 到 9 周

## 在 tmux 里运行：实时输出并保存日志
    mkdir -p logs
    python -u scripts/xxx.py ... 2>&1 | tee -a logs/xxx.log
- `-u` 让 Python 每行立即输出；`2>&1` 把报错也算进来；`tee -a` 同时显示在屏幕并追加写入日志。
- 队列会把每个训练任务的输出加上 `[gpuN]` 前缀实时打到屏幕，同时写入 `runs/<任务名>/stdout.log`。
- 离开 tmux 但保持运行：Ctrl-b 再按 d；回来：`tmux attach`。

## 队列与进度
- 队列进度写在 `runs/queue_<清单文件名>`，例如 `runs/queue_manifest_sweep.csv`；Git 跟踪的清单文件不会被修改。
- 查看完成数：`grep -c ,done, runs/queue_manifest_sweep.csv`；查看失败：`grep ,failed, runs/queue_manifest_sweep.csv`。
- 失败的任务超过重试次数后标为 failed。排查后把状态文件里该行的 failed 改成 pending，再启动一次队列即可，已完成的任务不会重跑。
- 结果指标：`val_*` 为冷却终点损失（主指标），`valavg_*` 为尾部三点均值（稳健性指标）。
