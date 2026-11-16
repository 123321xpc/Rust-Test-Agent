# Jifang 五任务验证集

固定清单为 `manifest.json`，服务器上的入口是 `~/xpc/validation/validate`，使用 rootful Docker。通过 `--network proxy`（默认）或 `--network direct` 选择网络模式。代理模式要求 Windows 端 11321 代理和 SSH 反向隧道，入口在启动及每次评测前检查代理；直连模式清空智能体、构建步骤及测试容器的代理环境变量，不检查反向隧道。

该选项不修改 Docker 守护进程的全局代理配置；如需拉取镜像，守护进程仍使用自身配置的代理。此验证集会在运行前检查任务镜像已经缓存。直连模式下模型 API 和 GitHub 等目标仍需能从服务器直接访问。

```bash
# 登录 Jifang 后运行。--list、--dry-run 不调用模型。
~/xpc/validation/validate --agent all --list
~/xpc/validation/validate --agent all --dry-run
~/xpc/validation/validate --agent all --network direct --dry-run

# 运行一个智能体、生成补丁并评测全部 5 项
~/xpc/validation/validate --agent normal
~/xpc/validation/validate --agent swe-agent
~/xpc/validation/validate --agent RustForger
~/xpc/validation/validate --agent ACI-impr
~/xpc/validation/validate --agent trace-impr
~/xpc/validation/validate --agent test

# 顺序运行全部 6 个智能体（最多 30 次任务运行）
~/xpc/validation/validate --agent all

# 后台运行：--background 必须放在第一个参数，命令会打印启动日志位置
~/xpc/validation/validate --background --agent all
~/xpc/validation/validate --background --agent normal --network direct

# 指定模型；此值覆盖 .env 和 shell 环境中的 MODEL
~/xpc/validation/validate --agent normal --model deepseek-v4.1-flash

# 仅跑一个或多个验证集内的任务
~/xpc/validation/validate --agent normal --instance chronotope__chrono-991
~/xpc/validation/validate --agent test --instance chronotope__chrono-991 --instance biomejs__biome-2621

# 控制评测并发和单任务测试超时；智能体仍按任务顺序执行
~/xpc/validation/validate --agent normal --max-workers 2 --timeout 1800
```

模型、API 地址、密钥和调用上限读取 `~/xpc/.env`，同名 shell 环境变量优先；命令行 `--model` 可单次覆盖模型名。实际模型名记录在 `run_metadata.json` 中。运行会调用模型并可能产生费用。后台任务与 SSH 会话独立；代理模式下代理或反向隧道中断会影响后续操作。网络模式记录在 `run_metadata.json` 中。

入口使用服务器上的代码：`~/xpc/baselines/{normal,ACI-impr,trace-impr,test}`、`~/xpc/RustForger` 和 `~/xpc/sweagent-run/SWE-agent`。如果在 Windows 工作区修改了智能体，须先将相应修改同步到这些服务器目录；验证命令不会自动同步代码。

每次运行创建 `~/xpc/validation/runs/validation_<时间>_<随机数>/`：

- `manifest.json`：冻结的验证集清单；`run_metadata.json`：实际任务、模型名、API 上限、智能体源码散列、数据集散列及 Docker 镜像 ID。
- `<agent>/<instance>/agent.log` 和相应轨迹：智能体日志。
- `<agent>/predictions.jsonl`、`agent_status.json`、`evaluation.log`：补丁、运行状态与评测日志。
- `summary.json`：每个任务的状态和汇总计数。`resolved` 和 `unresolved` 有完整评测报告；`no_patch` 表示成功退出但没有补丁；`agent_error`、`evaluation_error`、`error` 分别表示智能体、评测或报告错误。错误与未解决分开计数。
- 测试明细与报告保存在 `~/xpc/Rust-bench/logs/run_evaluation/<run_id>_<agent_id>/<agent>/<instance>/`；`agent_id` 是将智能体名称中的 `-` 替换为 `_` 后的名称，例如 `swe_agent`。个别失败不会阻止后续任务，但最终命令会以非零状态退出。

## 抽样规则

从抽样时的 110 个 `rustbench/` 本地任务镜像出发，根据仓库、PR 编号和基准提交精确匹配数据集，剔除没有问题描述映射或曾有评测报告的任务。`manifest.json` 记录了当时的排除列表。排除只依据当时在服务器上找到的报告，不保证此前从未被运行过。

以标准补丁中实际增删行数之和衡量规模：小 ≤15 行，中 16–150 行，大 >150 行。15 和 150 采用论文分析中出现的尺度，作为本验证集的操作性分界，并非论文给出的正式三类标准。原始十项清单按 `rustbench-validation-v1` 种子和任务 ID 的 SHA-256 排序，在各档优先选择此前未选过的仓库，选出大、中、小各 3、4、3 个。当前清单从中保留全部 3 个小补丁任务，以及改动行数最少的 2 个中补丁任务（31、63 行）；规模并不等价于任务难度。

`select_tasks.py` 记录抽样过程，发现已有 `manifest.json` 时拒绝覆盖。任务清单固定后，日常运行不会重新抽样。验证集如果被反复用于优化，就属于开发验证集，不应作为最终独立测试集。

## bytes-377 配对干预实验

`run_bytes_p1.py` 在机房使用任务镜像和直连网络执行定向实验。`capture` 强制经过 Architect，保存其方案交接断点；`prepare` 可将历史轨迹中的方案与新采集的 Navigator 状态组合；`replay` 在两个独立容器中分别重放对照组和单字段干预组，并调用基准评测；`report` 汇总结果。组合断点并非原始运行的精确复现，实验使用了历史评测信息，应按事后分析解读。

```bash
cd ~/xpc
export LD_LIBRARY_PATH="$HOME/xpc/gcc-12/lib:$HOME/xpc/gcc-12/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"
PY=~/xpc/envs/rustbench/bin/python
EXPERIMENT=~/xpc/validation/runs/<experiment-name>
$PY validation/run_bytes_p1.py capture --dir "$EXPERIMENT"
# 如需重用历史方案，在另一个实验目录执行 prepare：
$PY validation/run_bytes_p1.py prepare --dir <paired-dir> --source-checkpoint "$EXPERIMENT/checkpoint" --plan-trace <original-trace.jsonl>
$PY validation/run_bytes_p1.py replay --dir <paired-dir>
$PY validation/run_bytes_p1.py report --dir <paired-dir>
```

`RUSTBENCH_DUMP_POST_PLAN` 控制交接断点输出目录；`RUSTBENCH_FORCE_PLAN_CAPTURE=1` 仅用于需要捕获 Architect 方案的实验。断点内的 `repair_plan` 使重放跳过 Architect；对旧的 Navigator 断点执行方案字段干预时，系统会在 Architect 生成方案后再应用干预。

## 测试

`python -m unittest discover -s validation -p 'test_*.py' -v` 测试抽样统计、固定清单、SWE-agent 输入与输出转换及空补丁处理。`smoke_checks.py` 提供服务器上的无模型容器和评测检查；`smoke-normal.sh` 是仅限一次模型请求的单任务检查，不代表正常预算下的成功率。

## 2026-11-13：原十项验证集 crate 离线缓存

以下为缩减前原十项清单的离线缓存记录。已通过 Windows 11321 代理和 SSH 反向隧道，为当时十项任务镜像缓存锁文件中的 Cargo 依赖；`cargo-generate__cargo-generate-880` 还缓存了参考补丁新增的依赖。原任务镜像保存在对应镜像标签末尾 `-before-offline-20261113` 的回退标签中；冻结清单里的原标签指向已缓存镜像。任务源码、锁文件和镜像默认启动配置均与回退镜像核对一致。

当时十项均在 `--network none` 容器中通过 `cargo metadata --locked`；应用参考补丁和测试补丁后，十项的相关测试目标均通过 `cargo test --no-run` 的离线编译。`sharkdp__bat-562` 的参考补丁中 `Cargo.lock` 改动无法应用到镜像现有锁文件，编译检查因此跳过了该锁文件改动并使用镜像现有锁文件。编译检查不运行测试，也不保证任意新补丁引入的依赖已经缓存；模型 API 和可能在测试运行期间主动发起的网络请求不属于此缓存验证。

审计记录及每项日志位于 `~/xpc/validation/offline-cache-20261113/`，其中 `final-audit.json` 记录镜像配置、工作区与离线依赖解析结果，`build-summary.json` 记录断网编译结果和例外。
