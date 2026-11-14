# Jifang 50 任务测试集

`~/xpc/testset/manifest.json` 固定了 50 个任务：先从原有 100 个非验证集任务中按仓库名称和实例 ID 排序，逐轮取满 50 个，再将其中 7 个 Rust/Cargo 1.40 之前的任务替换为 1.40 或更新版本的任务。替换任务仍来自原 100 任务清单 `manifest-100.json`；原有任务镜像均保留。清单记录任务、基础提交和镜像标签及来源数据散列；不会随新增镜像自动变化。所选任务覆盖原有的全部 23 个仓库，并未根据历史运行记录排除任务，因此部分任务此前可能已被评测或用于开发，不能仅凭与验证集不重叠而声称它们是未接触过的独立测试集。

Cargo crate 和 Git 依赖保存在服务器共享目录 `~/xpc/cargo-cache/`，评测容器按需挂载共享缓存；每个任务镜像自带的 crates 索引继续从镜像读取，不会复制或改写任务镜像。使用代理预取参考测试补丁所需依赖，并在无网络容器中核验：

```bash
~/xpc/envs/rustbench/bin/python ~/xpc/testset/cache_cargo.py --workers 2
```

运行测试时，Cargo 命令使用 `--frozen`；如果参考测试补丁使锁文件需要更新，预取时生成的版本会在补丁应用后恢复。Cargo 1.35 也支持此模式。缓存状态和逐任务日志保存在 `~/xpc/cargo-cache/cache-manifest.json` 与 `~/xpc/cargo-cache/logs/`。

```bash
# 登录 Jifang。以下两条命令不调用模型。
~/xpc/testset/testbench --agent all --list
~/xpc/testset/testbench --agent all --dry-run

# 选择一个智能体在后台运行完整测试集，输出 PID 和启动日志路径。
~/xpc/testset/testbench --background --agent normal
~/xpc/testset/testbench --background --agent swe-agent
~/xpc/testset/testbench --background --agent RustForger
~/xpc/testset/testbench --background --agent ACI-impr
~/xpc/testset/testbench --background --agent trace-impr
~/xpc/testset/testbench --background --agent test

# 也可按顺序运行全部注册智能体；--background 必须是第一个参数。
~/xpc/testset/testbench --background --agent all --max-workers 2 --timeout 1800

# 单个或多个清单内任务；省略 --background 可在前台执行。
~/xpc/testset/testbench --agent normal --instance BurntSushi__ripgrep-2209
```

每次实验读取服务器 `~/xpc/.env` 和服务器上的智能体代码，可能产生模型调用费用。启动前须在 Windows 端启动 11321 代理并保持 SSH 反向隧道；入口检查服务器到代理的连通性，后台启动前先同步执行一次 dry-run。`--max-workers` 控制补丁评测并发，智能体任务仍顺序运行；`--timeout` 是单任务测试超时（秒）。远程代理或隧道中断会影响运行。

启动日志位于 `~/xpc/testset/launcher-logs/`。具体运行目录为 `~/xpc/testset/runs/testset_<时间>_<随机数>/`，包含清单、运行元数据、智能体日志、预测及 `summary.json`。测试明细位于 `~/xpc/Rust-bench/logs/run_evaluation/<run_id>_<agent_id>/<agent>/<instance>/`。状态 `resolved`、`unresolved`、`no_patch`、`agent_error` 和 `evaluation_error` 含义与 `~/xpc/validation/README.md` 相同。

`select_tasks.py` 仅用于最初冻结清单，已有清单时拒绝覆盖。执行运行器前，请确认 Windows 端 11321 端口处于监听状态，以及 Jifang 通过 SSH 反向隧道能访问该代理。
