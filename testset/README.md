# Jifang 50 任务测试集

`~/xpc/testset/manifest.json` 固定了 50 个任务：25 个小补丁（参考补丁增删行数不超过 15）和 25 个中补丁（16–150 行），不包含大补丁。任务难度按参考补丁 hunk 中的增删行数计算。新清单从原清单中移除全部 6 个大补丁及 7 个中补丁，保留其余任务；新增 10 个已缓存镜像的小补丁，以及 3 个 Rust 1.40 或更新版本的小补丁。新增任务都排除了旧 Cargo 工具链。三个额外候选记录在 `manifest-additions.json`，原 100 项候选清单 `manifest-100.json` 保持不变。镜像缓存使用代理构建；调整清单时没有删除任何 Docker 镜像，移出测试集的大补丁镜像仍保留在本机。

测试任务与固定验证集不重叠。清单仍记录任务、基础提交、镜像标签及来源数据散列，不会随新增镜像自动变化。原 50 项清单覆盖 23 个仓库；本次按补丁大小重配后仓库覆盖会随入选任务变化，因此不要将其视为与旧清单具有相同的仓库分布。历史运行记录未用于筛选任务，不能仅凭与验证集不重叠而声称它们是未接触过的独立测试集。

Cargo crate 和 Git 依赖保存在服务器共享目录 `~/xpc/cargo-cache/`，评测容器按需挂载共享缓存；每个任务镜像自带的 crates 索引继续从镜像读取，不会复制或改写任务镜像。使用代理预取参考测试补丁所需依赖，并在无网络容器中核验：

```bash
~/xpc/envs/rustbench/bin/python ~/xpc/testset/cache_cargo.py --workers 2
```

运行测试时，Cargo 命令使用 `--frozen`；如果参考测试补丁使锁文件需要更新，预取时生成的版本会在补丁应用后恢复。Cargo 1.35 也支持此模式。缓存状态和逐任务日志保存在 `~/xpc/cargo-cache/cache-manifest.json` 与 `~/xpc/cargo-cache/logs/`。

```bash
# 登录 Jifang。以下两条命令不调用模型。
~/xpc/testset/testbench --agent all --list
~/xpc/testset/testbench --agent all --dry-run

# 默认在后台运行完整测试集并使用直连网络，输出 PID 和启动日志路径。
~/xpc/testset/testbench --agent normal
~/xpc/testset/testbench --agent swe-agent
~/xpc/testset/testbench --agent RustForger
~/xpc/testset/testbench --agent ACI-impr
~/xpc/testset/testbench --agent test

# 也可按顺序运行全部注册智能体（默认后台、直连）
~/xpc/testset/testbench --agent all --max-workers 2 --timeout 1800

# 单次覆盖 .env 和 shell 环境中的 MODEL
~/xpc/testset/testbench --agent normal --model deepseek-v4.1-flash

# 前台执行：指定 --foreground
~/xpc/testset/testbench --foreground --agent normal --instance BurntSushi__ripgrep-2209

# 如需使用本地代理和反向隧道，指定 --network proxy
~/xpc/testset/testbench --agent normal --network proxy
```

每次实验读取服务器 `~/xpc/.env` 和服务器上的智能体代码，可能产生模型调用费用；`--model` 可单次覆盖模型名，实际模型记录在运行元数据中。默认使用直连模式，不检查反向隧道；若指定 `--network proxy`，启动前须在 Windows 端启动 11321 代理并保持 SSH 反向隧道。后台启动前先同步执行一次 dry-run。`--max-workers` 控制补丁评测并发，智能体任务仍顺序运行；`--timeout` 是单任务测试超时（秒）。

启动日志位于 `~/xpc/testset/launcher-logs/`。具体运行目录为 `~/xpc/testset/runs/testset_<时间>_<随机数>/`，包含清单、运行元数据、智能体日志、预测及 `summary.json`。测试明细位于 `~/xpc/Rust-bench/logs/run_evaluation/<run_id>_<agent_id>/<agent>/<instance>/`。状态 `resolved`、`unresolved`、`no_patch`、`agent_error` 和 `evaluation_error` 含义与 `~/xpc/validation/README.md` 相同。

`select_tasks.py` 仅用于最初冻结清单，已有清单时拒绝覆盖。执行运行器前，请确认 Windows 端 11321 端口处于监听状态，以及 Jifang 通过 SSH 反向隧道能访问该代理。
