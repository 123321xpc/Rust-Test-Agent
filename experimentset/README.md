# Jifang 100 任务实验集

`~/xpc/experimentset/manifest.json` 固定了 100 个任务：40 个小补丁（参考补丁增删行数不超过 15）、55 个中补丁（16–150 行）以及 5 个大补丁（>150 行）。

- **覆盖度**：完整包含 50 项测试集（`~/xpc/testset`）中的全部 25 个小补丁和 25 个中补丁。
- **验证集隔离**：排除了验证集中的 3 个小补丁任务。
- **大补丁分布**：5 个大补丁分别来自 5 个不同仓库（`async-graphql`, `bevy`, `biome`, `cargo-generate`, `cargo-edit`），均已预先建立共享 Cargo 缓存。
- **Cargo 缓存挂载**：评测阶段默认启用共享缓存挂载模式（`RUSTBENCH_CARGO_CACHE_ROOT=~/xpc/cargo-cache`），挂载共享依赖并进行离线/frozen 编译评测。

```bash
# 登录 Jifang。以下命令不调用模型：
~/xpc/experimentset/experimentbench --agent all --list
~/xpc/experimentset/experimentbench --agent all --dry-run

# 默认后台执行：
~/xpc/experimentset/experimentbench --agent normal

# 前台单题测试：
~/xpc/experimentset/experimentbench --foreground --agent test --instance BurntSushi__ripgrep-2209
```
