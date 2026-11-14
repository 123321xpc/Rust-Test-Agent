# bytes-377 配对干预实验

对照组：unresolved
干预组：unresolved
正向翻转：否
干预事件数：1

两组使用同一个断点，干预组仅覆盖 Architect 方案中的 arbitration_rationale。
断点结合了新采集的 Navigator 状态与 P0 轨迹中的 Architect 方案，不能视为 P0 原始断点的精确复现。
该定向干预利用了此前评测结果，属于事后分析，不能作为独立测试集的泛化结论。
题述允许不触发 panic 的合规实现，不能将 panic 表述为题目的唯一要求。
单次运行受模型随机性和其他方案字段影响，不能单凭结果断定该字段的稳定因果效应。

## 失败测试

### control

```text
---- test_vec_advance_mut stdout ----
thread 'test_vec_advance_mut' panicked at 'new_len = 12; capacity = 8', <::core::macros::panic macros>:5:50
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
note: panic did not contain expected string
      panic message: `"new_len = 12; capacity = 8"`,
 expected substring: `"cannot advance"`
```

### intervention

```text
---- test_vec_advance_mut stdout ----
thread 'test_vec_advance_mut' panicked at 'new_len = 12; capacity = 8', <::core::macros::panic macros>:5:50
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
note: panic did not contain expected string
      panic message: `"new_len = 12; capacity = 8"`,
 expected substring: `"cannot advance"`
```

## 原始状态

```json
{
  "control": {
    "arm": "control",
    "agent_exit_code": 0,
    "seconds": 344.71,
    "patch_bytes": 1368,
    "verdict": "unresolved",
    "evaluation_exit_code": 0,
    "report": "/home/xupeicai/xpc/Rust-bench/logs/run_evaluation/p1_bytes_20261114_original_plan_pair_control/test/tokio-rs__bytes-377/report.json",
    "test_output": "/home/xupeicai/xpc/Rust-bench/logs/run_evaluation/p1_bytes_20261114_original_plan_pair_control/test/tokio-rs__bytes-377/test_output.txt"
  },
  "intervention": {
    "arm": "intervention",
    "agent_exit_code": 0,
    "seconds": 145.28,
    "patch_bytes": 815,
    "verdict": "unresolved",
    "evaluation_exit_code": 0,
    "report": "/home/xupeicai/xpc/Rust-bench/logs/run_evaluation/p1_bytes_20261114_original_plan_pair_intervention/test/tokio-rs__bytes-377/report.json",
    "test_output": "/home/xupeicai/xpc/Rust-bench/logs/run_evaluation/p1_bytes_20261114_original_plan_pair_intervention/test/tokio-rs__bytes-377/test_output.txt"
  }
}
```

## 来源

```json
{
  "checkpoint": "/home/xupeicai/xpc/validation/runs/p1_bytes_20261114_forced_plan/checkpoint",
  "plan_trace": "/home/xupeicai/xpc/validation/runs/validation_20261114_023937_af2e8d/test/tokio-rs__bytes-377/traces/67ecef2411e5498faa19c1005ee8dd42.jsonl",
  "note": "Original plan combined with newly captured Navigator state; not an exact reconstruction."
}
```
