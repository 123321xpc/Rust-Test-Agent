# test 智能体优化分析：验证轨迹 × Multi²Fixer

分析对象：`validation_20261114_102110_3b48da`，2026-11-14，agent=`test`，模型=`deepseek-v4.1-flash`，API_LIMIT=300，固定 10 任务开发验证集。

**主要判断：优先改进“修复状态、验证证据和失败后的调度”，再引入有条件的多候选生成。当前已经具备多角色、快慢路径、CKG、Red–Green 和独立 QA；继续堆角色或扩大 API_LIMIT，不能直接解决本轮暴露的问题。**

本文依据用户提供的 Multi²Fixer 中文解读，未独立复核论文原文中的实验数字。实际运行证据来自服务器的全部 10 份 trace、预测补丁、报告及测试输出；重新计算的服务器源码摘要与该 run 的 `agent_source_sha256.test` 一致，因此实现分析对应本次运行版本。

附带数据：`trace_metrics.json` 保存逐任务与逐角色统计、阶段事件、交接内容、补丁和评测错误；`focused_evidence.json` 保存关键工具反馈与降级提交事件；`cost_summary.csv` 便于比较后续实验。未修改智能体代码或启动新的模型实验。

## 1. 先纠正本轮结果的解释

官方结果为 **3/10 resolved（30%）**：3 resolved、5 unresolved、1 agent_error、1 评测执行错误。不能把出补丁或 agent exit 0 视为修复成功。

上一次状态汇报的三处解释不准确：

- `No-Op` 是报告分类，不能据此推断“定位没改到点上”。本轮四个 No-Op 均出现评测编译错误，测试未正常执行；这与行为测试真正运行后失败不同。
- biome 的 `Test patch failed to apply` 不能直接归为纯基础设施故障。提交补丁修改了已有测试 `crates/biome_json_parser/src/lexer/tests.rs`，把 `JSON_STRING_LITERAL:35` 改为 `ERROR_TOKEN:35`；评测日志恰好在该 hunk 应用失败。这是提交补丁与评测测试补丁冲突。
- cargo-generate 的预测只修改了 `Cargo.toml`，没有修改 `tests/integration/git.rs`。`gix_config` 未找到出现在评测测试加入后，不能说是 agent 修改了测试 import 导致的。

这三个事实改变优化优先级：需要检查 API/依赖契约、验证覆盖、测试隔离与环境约束，不能把全部失败简化为“故障定位不准”。

| 任务 | 最终结果 | API 调用 | 总 token，含缓存输入 | 本轮关键证据 |
|---|---|---:|---:|---|
| cargo-generate-880 | unresolved / No-Op | 184 | 2,457,650 | 目标依赖版本不可用；最终只改 `~0.16.1 → ~0.16.2` 并添加 MSRV；评测新增测试引用 `gix_config`，编译失败 |
| tracing-1523 | agent_error | 299 | 7,915,618 | 232 次 `read_lines`；只有 1 次显式 `cargo_check`、0 次显式 `cargo_test`；两轮 Fixer 各耗尽 114 调用；最终无补丁 |
| aya-253 | unresolved / No-Op | 67 | 914,825 | 修改 19 个文件；自测主要验证 mock 的 detach 计数；评测缺少 `OwnedLink`，并把 `forget` 返回值当 `()` 而非可 detach 对象 |
| biome-2621 | error | 298 | 6,981,662 | 三次规划、37 次显式 `cargo_test`；改动已有 lexer 测试导致 test patch 冲突 |
| cargo-edit-572 | unresolved / No-Op | 150 | 1,744,634 | 固定保留两参数 `set_git`，自测围绕新加的 `set_rev`；评测调用四参数 `set_git`，E0061 |
| cargo-dist-555 | unresolved / Regression | 299 | 5,749,758 | 17 次 `write_regression_test`；最终 `auto_submit_degraded`，通过回归数为 0；模板报 `unknown statement elif` |
| ast-grep-1707 | unresolved / No-Op | 149 | 3,391,445 | fast path，仅修改 CLI `scan.rs`；评测核心 config 测试需要不同 matches 表示及 `set_unused_suppression_rule`，编译失败 |
| bytes-377 | resolved | 46 | 647,365 | fast path，真实契约修复通过；保留了与旧测试冲突时依据 issue 判断的能力 |
| chrono-991 | resolved | 35 | 413,564 | planned path，Eq/Ord/Hash 保持一致，通过 |
| bat-562 | resolved | 138 | 2,750,364 | 19 次回归测试写入；补丁 7 文件、201 增删行；官方参考修复为 3 增删行，存在值得比较的低成本候选空间 |

四项编译错误暴露的是**当前补丁与评测所要求接口不兼容**。对于未在 issue 中明确给出精确接口的功能请求，不能仅凭隐藏测试指定另一种接口，就证明当前替代方案的全部语义都错误；但按当前 benchmark，它们确实不算 resolved。提升真实能力应从公开 issue、文档、可见调用方和资源语义推导契约，隐藏评测仅用于事后归因。

按固定 manifest 的参考补丁大小划分：small 3/3，medium 0/4，large 0/3。这个单轮小样本提示复杂任务是短板，不能据此声称已证明某种架构稳定优劣。

## 2. 成本到底花在哪里

全轮累计：

- 模型 HTTP 请求 **1,665 次**。
- 输入 **32,583,403 token**，输出 **383,482 token**，合计 **32,966,885 token**。
- 其中缓存命中输入 **22,092,928 token**，约占输入的 **67.8%**。
- 10 个 agent 的记录耗时合计 **6,246.64 秒**，约 104.1 分钟；不含随后约 10.6 分钟的集中判题。
- tracing、biome、cargo-dist 合计 **20,647,038 token，62.63%**，三项均未成功。

以上是 API usage 的累计处理量，包含每轮重复发送及缓存命中的输入，既不是独立源代码量，也不是输出代码量。trace 同时记录 model 与 HTTP 事件，统计只使用一次 HTTP usage / 最终 agent_metrics，没有把重复日志副本重复计费。

| 角色 | API 请求 | 输入 token | 输出 token |
|---|---:|---:|---:|
| Navigator | 191 | 2,667,836 | 53,739 |
| Architect/Suggester | 120 | 1,762,323 | 49,388 |
| Fixer | 1,297 | 27,607,679 | 253,004 |
| Verifier | 57 | 545,565 | 27,351 |

**Fixer 占约 84.5% 的总 token；最终 Verifier 约 1.7%。** 因此，首先删独立 QA 并不能省掉主要成本，反而可能放大模板错误、接口遗漏和自证式测试。更值得做的是减少 Fixer 无进展的浏览、重复试错和整轮重来。

实际费用应按供应商的非缓存输入、缓存输入、输出单价分别计算：

`费用 = 非缓存输入 × 单价_u + 缓存输入 × 单价_c + 输出 × 单价_o`

trace 中 `usage.credit` 的单位未确认，不把它解释为美元或人民币。论文解读中的统一 token 价格也不能用于本轮账单。

## 3. 从 Multi²Fixer 应借鉴什么

用户提供的解读强调了四件事：多位置之间的依赖、持续维护部分补丁状态、候选生成与选择分离、先编译后行为测试。它也指出：增加 Proposer 数量可能降效；局部候选多数意见不等于正确；复杂依赖会让 Coordinator 循环。

与当前 test 的对应关系如下：

| Multi²Fixer 的机制 | test 已有能力 | 仍然缺少的关键行为 |
|---|---|---|
| BugAnalyzer | Navigator、CKG、候选故障链、临时复现 | 故障假设和测试预期缺少独立验证，可能把假设变成自己的验收标准 |
| 动态 Hunk 调度 | 单个 `RepairPlan.ordered_edits`、快慢路径、失败后重规划 | 没有对每个修复单元维护“未完成依赖 / 证据 / 已验证状态”，容易全量重来 |
| 多个 Proposer 产生局部候选 | 单个 Architect 选链、单个 Fixer 迭代 | 出现分歧后通常继续同一路线，而不是有预算地比较不同机制 |
| 部分补丁与状态兼容性 | checkpoint、best verified / last compiling snapshot | 升级重规划时常恢复整个 `pre_fix`；缺少按依赖范围保留已验证成果 |
| Syntax/Test Refinement | cargo 工具、确定性 QA、LLM QA、修复循环 | 模板/资源/API 兼容性不一定被现有 Cargo 编译检查覆盖；预算耗尽兜底可能绕过原来的验证要求 |

**不应照搬固定 3 个 Proposer、给每个 hunk 多次模型调用。** 论文主要实验具有真实故障函数位置，test 从自然语言 issue 开始；Rust 的 traits、feature gates、宏及资源生命周期也要求不同的依赖和验证方式。

## 4. 优先优化一：让最终提交保留已经承诺的验证强度

最明确的案例是 cargo-dist：最终 trace 为：

```json
{"event":"auto_submit_degraded","passed_regressions":0,"all_non_rust":true,"unverified_compiling":true}
```

对应服务器 `baselines/test/multi_agent.py:952–1009` 的 `_try_degraded_patch_fallback`：允许非 Rust 文件或仅能 post-strip 编译的补丁兜底。`tools.py:1451` 的 post-strip 检查已经使用 `cargo check --tests`，但 Cargo 编译不会解析 Jinja 模板、执行渲染或验证生成的 shell。

**具体建议：**

1. 保留候选输出用于诊断，但把“仅编译通过、未完成行为验证”单独标成候选状态；不让预算耗尽自动抹掉既有的行为失败或 QA 拒绝。
2. 有 fully verified checkpoint 时优先恢复该候选；没有时记录确切缺口，不再用 `all_non_rust` 推断安全。
3. 按改动类型分派验证：Rust 源码→受影响包的编译/测试；模板→实际模板引擎加载并渲染；shell→渲染后 `sh -n` 加最小沙箱行为检查；配置/依赖→真实解析、解析出的版本图及必要构建。
4. 渲染、测试执行、输出摘要都通过确定性工具完成，不需要每一步额外调用一个 LLM。

**收益及边界：** 直接减少“自测已报错但最终仍提交”的补丁；本身不保证 resolved 增加，可能把某些 unresolved 更诚实地改成 verification_error。只有与后续的失败定向修复配合，才可能转化为更多成功。

## 5. 优先优化二：修复一次验证一个有效单元，阻断无证据循环

tracing 用掉 299 次请求，其中 `read_lines` 232 次，明确的 `cargo_check` 只有末尾 1 次，显式 `cargo_test` 为 0。`write_regression_test(run_now=True)` 仍有自动测试，因此不能说全程完全没有执行测试。其末尾编译错误包括缺少 `register_filter`、`Context::with_filter` 等相互依赖的接口。

当前已有 pacing notice 与重复失败检测，但两者均存在缺口：

- pacing notice 主要是追加文本提醒，tracing 中累计出现 23 次，仍未形成及时验证。
- 重复失败计数依赖调用 Cargo 后的反馈；如果一直读代码或改代码，不会触发这种升级条件。
- stage 失败升级处恢复 `pre_fix`，会让后续规划重新探索及实施；没有按已完成依赖单元保存成果。

**具体建议：**

- 把修复计划拆成少量可验证单元。例如 tracing 的“接口定义 / filter 注册 / Context 传播 / registry 状态”各有完成条件和依赖。
- 给每个单元维护 `pending / implementing / compile_checked / behavior_checked / blocked`，而不是只有当前 diff 和整阶段调用数。
- 完成一个依赖闭合的单元就运行最便宜的相关检查。相互依赖的多个 hunk 可联合验证；不要要求每个尚未补全依赖的单行修改都立即跑全套测试。
- 若连续若干调用没有新增相关符号、可用复现、实现进展或新验证证据，调度器要求提交当前假设和阻塞点，决定检查、换假设或停止这条分支。可从 15–20 次无进展调用作为实验起点，阈值需验证。
- 重规划只回退失败单元及其依赖影响范围；保留已有编译/行为证据有效的单元。接口改变导致证据过期时明确失效。

这才是 Multi²Fixer 动态状态与调度对当前 test 的主要价值。仅将 Architect 的提示词写得更长，无法改变上述循环机制。

## 6. 优先优化三：验证预期行为，避免测试追随补丁

cargo-generate 的前两轮 QA 已指出：真正要求是固定依赖版本，补丁停在 `~0.16.2`，测试却弱化为“不是 0.16.1”及部分 API smoke test。第三轮 QA 改为 APPROVE，并接受离线环境的解释。这说明 paired Red–Green 能成立，却仍可能没有证明用户要求得到满足。

cargo-dist 的临时测试曾检查模板文本包含 `gh_cargo_safe`、`PATH` 等字符串。这些检查不能证明模板能渲染、PATH 判断正确或输出命令的 quoting 正确。

aya 的 mock 只证明 `LinkMap` 没有调用 mock 的 `detach()`；没有覆盖真实 fd 的行为。预测中 `FdLink::forget` 和原有 `detach` 都执行 `close(self.fd)`，实现相同。这至少说明“没有调用名为 detach 的方法”不足以证明真实 link 存续；需按具体 link 的资源语义核对释放 fd 的效果。

**具体建议：**

1. 在实施前把 issue 转为少量验收条目，每条带来源、输入、可观察结果和允许的替代行为。语义条目与实现选择分开。
2. 对验收关键断言保留变更记录。测试改了 import、fixture 可以修；测试预期从行为约束变成更弱检查时，需要独立复核这一变化，而不是重新运行出 Green 即可。
3. Red 阶段记录失败原因：运行时断言、缺少明确要求的新 API、测试自身错误、缺依赖、0 个测试、超时。只有前两类在语义确认后构成有效复现。
4. 资源生命周期、CLI、模板优先使用真实对象、真实 binary 或真实模板引擎；mock 可验证协作，但不能替代关键的资源/端到端行为。
5. 独立 QA 以 issue 和直接代码/执行证据为依据，不把 Architect 的结论或测试注释当事实。对同一已拒绝语义不因“环境做不到”而放宽契约。

bytes 是需要保留的正例：issue 允许 panic 或特定边界行为；旧测试断言了 issue 指出的错误行为。优化不能退回“任何旧测试失败就禁止提交”的绝对规则。

## 7. 优先优化四：把 API 契约与跨文件依赖变成显式义务

三个案例体现不同缺口：

- cargo-edit 的计划明确要求保留两参数 `set_git` 并新增 `set_rev`；自测只覆盖这一路。公开 issue 没有规定精确四参数接口，因此评测失败含有接口约定因素；但计划不能把一种选择未经论证地固化为禁止更改的约束。
- aya 把 `forget` 扩散到 19 个文件，但没有验证“谁持有资源、谁负责何时 detach、是否应转移 ownership”的完整公共契约。复制相同方法不等于完成资源模型设计。
- ast-grep 将改动限定到 CLI；自测从 CLI 入口成功，评测核心 crate API 却不兼容。只统计修改了几个文件，会把这类功能低估为简单问题。

现有 `build_handoff_card` 有一个具体的信息损失点：只传计划前 4 个 ordered edits、前 3 个 invariants、前 3 个 forbidden edits，且短卡没有完整 API checklist。本轮 aya 的计划实际有 13 个 ordered edits，biome 某轮有 6 个。不能把这种静态截断直接认定为本次失败的因果原因，但它确实不能保证关键契约在交接中保留。

**建议以“依赖义务”替换按顺序截断：**

```text
义务：CLI --rev → Args.rev → dependency source → TOML rev
义务：原有 git/branch 路径保持正确
义务：branch/tag/rev 组合是否合法，依据 Cargo 与项目约定检查
义务：公开方法的调用方、文档示例与返回值是否一致
状态：每条义务关联到实现文件、验证目标和当前证据
```

根因位置与依赖关系可以利用现有 `inspect_rust_neighborhood`、symbols、workspace metadata 构造；先做符号/模块级的轻量图，不必一开始实现完整语义分析。循环依赖的单元联合实施，后续再考虑 SCC 等更复杂调度。

压缩上下文应省略历史浏览和重复代码，不能省略未完成的验收义务。给 Fixer 当前单元的细节，同时保留全部必需契约的短清单；完成后再取下一个单元。

## 8. 优先优化五：把环境阻塞从“让模型想办法”中分离

cargo-generate 需要的 `git-config 0.16.3` 不在冻结替代 registry 中，而工具和提示又禁止写 `Cargo.lock`。于是 agent 在版本号、自测与解释之间来回切换。当前 environment blocker 提示甚至建议回退不可用版本；这个提示有助于避免空转，却也可能推动放弃 issue 的核心要求。

**建议：**

- 在模型调用前做确定性环境检查：实际 rustc/cargo、registry 来源、已锁版本、issue 明确涉及的依赖是否存在、相关功能是否能在当前镜像运行。
- 针对确实需要更新依赖的 issue，不应全局禁止 lockfile 更新；允许 Cargo 根据真实 registry 解析，禁止猜 checksum。评测若明确忽略 lockfile，也应把这一限制与任务本身要求一起记录。
- agent 与评测使用一致、可复现的缓存/registry 配置；缺 artifact 时先修复实验环境，或报告环境阻塞，不把降低版本要求视为修复。
- 在恢复合法环境后按原 issue 重跑，不改变冻结 manifest。

这能同时提升可验证性并减少无效请求。当前证据不能直接证明只要开放网络就能修好，需要先确定所需依赖与基准配置。

ast-grep 的 trace 实际使用 rustc 1.99.1，而 manifest 的默认版本是 1.83.0（仓库的 stable pin 优先）。这提示应比较两边**实际生效**的工具链，不能只看 CLI 参数；目前未据此断定评测与 agent 版本不一致。

## 9. 怎样引入 Multi²Fixer 的多候选而不放大成本

当前优势是 cheap fast path；应保留一个默认候选，只在证据显示不确定时扩展。

触发第二候选的情形可以是：

- 两条根因链都能解释 issue，现有证据不能区分。
- 一个依赖闭合的单元经过验证仍失败，已有假设被反证。
- 预计需要大范围 API/资源模型变化，而局部替代实现也可能满足同一契约。
- 补丁范围或测试构造成本显著增长，但新增改动没有带来新增验证证据。

优先让两个候选产出**不同机制的简短方案**，随后只实施最有证据的一个；必要时在隔离 checkpoint 验证第二个。温度不同但机制相同的补丁不算有用多样性。

候选选择优先级建议：

1. 满足 issue 契约及不可违反的资源/API 约束。
2. 有效的编译/运行证据与覆盖范围。
3. 与已接受修复单元的依赖兼容性。
4. 在前三项相当时比较改动范围、实现复杂度与验证成本。

LLM 语义比较用于补足证据缺口，不用多数票代替可执行验证。候选数初期限制为 2，不默认给每个位置 3–5 个 Proposer。

bat 值得作为成本对照：当前成功修复付出了 19 次回归测试写入、138 次请求、7 文件修改。可以比较局部消除歧义与完整 CLI 迁移等方案，但 issue 也确实建议 `--cache`；不能仅凭参考补丁更短，就断言当前更完整迁移不合理。

chrono 的手工 Eq/Ord/Hash 也不能仅因 37 行大于参考 3 行就删除：一致性是必要契约。目标是同样正确且成本更低，而不是机械最小 diff。

## 10. 降成本的具体落点

**A. 限制长期无进展的 Fixer 会话。** 当前 API_LIMIT 是请求次数上限，与每次携带多少上下文无关。本轮某些 prompt 达到 4–5 万 token。应额外统计非缓存输入、阶段 token、工具耗时、验证进展，不能只看剩余调用。

**B. 保留已有压缩，再做语义状态压缩。** `agent.py:_compress_history_messages` 已实现历史输出压缩、旧代码读取压缩和前缀缓存友好的分块保留；服务器也已有五行 handoff card。建议在失败边界或完成单元时用短状态重建后续上下文，保留关键契约、当前 diff、最近一条相关错误和未完成义务。不要每一轮重写摘要，破坏缓存前缀。

**C. 复用验证结果需有完整配置键。** 至少区分源码版本、测试内容、manifest/package/features/target、toolchain、构建输入及关键运行条件。仅在相同有效状态下省略重复检查；有状态/外部环境相关测试不能仅按命令字符串缓存。已有任何写入都失效所有验证的策略可靠但较粗，可先对无内容变化的编辑和纯格式/无关文件做准确判断，再考虑依赖范围复用。

**D. 修复重复测试构造。** 利用现有 integration test 的 process helper、fixture、assert_cmd 风格生成一次模板；限制同一测试因 crate path、private visibility、fixture 错误反复重写。允许修正测试基础设施，保留语义预期的变更记录。bat 的 19 次写入、cargo-dist 的 17 次写入是明确的优化目标。

**E. 把验证保留额改为按任务计算。** 当前 300 次预算下 Fixer 单次上限 114，而 Verifier 预留 5；tracing/biome/cargo-dist 耗尽后只剩极少请求救火。可试验保留 10–15 次收尾请求，并依据后续统计调整；未使用的保留额不会产生模型费用。更重要的是及时执行验证，而非结尾单纯多留预算。

**F. 工具反馈按失败责任分类。** 编译/类型错误→相关符号和依赖单元；测试断言→行为假设；模板解析→模板加载器；registry 不可用→环境；无效测试→测试构造。无需每类都新增 LLM 角色，可由确定性分类加现有角色处理。

不建议先省略独立 QA、统一降低所有任务上限，或把每个阶段替换为更强模型。前者节省占比低，后两者会混淆框架改进与算力/预算变化。

## 11. 实施和验证顺序

| 优先级 | 改动 | 验证目标 |
|---|---|---|
| P0 | 修复降级提交的验证缺口；模板必须经过实际加载/渲染 | cargo-dist 不再提交已知无法加载的模板；保留当前 resolved 案例 |
| P0 | 输出补丁中隔离已有测试修改；允许工作副本维护临时测试但不把这类修改混入正式预测 | biome 不再由自身 test hunk 引起官方测试补丁冲突；仍需正式判题才能知道是否 resolved |
| P0 | 核对依赖任务的 registry/lockfile 限制及 issue 契约 | cargo-generate 不再用降低版本要求获得伪 Green |
| P1 | 轻量修复单元、依赖义务、无进展升级、保留有效 checkpoint | tracing 显著减少重读与全量重做；更早拿到能指导修复的编译反馈 |
| P1 | 测试预期变更审计、资源/端到端验证、统一 API 约束 | aya/cargo-edit 的自测不再仅围绕选定实现自证；准确记录接口不兼容 |
| P1 | 按任务验证范围选择定向测试，压缩完整义务而非固定前 4 条 | ast-grep 的核心/CLI 关系不再被 fast-path 文件数掩盖 |
| P2 | 证据触发的第二方案，以及成本感知的候选选择 | 相同预算下比较不同机制；避免固定扩展 agent 数量 |

离线阶段可先用现有 trace 检查会在哪些节点触发无进展转移、哪些 QA 证据被放宽、哪些测试构造重复；这些是诊断，不能当成新策略的因果收益。

后续在线消融应逐项比较，避免一次改动多个机制：

- A0：本轮实现；当前单轮记录为 3/10、1,665 请求、32.97M token。
- A1：仅修验证/提交规则与失败分类。
- A2：在 A1 上增加依赖单元和进展调度。
- A3：在 A2 上增加上下文和验证复用。
- A4：在 A3 上增加自适应双候选。

先对失败路径及至少两个已成功任务做定向试验，再冻结实现做完整 10 任务配对运行；关键对照重复至少 3 次并报告全部运行。固定模型、模型参数、manifest、工具链/镜像、API_LIMIT 与评测配置，分别报告总 token、缓存/非缓存 token、时间和 resolved。不能把独立随机运行的最好一次用于比较。

建议接受标准是：**resolved 不因降成本而减少，且每个 solved task 的累计成本下降；复杂任务至少出现可重复的正向变化。** 具体降幅和新增通过数需由实验给出。本轮不足以承诺提升到 6/10 或降低 50% 费用。

50 任务测试集应在策略冻结后作为独立评估；这 10 任务已经用于诊断和开发，不再作为独立泛化证据。

## 12. 代码定位与证据入口

以下为本轮服务器源码位置；Windows 工作区版本与服务器存在差异，实施前应明确以哪一份为基线，不宜直接覆盖同步。

- `/home/xupeicai/xpc/baselines/test/multi_agent.py:128`：阶段预算。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:189`：导航后的 fast/planned 路由。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:325`：五行交接卡及静态截断。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:952`：降级补丁兜底。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:1013`：确定性 QA。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:1073`：已有 inline tests 与 API 修改的启发式约束。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:1407`：环境阻塞与 pacing 提示。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:2532`、`:2628`：升级重规划时恢复 `pre_fix`。
- `/home/xupeicai/xpc/baselines/test/multi_agent.py:2700`：Verifier 能读代码，但不直接执行测试；接收确定性层预先给出的证据。
- `/home/xupeicai/xpc/baselines/test/tools.py:840`、`:878`：编译/测试重复失败计数。
- `/home/xupeicai/xpc/baselines/test/tools.py:1451`：post-strip `cargo check --tests`，以及 downstream 普通编译检查。
- `/home/xupeicai/xpc/baselines/test/tools.py:1625`：每次写入使既有验证失效。
- `/home/xupeicai/xpc/baselines/test/agent.py:103`：已有历史压缩。
- `/home/xupeicai/xpc/baselines/test/agent_prompts.py:29`、`:43`、`:67`：部分角色允许有证据的签名适配，而 Verifier 仍有偏向固定旧签名/测试隔离的限制；应统一条件。

运行证据根目录：

- `/home/xupeicai/xpc/validation/runs/validation_20261114_102110_3b48da/`
- `/home/xupeicai/xpc/Rust-bench/logs/run_evaluation/validation_20261114_102110_3b48da_test/test/`

论文解读：`C:/Users/vickey/Desktop/paper-analysis/论文详细解读：Multi²Fixer——基于 Coordinator–Proposer 多智能体协作的多位置程序自动修复框架.md`，重点对应 §4 上下文、§5 调度与候选、§6 两阶段 refinement、§8.3 消融、§10 局限。
