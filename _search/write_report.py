"""Write OVERLAP_ANALYSIS.md with explicit UTF-8 encoding."""
import io
import sys

REPORT = r"""# `baselines/test` 多智能体 Rust 仓库级修复：顶会先行工作查重与重合度分析

**生成日期**：2026-11-20　**检索语料**：783 条原始记录 / 568 篇去重后文献

**数据源**：arXiv、OpenAlex、Crossref、Semantic Scholar。DBLP 全程被反爬挑战拦截（返回 0），OpenReview 本轮返回 0——两个来源的缺口见"检索局限"。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 顶会上有没有"多智能体做仓库级修复"？ | **有，且很多。** 至少 10 篇已正式发表，最接近的是 SGAgent (TOSEM 2026) |
| 顶会上有没有"多智能体做 Rust 修复"？ | **有。** AkiraRust (DAC 2026)、RustBrain (DAC 2025)、LANTERN (ICSE 2026) |
| 顶会上有没有"Rust 仓库级 issue 修复 benchmark"？ | **有，且早于本项目。** Multi-SWE-bench (NeurIPS 2025) 含 239 条 Rust 任务 |
| 本项目整体架构重合度？ | **高**。角色分工、图路由、快路径、复现测试、验证环节均已被覆盖 |
| 本项目还能主张什么？ | **Rust 特化的问题设定** + **若干具体护栏机制的组合**，但每个单点机制都有近邻 |
| 最重要的一条 | **本项目自己的 ICSE 2026 论文已发表**——`test` 是相对自家论文的增量，不能作为"首个 Rust 仓库级工作"来包装 |

---

## 1. 必须先纠正的三个定位问题

### 1.1 本项目已有一篇 ICSE 2026 论文，`test` 是增量而非首发

> **Evaluating and Improving Automated Repository-Level Rust Issue Resolution with LLM-based Agents**
> Jiahong Xiang, Wenxiao He, Xihua Wang, Hongliang Tian, Yuqun Zhang
> ICSE 2026, DOI [10.1145/3744916.3773108](https://doi.org/10.1145/3744916.3773108) · arXiv [2602.22764](https://arxiv.org/abs/2602.22764)

该论文包含 **Rust-SWE-bench**（500 任务 / 34 仓库）与 **RustForger**（`syn`/`quote` AST 过程宏插桩 + `trace_cli` 动态追踪，Claude-3.7-Sonnet 解决率 28.6%），评测的四个基线是 SWE-agent、OpenHands+CodeAct v2.1、Agentless、AutoCodeRover。

**关键结论**：这篇论文**没有**包含 `baselines/test` 的多智能体图系统。`test` 相对于自家已发表成果是**尚未发表的新增量**，这是它目前唯一干净的 novelty 空间。论文见刊（2026-04-12）到今天（2026-11-20）约 7 个月，**这个窗口正在关闭**——SGAgent (TOSEM 2026, online 2026-11-04)、MultiFixer (ASE 2026) 等近邻已经进场。

### 1.2 "首个 Rust 仓库级 benchmark"的说法不成立

**Multi-SWE-bench** (NeurIPS 2025, DOI [10.52202/085713-2111](https://doi.org/10.52202/085713-2111), arXiv [2504.02605](https://arxiv.org/abs/2504.02605)) 覆盖 Java/TS/JS/Go/**Rust**/C/C++，共 1,632 条。其论文 Table 2 的 Rust 分布经原文核对为：

| 难度 | Rust 任务数 |
|---|---:|
| Easy | 66 |
| Medium | 126 |
| Hard | 47 |
| **合计** | **239** |

比本项目（2026-02）早约 10 个月。**Rust-SWE-bench 的可辩护说法是"更大规模、Rust 专项、500 题 / 34 仓库"，而不是"首个"。**

顺带一提，Multi-SWE-bench 的 Rust 解决率极低（Agentless+GPT-4o 仅 5.86%，o1 仅 7.11%）——这**支持**了本项目"Rust 仓库级修复难"的前提论证，是可引用的有利证据。

### 1.3 "多智能体 + Rust"组合本身已被顶会占据

| 论文 | 出处 | 与本项目的交集 | 但任务层差异 |
|---|---|---|---|
| **AkiraRust** | DAC 2026, DOI [10.1145/3770743.3804397](https://doi.org/10.1145/3770743.3804397) | Rust + 多 agent + FSM + 执行反馈 + 回滚 | 任务是 Miri 可检测的 47 种 UB，非 GitHub 仓库 issue |
| **RustBrain** | DAC 2025, DOI [10.1109/dac63849.2025.11133064](https://doi.org/10.1109/dac63849.2025.11133064) | Rust + 多 agent + 快慢思考 + adaptive rollback | 同上，UB 片段级 |
| **LANTERN** | ICSE 2026, DOI [10.1145/3744916.3773227](https://doi.org/10.1145/3744916.3773227) | 多 agent + Rust 修复 | 跨语言题目级（xCodeEval），非仓库级 |
| **Crustify** | arXiv [2611.10394](https://arxiv.org/abs/2611.10394)（预印本） | 仓库级 Rust + 依赖排序分波次 + 并发 agent + 覆盖引导测试 | C→Rust FFI 绑定生成，非 issue 修复 |

---

## 2. 逐机制查重：把 `test` 拆成可检验的部件

`baselines/test` 的实际结构（经代码级核实）：**顺序协作 + 动态回退**的四角色系统，LangGraph `StateGraph` + `Command` 实现，确定性协调器负责路由/预算/修复单元/证据/部分 QA。

### 2.1 重合度高的部件（不可作为独立贡献主张）

| 机制 | 最接近的先行工作 | 重合判断 |
|---|---|---|
| **四角色分工**（Navigator→Architect→Fixer→Verifier） | **MAGIS** (NeurIPS 2024, DOI [10.52202/079017-1647](https://doi.org/10.52202/079017-1647)) 的 Manager/Repository Custodian/Developer/QA | **几乎同构**。角色数、职责切分、共享黑板模式一致 |
| **定位→规划→修复→验证的 localize-then-fix 范式** | **SGAgent** (TOSEM 2026) 明确以 "localize-then-fix 有 reasoning gap" 为卖点提出 **localize-suggest-fix**；CodeR / MASAI / HyperAgent 同属此范式 | **完全覆盖**。SGAgent 还多了一个 suggester |
| **图路由 + 简单任务快路径** | **CodeR** (arXiv [2406.01304](https://arxiv.org/abs/2406.01304)，预印本)：预定义 task graph，Plan A 标准流 / Plan B 直接解决简单任务 / Plan C 测试反馈循环 | **覆盖**。`route_after_navigation` 的 Fast Path 与之同构 |
| **仓库级图/知识图谱定位** | **SGAgent** 自建 KG（3 entity / 7 relation / 14 tools）；**ARISE** (arXiv [2605.03117](https://arxiv.org/abs/2605.03117)) 仓库级程序图 + data-flow slicing；**HGARena** (COMPSAC 2026) typed RepoGraph；**SWE-Debate** (ICSE 2026, DOI [10.1145/3744916.3787810](https://doi.org/10.1145/3744916.3787810)) 依赖图生成多条 fault propagation trace | **覆盖**。`rust_ckg.py` 是 Rust 词法近似版，与 SGAgent 的 KG 工具箱定位相似 |
| **多故障链候选 + 比较** | **SWE-Debate**（依赖图生成多条 fault propagation trace → 辩论 → 汇总）；**EnAgent** (ICSE 2026, DOI [10.1145/3744916.3787761](https://doi.org/10.1145/3744916.3787761)) ensemble generation/pruning/selection | **覆盖**。`repair_candidates.py` 的按需 A/B 机制卡是 SWE-Debate 辩论的轻量化、确定性版本 |
| **依赖闭包修复单元 + 修复顺序 + 部分回滚** | **MultiFixer** (ASE 2026, DOI [10.1145/3832783.3837527](https://doi.org/10.1145/3832783.3837527), arXiv [2607.26591](https://arxiv.org/abs/2607.26591))：明确讨论 hunk 间依赖、决定修复顺序、**producer hunk 必须先于 consumer hunk**、部分补丁单独验证会被依赖遮蔽 | **高度重合**。这是 `repair_units.py` 最危险的近邻 |
| **Red-Green 复现测试生成** | **CodeR** 的 Reproducer、**MASAI** 的 Issue Reproducer + Test Template Generator、**Multi-SWT-Bench** (arXiv [2609.34752](https://arxiv.org/abs/2609.34752)) 专门做 1,963 条跨语言（含 Rust）复现测试基准 | **覆盖** |
| **rustc 诊断分类与符号定位** | **RustAssistant** (ICSE 2025, DOI [10.1109/icse55347.2025.00022](https://doi.org/10.1109/icse55347.2025.00022)) 迭代修 Rust 编译错误；**CiL-Agent** (ASE 2026) compiler-in-the-loop | **覆盖**，但 Rust 专项化程度可区分 |
| **运行时动态追踪辅助定位** | **CT-Repair** (arXiv [2607.12605](https://arxiv.org/abs/2607.12605)) CPG + Temporal Execution Graph + 三视角 FSM agent；AkiraRust/RustBrain 执行反馈 | **中高重合** |
| **多角色 + 预算约束 + Docker 测试为唯一验收** | **HGARena** (COMPSAC 2026)：typed RepoGraph + submitter/reviewer/executor 循环 + 20 次 LLM 调用预算 | **覆盖** |

### 2.2 未检索到同构先行工作的部件（潜在可主张点，但须实证）

以下三点经检索**未找到机制等同的先行工作**。注意："未检索到"不等于"已证明首创"，需要更彻底的检索与实证支撑。

1. **Red 阶段测试的 SHA-256 防篡改锁**——确认在原始代码上失败的临时回归测试被冻结，`write_regression_test` 拒绝改写/重命名、`revert_file` 拒绝删除，违规则清除该测试的 Red 证据；Architect 可附证据释放锁。
   - 近邻：Agent-CoEvo (arXiv [2604.04580](https://arxiv.org/abs/2604.04580)) 主张 code 与 test **共同演化**、明确反对"测试是不可变 oracle"；PROBE (ASE 2026, DOI [10.1145/3832783.3837534](https://doi.org/10.1145/3832783.3837534)) 用程序变体找弱测试。两者都触及防 reward-hacking，但机制不同——**Agent-CoEvo 实际上是对本项目该设计的方法论挑战**，必须在相关工作中正面回应。

2. **版本绑定的执行证据账本**——Cargo 结果绑定 `candidate_evidence_id` + 源码 revision + candidate + episode；同配置成功重跑可取代历史失败；**零测试运行不构成行为证据**。
   - 近邻：SWE-ABS (arXiv [2603.00520](https://arxiv.org/abs/2603.00520)) 指出 top-30 agent 的"已解决"patch 有 1/5 语义错误，仅因弱测试未暴露；AgentLens (arXiv [2605.12925](https://arxiv.org/abs/2605.12925)) 揭示 "lucky pass" 问题。这些是**动机**来源，不是同构机制。

3. **受限 Verifier 探针 + 证据化拒绝规则**——Verifier 只能在 `existing_test` / `assertion` / `snapshot` 三类探针内取证；prompt 要求对已通过 Red-Green 的补丁，**必须引用具体探针结果/同级实现行/编译诊断/失败测试名/snapshot 差异**才能拒绝，投机性疑虑不得触发拒绝，与探针结果矛盾时必须批准。
   - 近邻：HGARena 的 reviewer 环节、AdverIntent-Agent 的对抗推理。**"证据化拒绝"这一强约束形式**未见同构先例。

### 2.3 重合度特征矩阵

不给伪精确的"重合 x%"，改用可检验的分级：

| 特征维度 | 重合度 | 依据 |
|---|---|---|
| 问题设定（Rust + 真实 GitHub 仓库 issue） | **低（最具区分度）** | Rust 仓库级 issue 修复的顶会工作基本只有本项目自家 ICSE'26；Multi-SWE-bench 是 benchmark 不是 agent |
| 整体架构（多角色 + 图路由 + 快路径） | **高** | MAGIS / CodeR / SGAgent / HyperAgent / MASAI 全覆盖 |
| 定位机制（仓库图 / 故障链） | **高** | SGAgent KG / ARISE / SWE-Debate / HGARena |
| 候选机制比较 | **高** | SWE-Debate / EnAgent |
| 修复单元 / 依赖调度 / 部分回滚 | **中高** | MultiFixer (ASE 2026) 是最危险的近邻 |
| Rust 特化（编译器诊断 / 所有权 / trait / 生命周期） | **低-中** | RustAssistant 只做编译错误；AkiraRust/RustBrain 只做 UB |
| 护栏机制（Red 锁 / 证据账本 / 受限验证探针） | **中（未见同构，但需实证）** | 见 2.2 |

**综合判断**：**架构相似度高，机制重合中高，Rust 特化与护栏组合是唯一可辩护的差异点。**

---

## 3. 检索方法与局限

### 3.1 检索执行

- 6 组查询 × 5 个查询源，跨 arXiv / OpenAlex / Crossref / Semantic Scholar 并发，共 **783 条原始记录**，去重后 **568 篇**。
- 另经 arXiv API 与 Crossref API **逐条一手核验**了本文列出的全部关键论文的标题、作者、年份、venue、DOI/arXiv ID 与摘要。
- 语料年份分布：2026 年 224 篇、2025 年 112 篇、2024 年 62 篇、2023 年 34 篇。

### 3.2 局限（必须诚实标注）

- **DBLP 全程失败**：返回 Anubis 反爬挑战页，本轮 0 条命中。DBLP 是 SE 顶会最权威的索引，这是真实缺口。
- **OpenReview 本轮 0 条**：`openreview-py` 已安装，但 NeurIPS/ICML/ICLR 的 workshop 与 rejected 论文可能遗漏。
- **Semantic Scholar 大量 429 限流**：多批查询失败，最终仅覆盖 63 篇。
- 因此，**顶会覆盖是"已核验的下界"，不是穷尽**。投稿前必须补做 DBLP + OpenReview + Semantic Scholar 的完整检索。

### 3.3 本项目自身的实验数据边界

`testset_results_20261118/` 记录了 50 任务历史运行（模型 `gpt-6-luna`，API 上限 150）：

| Agent | resolved | unresolved | no_patch | agent_error | error | total |
|---|---:|---:|---:|---:|---:|---:|
| **test** | **23** | 23 | 0 | 1 | 3 | 50 |
| graph | 22 | 25 | 0 | 1 | 2 | 50 |
| graph-2 | 19 | 23 | 0 | 6 | 2 | 50 |
| normal | 16 | 33 | 0 | 0 | 1 | 50 |
| ACI-impr | 13 | 18 | 0 | 18 | 1 | 50 |

三点必须注意：

1. **agent_error 严重不均**（ACI-impr 18/50 vs normal 0/50），`resolved` 绝对值不能直接横向比较，需按 error 归一化或排除后重报。
2. **与自家 ICSE'26 论文不可直接比较**：模型不同（Claude-3.7-Sonnet vs gpt-6-luna）、任务集不同（500 题 vs 50 题）、RustForger 未在 50 题上重跑。
3. `testset/README.md` 明确说明：该 50 题按参考补丁大小选取，**不能仅凭与验证集不重叠就称其为未接触过的独立测试集**；且当前源码与历史运行的 source SHA 需核验。

另外，`test` 与 `ACI-impr` 共享大量 Rust 定向工具、结构化 Cargo 诊断、回归测试与 Red-Green 检查。**多智能体带来的独立增益必须单独消融**，否则消融结论不成立。

---

## 4. 对后续研究的建议

### 建议 1（最高优先）：立刻把 novelty 声明收窄到"Rust 特化 + 护栏机制"，并正面回应 Agent-CoEvo

不要主张"首个 Rust 仓库级多智能体修复"——Multi-SWE-bench (NeurIPS 2025) 的 Rust 子集、AkiraRust/RustBrain/LANTERN 的 Rust 多智能体修复都已经存在。

改为主张：**"面向 Rust 严格类型/trait/生命周期语义的、仓库级 issue 修复系统，其中修复的可信性由一组确定性护栏保证（Red 阶段测试防篡改、版本绑定执行证据、受限验证探针的证据化拒绝）"**。

**必须正面回应**：Agent-CoEvo 主张"仓库级修复不是在固定测试下优化，而是对演化行为约束的搜索"，并批评"把测试当作不可变 oracle"会导致欠约束搜索和过拟合修复。你的 Red 阶段锁在方法论上正是"把测试当作不可变 oracle"。这不是可以绕开的问题——需要一个受证据支撑的回应，例如：

- 区分"**issue 契约**"与"**实现细节**"：锁的是**复现 issue 的语义契约**，而允许在契约不变的前提下**补充**测试；
- 或论证**分层 oracle**：Red 测试锁不可变（防篡改），但 Architect 可附证据释放并重建（你已有 `release_locked_regression`），这恰是"演化约束"的受控版本；
- 或实证展示：锁与不锁在**弱测试子集**上的解决率差异，并用 SWE-ABS 的对抗增强流程来评估。

### 建议 2：把 `repair_units` 对标 MultiFixer 做差异化，并把消融做扎实

MultiFixer (ASE 2026) 已覆盖"多 hunk 依赖 + 修复顺序 + producer 先于 consumer + 部分补丁验证被依赖遮蔽"。你的差异点必须精确到：

- **依赖闭包单元的合并规则**（同文件重叠 / 依赖环 → 合并为联合验证单元）；
- **文件级定向回滚**（只回滚失败单元及其 dependents，不动无关已验证单元）；
- **Rust 特异的单元边界**——trait impl 与调用方必须同单元、枚举变体与 match 臂必须同单元、doctest 行号中性约束。

同时，MultiFixer 在 Defects4J 上评的是 **Java**，而你在 Rust 上受 43.7% 路径解析错误、32.6% trait/类型契约错误、44.5% 复现阶段失败率支配。**"Rust 的依赖闭包切分规则与 Java 的 hunk 依赖不同"是一个可实证、可辩护的研究问题**，值得单独成节。

### 建议 3：补做 DBLP / OpenReview / Semantic Scholar 的穷尽检索，并做正式的新颖性审计

本轮 DBLP 完全失效（反爬）、Semantic Scholar 大量限流。在投稿前必须补齐，且要按 **ICSE / FSE / ASE / ISSTA / ICSE-SEIP / TOSEM / OSDI / SOSP** 逐个 venue 扫一遍 + 逐篇核对引用。

### 建议 4：把"实验设计"升级为"可信性评测"，这是最可能的新赛道

文献中有一个正在成型的、**你尚未进入但极易进入**的方向：SWE-bench 类基准的成功率被系统性高估。

- **SWE-ABS** (arXiv [2603.00520](https://arxiv.org/abs/2603.00520))：top-30 agent 的"已解决"patch 中 **1/5 语义错误**，仅因弱测试未暴露；增强测试后 top agent 从 78.80% → 62.20%。
- **AgentLens** (arXiv [2605.12925](https://arxiv.org/abs/2605.12925))：揭示 "lucky pass"。
- **"Are 'Solved Issues' in SWE-bench Really Solved Correctly?"** (ICSE 2026, DOI [10.1145/3744916.3773269](https://doi.org/10.1145/3744916.3773269))。
- **"What's in a Benchmark? The Case of SWE-Bench in Automated Program Repair"** (ICSE 2026)。

**你的位置无可替代**：Rust-SWE-bench 已经在 ICSE'26 建立了 500 题基准，`test` 是**第一个能把"红阶段测试防篡改"做成系统机制的 agent**。把这两者结合——**用基准评测基准自身的可信性**，即"在 Rust-SWE-bench 上，用受防篡改保护的 agent 产生/验证的回归测试来对抗性增强 Rust-SWE-bench，再测量各 agent 的解决率下降幅度"——这是一个**几乎没有竞争者的方向**，且能一次性把 Rust-SWE-bench、test agent、Red 锁三个组件变成一个连贯的研究故事。

### 建议 5：用 Multi-SWT-Bench 做 `test` agent 的外部验证集

**Multi-SWT-Bench** (arXiv [2609.34752](https://arxiv.org/abs/2609.34752)) 提供 1,963 条跨 8 语言（含 **Rust**）的**复现测试生成**基准。这正好是你 Navigator 角色（写临时回归测试、确认 Red 阶段）的能力标尺，而且它与你的主基准**不重叠**——这解决了 `testset/README.md` 里"不能声称独立未接触测试集"的历史包袱。

具体做法：在 Multi-SWT-Bench 的 Rust 子集上单独报告"复现测试生成成功率"（而非解决率），这会让 `test` 有一份**独立于自家 50 题历史运行**的能力证据。

### 建议 6：把 Rust CKG 从"词法近似"升级为"编译器精确"，这可能是最稳的增量

你 CKG 目前是 token-aware 词法近似（README 与 `rust_ckg.py` 自述如此），而 ARISE 已用**真正的 data-flow slicing**（statement 级 def-use 边）把 SWE-agent 的 Function R@1 从 0.43 提到 0.60，**+4.7pp 解决率**。这说明"图表示精度 → 定位精度 → 解决率"的因果链已经被验证过。

Rust 侧有一个无可替代的抓手：**`cargo check` 的 JSON 诊断 + `rustc` 的 trait/impl 解析**可以做到**编译器精确的 trait 实现解析与可见性判定**，而这正好命中 Rust-SWE-bench 的两大痛点（43.7% 路径解析 + 32.6% trait 契约）。把 `rust_ckg.py` 从正则/词法升级到 `rust-analyzer`/HIR 或 rustc 私有接口驱动，**是一个投入明确、收益可量化、且与所有先行工作拉开差距的方向**——因为 SGAgent/ARISE/HGARena 全都是语言无关的图，而 Rust 特异的编译器级图，目前没有人在做。

### 建议 7：发布前必须修正的表述清单

| 不要写 | 改成 |
|---|---|
| "首个 Rust 仓库级 benchmark" | "规模最大的 Rust 专项仓库级基准（500 题 / 34 仓库）；Multi-SWE-bench (NeurIPS 2025) 此前含 239 条 Rust 任务" |
| "首个多智能体 Rust 修复" | "面向 Rust 仓库级 issue 的多智能体修复系统"，与 AkiraRust/RustBrain/LANTERN 明确区分任务层 |
| "首创多智能体修复架构" | "在已有 localize-then-fix 范式上，引入确定性护栏保证 Rust 特异性修复的可信性" |
| 直接引用 28.6% 作为本系统成绩 | 明确标注那是 RustForger（ICSE'26），`test` 的成绩需独立重跑 |
| 用 50 题 `resolved` 绝对值横向比较 | 先做 agent_error 归一化，并在 500 题全集上重跑 |

---

## 5. 推荐阅读路径

**必读（直接决定能否过审）**

1. **SGAgent** — TOSEM 2026, DOI [10.1145/3818617](https://doi.org/10.1145/3818617) · arXiv [2602.23647](https://arxiv.org/abs/2602.23647)。最接近的设计近邻：localize-suggest-fix + 仓库 KG + 14 个 KG 工具。
2. **MultiFixer** — ASE 2026, DOI [10.1145/3832783.3837527](https://doi.org/10.1145/3832783.3837527) · arXiv [2607.26591](https://arxiv.org/abs/2607.26591)。`repair_units` 的直接威胁：多 hunk 依赖、修复顺序、部分补丁验证遮蔽。
3. **Agent-CoEvo** — arXiv [2604.04580](https://arxiv.org/abs/2604.04580)。方法论挑战者：反对"测试是不可变 oracle"。
4. **MAGIS** — NeurIPS 2024, DOI [10.52202/079017-1647](https://doi.org/10.52202/079017-1647)。四角色结构的原型。

**必读（动机与评测设计）**

5. **SWE-ABS** — arXiv [2603.00520](https://arxiv.org/abs/2603.00520)。为什么你的 Red 锁和证据账本应该存在。
6. **"Are 'Solved Issues' in SWE-bench Really Solved Correctly?"** — ICSE 2026, DOI [10.1145/3744916.3773269](https://doi.org/10.1145/3744916.3773269)。
7. **Multi-SWT-Bench** — arXiv [2609.34752](https://arxiv.org/abs/2609.34752)。你 Navigator 角色的外部验证集。

**需要知道存在**

8. **SWE-Debate** — ICSE 2026, DOI [10.1145/3744916.3787810](https://doi.org/10.1145/3744916.3787810)。故障传播链 + 辩论 = 你的候选比较机制。
9. **CodeR** — arXiv [2406.01304](https://arxiv.org/abs/2406.01304)。图路由 + 快路径 + 复现测试的出处。
10. **ARISE** — arXiv [2605.03117](https://arxiv.org/abs/2605.03117)。仓库级图 + data-flow slicing 的精度标杆。
11. **AkiraRust** — DAC 2026, DOI [10.1145/3770743.3804397](https://doi.org/10.1145/3770743.3804397)。Rust + 多 agent + FSM + 回滚。
12. **EnAgent** — ICSE 2026, DOI [10.1145/3744916.3787761](https://doi.org/10.1145/3744916.3787761)。集成式候选搜索。

---

## 附：本仓库内的对照材料

- `baselines/test/README.md` — `test` agent 的完整机制说明（Red 锁、Verifier 探针、CKG 注入、修复单元、候选比较门控）
- `paper_text.txt` / `extracted_paper_text.txt` — 本项目 ICSE 2026 论文全文（确认不含 `test` agent）
- `improvment_record.md` — ACI 优化记录，含 Rust 错误码分布（43.7% 路径解析 / 32.6% trait 契约 / 44.5% 复现失败）与两轮优化明细
- `testset_results_20261118/` — 50 题历史运行结果与元数据（模型、API 上限、source SHA）
- `testset/README.md` — 任务清单构造规则与"不可声称独立测试集"的明确声明
"""

out = io.open(r"C:\Users\vickey\Desktop\Rust-SWE-Bench-main\OVERLAP_ANALYSIS.md", "w", encoding="utf-8")
out.write(REPORT)
out.close()
sys.stdout.write("written OK\n")
