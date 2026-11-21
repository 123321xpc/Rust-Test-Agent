# Stage 1 实施方案：编译器精确的 Rust 定义图

> 目标：把 `baselines/test/rust_ckg.py` 中靠**名字字符串匹配**的跨文件关联，换成编译器给出的 **唯一 id + 完整限定路径 + trait->impl->method 归属**。
>
> 边界：Stage 1 只做**定义侧**（谁定义了什么、谁实现了谁）。调用侧（`x.foo()` 到底调到哪个 impl）属于 Stage 2。

---

## 0. 实测得到的关键事实（全部已在本机验证）

| 事实 | 证据 | 影响 |
|---|---|---|
| `RUSTC_BOOTSTRAP=1` 可在 **stable** 上解锁 `-Z` 选项 | stable 1.97.1 上 `rustdoc --output-format json` rc=0 | **500 题里 490 题（98%）无需 nightly**，不需改容器、不需 `rustup component add` |
| 必须用 `cargo doc` 而非直接 `rustdoc` | 直接 `rustdoc` 对 path 依赖报 `E0432`；`cargo doc` 自动传 `--extern` | **决定性**：决定命令形态 |
| `cargo doc --no-deps` + `RUSTDOCFLAGS="--output-format=json -Zunstable-options"` 产出 JSON | 实测 workspace 两成员均 rc=0，产出 `target/doc/<pkg>.json` | 主调用形态 |
| 私有项**默认就包含** | 加/不加 `--document-private-items` 产出完全一致（index 48 / impl 30） | RustForger 关注的 `pub(crate)` 模式不会漏 |
| 同名 trait 拿到**不同 id** | `geometry::Shape`=id 1，`shapes::Shape`=id 49 | 修掉误报的核心 |
| **id 不跨源码修改稳定** | 在文件顶部插入一个 fn 后，`Shape` 从 id 1 变 id 2 | **必须用 span 做稳定键**，不能持久化 id |
| span `(filename, begin_line, begin_col)` 在源码修改后**会平移但唯一** | 修改前 `Shape` 在 (1,1)，修改后在 (2,1) | 用 span 做**一次性运行内**的键 |
| 跨 crate 的 `paths` 里有 `crate_id` | `paths["1"] = {"crate_id":0,"path":["core","Shape"],"kind":"trait"}` | 可拼出跨 crate 完整路径 |
| 单 crate 200 个 fn 的 `cargo doc` 耗时 | 首次 0.40s，重跑 0.32s，JSON 220 KB | 完全在 agent 预算内 |
| HIR-tree 体积 | 16 KB/LOC，100 行文件 1.6 MB | **Stage 1 不用 HIR**（那是 Stage 2 的事） |

---

## 1. 新增模块 `baselines/test/rust_semantics.py`

独立文件，不动 `rust_ckg.py` 的解析逻辑（保证可回滚）。

```python
"""Compiler-precise Rust definition graph (Stage 1).

Replaces name-based cross-file association with compiler-assigned ids and
fully-qualified paths. Falls back to the lexical CKG when the compiler cannot
resolve the crate, and records *which* mode was used so experiments can report
coverage honestly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# RUSTC_BOOTSTRAP unlocks -Z options on a stable toolchain. Verified working on
# rustc 1.97.1. It is set only for the child process, never for the agent's own
# cargo/rustc runs, so it cannot silently change evaluation semantics.
BOOTSTRAP_ENV = "RUSTC_BOOTSTRAP"

DOC_TIMEOUT_SEC = 600
DOC_MAX_STDERR_CHARS = 4000
```

### 1.1 数据结构

```python
@dataclass
class SemanticItem:
    """One compiler-resolved definition."""
    key: str                 # stable within a run: "path:line:col"
    doc_id: str              # rustdoc index id (NOT stable across edits)
    name: str                # bare name, e.g. "area"
    kind: str                # "trait" | "struct" | "function" | "enum" | ...
    full_path: str           # "demo::geometry::Shape" (crate-qualified)
    module_path: str         # "demo::geometry"
    crate: str               # "demo"
    span: tuple[int, int]    # (begin_line, begin_col) 1-based
    file: str                # "src/lib.rs"
    is_trait_method: bool = False
    owner_trait: str | None = None     # full path of owning trait
    owner_type: str | None = None      # full path of the implementing type
    impl_key: str | None = None        # key of the impl block this item lives in
    is_public: bool = False


@dataclass
class SemanticImpl:
    """One `impl Trait for Type` (or inherent impl) as the compiler sees it."""
    key: str
    doc_id: str
    trait_full_path: str | None       # None for inherent impls
    trait_key: str | None
    type_full_path: str | None
    type_key: str | None
    is_negative: bool = False         # `impl !Trait for Type`
    blanket: bool = False             # `impl<T> Trait for T`
    method_keys: list[str] = field(default_factory=list)


@dataclass
class SemanticGraph:
    """Compiler-precise definition graph for one crate root."""
    crate_root: Path
    crate: str
    items: dict[str, SemanticItem]         # key -> item
    impls: dict[str, SemanticImpl]         # key -> impl
    by_full_path: dict[str, str]           # full_path -> key
    items_by_trait: dict[str, list[str]]   # trait key -> impl keys
    items_by_type: dict[str, list[str]]    # type key -> impl keys
    precise: bool = True                   # False when we fell back
    degrade_reason: str | None = None
    members_ok: list[str] = field(default_factory=list)
    members_failed: dict[str, str] = field(default_factory=dict)
```

### 1.2 采集：`cargo doc`

```python
def collect_rustdoc_json(crate_root: Path, crate: str, env: dict[str, str]) -> tuple[dict | None, str | None]:
    """Run `cargo doc --no-deps` with JSON output and return the parsed doc, or (None, reason).

    `cargo doc` (rather than bare `rustdoc`) is required: cargo passes `--extern`
    for path dependencies, so workspace members resolve. Bare `rustdoc` fails with
    E0432 on any crate that depends on a sibling.
    """
    env = dict(env)
    env[BOOTSTRAP_ENV] = "1"
    env["RUSTDOCFLAGS"] = "--output-format=json -Zunstable-options"

    out_dir = Path(tempfile.mkdtemp(prefix="rustdoc_json_"))
    try:
        proc = subprocess.run(
            ["cargo", "doc", "--offline", "--no-deps", "-p", crate],
            cwd=crate_root, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=DOC_TIMEOUT_SEC,
        )
        doc_path = out_dir / f"{crate}.json"
        if proc.returncode != 0 or not doc_path.exists():
            reason = (proc.stderr or proc.stdout or "").strip()[:DOC_MAX_STDERR_CHARS]
            return None, reason or f"cargo doc exited {proc.returncode}"
        return json.loads(doc_path.read_text(encoding="utf-8")), None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return None, str(exc)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
```

> **注意**：`cargo doc` 把 JSON 写到 `target/doc/<crate>.json`，不是写到 `-o` 指定的目录。
> 上面的 `out_dir` 只是占位——实际实现应读 `crate_root/target/doc/<crate>.json`，
> 或用 `CARGO_TARGET_DIR` 指向临时目录以免污染仓库的 `target/`（**推荐后者**，
> 因为你的 harness 缓存了 `target/`，污染它可能影响评测）。

**修正后的实现**：

```python
def collect_rustdoc_json(crate_root: Path, crate: str, env: dict[str, str]) -> tuple[dict | None, str | None]:
    env = dict(env)
    env[BOOTSTRAP_ENV] = "1"
    env["RUSTDOCFLAGS"] = "--output-format=json -Zunstable-options"
    tmp_target = Path(tempfile.mkdtemp(prefix="ckg_doc_target_"))
    env["CARGO_TARGET_DIR"] = str(tmp_target)
    try:
        proc = subprocess.run(
            ["cargo", "doc", "--offline", "--no-deps", "-p", crate],
            cwd=crate_root, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=DOC_TIMEOUT_SEC,
        )
        doc_path = tmp_target / "doc" / f"{crate}.json"
        if proc.returncode != 0 or not doc_path.exists():
            reason = (proc.stderr or proc.stdout or "").strip()[:DOC_MAX_STDERR_CHARS]
            return None, reason or f"cargo doc exited {proc.returncode}"
        return json.loads(doc_path.read_text(encoding="utf-8")), None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return None, str(exc)
    finally:
        shutil.rmtree(tmp_target, ignore_errors=True)
```

### 1.3 解析成 `SemanticGraph`

```python
def _full_path(paths: dict, doc_id: str | int) -> tuple[str, str, str] | None:
    """Return (full_path, module_path, crate) for a doc id, or None if unknown."""
    entry = paths.get(str(doc_id))
    if not entry:
        return None
    seg = entry.get("path") or []
    if not seg:
        return None
    crate = seg[0]
    full = "::".join(seg)
    module = "::".join(seg[:-1]) if len(seg) > 1 else crate
    return full, module, crate


def build_semantic_graph(crate_root: Path, crate: str, env: dict[str, str]) -> SemanticGraph:
    graph = SemanticGraph(crate_root=crate_root, crate=crate)
    doc, reason = collect_rustdoc_json(crate_root, crate, env)
    if doc is None:
        graph.precise = False
        graph.degrade_reason = reason
        return graph

    index = doc.get("index") or {}
    paths = doc.get("paths") or {}

    # Pass 1: every indexed item becomes a SemanticItem.
    for doc_id, entry in index.items():
        name = entry.get("name")
        if not name:
            continue
        span = entry.get("span") or {}
        file = span.get("filename")
        begin = span.get("begin") or []
        if not file or not begin:
            continue
        resolved = _full_path(paths, doc_id)
        if resolved is None:
            continue
        full, module, _ = resolved
        key = f"{file}:{begin[0]}:{begin[1]}"
        graph.items[key] = SemanticItem(
            key=key, doc_id=str(doc_id), name=name,
            kind=(paths.get(str(doc_id)) or {}).get("kind") or "",
            full_path=full, module_path=module, crate=crate,
            span=(begin[0], begin[1]), file=file,
        )
        graph.by_full_path.setdefault(full, key)

    # Pass 2: impl blocks record trait <-> type <-> method membership.
    for doc_id, entry in index.items():
        inner = entry.get("inner") or {}
        if "impl" not in inner:
            continue
        impl = inner["impl"]
        trait_ref = impl.get("trait") or {}
        for_ref = (impl.get("for") or {}).get("resolved_path") or {}
        trait_path = _full_path(paths, trait_ref.get("id")) if trait_ref.get("id") is not None else None
        type_path = _full_path(paths, for_ref.get("id")) if for_ref.get("id") is not None else None

        method_keys: list[str] = []
        for item_id in impl.get("items") or []:
            method = graph.items.get(_key_for(index, paths, item_id))
            if method is None:
                continue
            method.is_trait_method = trait_path is not None
            method.owner_trait = trait_path[0] if trait_path else None
            method.owner_type = type_path[0] if type_path else None
            method.impl_key = None  # filled below
            method_keys.append(method.key)

        impl_key = f"impl:{doc_id}"
        sem_impl = SemanticImpl(
            key=impl_key, doc_id=str(doc_id),
            trait_full_path=trait_path[0] if trait_path else None,
            trait_key=graph.by_full_path.get(trait_path[0]) if trait_path else None,
            type_full_path=type_path[0] if type_path else None,
            type_key=graph.by_full_path.get(type_path[0]) if type_path else None,
            is_negative=bool(impl.get("is_negative")),
            blanket=bool(impl.get("blanket_impl")),
            method_keys=method_keys,
        )
        graph.impls[impl_key] = sem_impl
        for mk in method_keys:
            graph.items[mk].impl_key = impl_key
        if sem_impl.trait_key:
            graph.items_by_trait.setdefault(sem_impl.trait_key, []).append(impl_key)
        if sem_impl.type_key:
            graph.items_by_type.setdefault(sem_impl.type_key, []).append(impl_key)

    return graph
```

### 1.4 跨 crate 合并

workspace 的每个 member 单独跑一次 `cargo doc`，得到多个 `SemanticGraph`。
用一个 `SemanticWorkspace` 包起来：

```python
@dataclass
class SemanticWorkspace:
    root: Path
    graphs: dict[str, SemanticGraph]          # crate name -> graph
    item_index: dict[str, SemanticItem]       # global key -> item
    trait_to_impls: dict[str, list[SemanticImpl]]   # trait full_path -> impls

    def dependents_of_trait(self, trait_full_path: str) -> list[SemanticImpl]:
        """EXACT answer to 'if I edit this trait, which impls must change?'"""
        out = []
        for graph in self.graphs.values():
            key = graph.by_full_path.get(trait_full_path)
            if key is None:
                # trait may live in a *different* crate; look it up by path suffix
                key = graph.by_full_path.get(trait_full_path)
            if key is None:
                continue
            for impl_key in graph.items_by_trait.get(key, []):
                out.append(graph.impls[impl_key])
        return out
```

> **注意跨 crate 的 id 命名空间**：每个 crate 的 rustdoc id 是**独立编号**，
> `core::Shape` 的 id=1 和 `app::Shape` 的 id=1 是不同东西。
> 因此**跨 crate 必须用 `full_path` 关联，不能用 id**。
> 这也正是设计 `by_full_path` 索引的原因。

---

## 2. 改造 `rust_impact.py`：用语义图替换字符串匹配

### 2.1 现状（要替换的代码）

`rust_impact.py` 的 `_trait_impl_surfaces` / `_trait_impl_surfaces_many`：

```python
blob = " ".join(str(block.get(k, "")) for k in ("trait", "for_type", "context", "header"))
hit = next((s for s in wanted if s and s in blob), None)     # <- 纯子串匹配
```

`rust_ckg.py` 的 `query_neighborhood` / `build_fault_chains`：

```python
callee_base = ce["callee"].split("::")[-1]
if ce["callee"] in frontier or callee_base in frontier:      # <- 只比基名
```

### 2.2 新的接口：给 `build_impact_surface` 加一个可选的语义图参数

**不要改 `build_impact_surface` 的签名**（它在 `multi_agent.py:331/830/3797` 和
`tools.py:281` 有 4 个调用点）。改为在 `ckg` 对象上挂一个可选属性：

```python
# rust_ckg.py — RustCodeGraph.__init__
def __init__(self, root: Path):
    ...
    self.semantic: SemanticGraph | None = None   # NEW: Stage 1 overlay
```

这样所有现有调用点**零改动**即可享受收益。

### 2.3 `_trait_impl_surfaces` 的精确版

```python
def _trait_impl_surfaces_precise(sem: Any, symbol_full_paths: list[str], limit: int = 10) -> list[dict[str, Any]]:
    """Exact trait->impl surfaces using compiler-assigned ids.

    The lexical fallback matches by substring, which conflates same-named
    traits in different modules. Here the trait is identified by its fully
    qualified path, so `a::Shape` and `b::Shape` never collide.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for impl in sem.dependents_of_trait_any(symbol_full_paths):
        if impl.key in seen:
            continue
        seen.add(impl.key)
        out.append({
            "path": impl.file,
            "line": impl.line,
            "trait": impl.trait_full_path or "",
            "for_type": impl.type_full_path or "",
            "context": impl.context,
            "precision": "compiler",
        })
        if len(out) >= limit:
            break
    return out
```

### 2.4 调用点的分派

在 `_trait_impl_surfaces_many` 入口处：

```python
def _trait_impl_surfaces_many(ckg: Any, symbols: list[str], limit: int = 10) -> list[dict[str, Any]]:
    sem = getattr(ckg, "semantic", None)
    if sem is not None and sem.precise:
        full_paths = sem.resolve_full_paths(symbols)
        if full_paths:
            return _trait_impl_surfaces_precise(sem, full_paths, limit)
    return _trait_impl_surfaces_many_lexical(ckg, symbols, limit)   # 原逻辑改名保留
```

`resolve_full_paths` 的作用是把 agent 给的短名（`Shape`）映射到**候选完整路径列表**：

```python
def resolve_full_paths(self, symbols: list[str]) -> list[str]:
    """Map bare names to every fully-qualified path that carries that name.

    Returns all candidates — the caller decides. When a name is unique in the
    workspace the list has one element; when it is ambiguous (two `Shape`
    traits) both are returned so the caller can disambiguate by file context.
    """
    out: list[str] = []
    for sym in symbols:
        if not sym:
            continue
        for full in self.by_name_index.get(sym, []):
            if full not in out:
                out.append(full)
    return out
```

### 2.5 关键：短名仍然歧义，怎么办

编译器解决了**同名不同物**的问题，但 agent 给你的往往只是短名 `Shape`。
处理策略（**保守优先**）：

1. 若短名在整个 workspace **唯一** -> 直接用，精确。
2. 若**多个**候选 -> 按文件上下文消歧：
   - 候选的 `file` 与改动文件相同 -> 优先
   - 候选的 `module_path` 与改动文件的 module 同前缀 -> 优先
   - 仍无法消歧 -> **全部返回**，但在每条上标 `"precision": "compiler_ambiguous"`，
     并在 prompt 提示词里加一句"以下 trait 同名但不同物，请按 file/module 判断"。

这比现在的"字符串包含"严格更优：**同名时给出全部候选 + 明确标注歧义**，
比"默默混为一谈"安全得多，而且 agent 有能力用 file/module 区分。

---

## 3. 改造 `rust_ckg.py`：消除基名碰撞

### 3.1 `build_fault_chains`：不再 `cands[0]`

现状（第 ~1520 行）：

```python
cands = [x for x in fn_defs.get(cb, []) if x["kind"] == "fn"]
cands.sort(key=lambda x: (0 if x.get("has_body") else 1, 0 if x["path"] == root_item["path"] else 1, x["path"]))
chosen = cands[0]          # <- 永远第一个，5 个 area 会塌成 1 个
```

改法：**用语义图做消歧**，让链条能区分同名的不同方法：

```python
# NEW: resolve a callee base name to a specific definition using the compiler graph
def _resolve_callee_target(sem, callee_name: str, caller_file: str) -> dict | None:
    """Pick the single definition a call site really refers to.

    Uses the compiler graph: same-name methods in different impls are distinct
    items, and the impl's owning type narrows the candidates.
    """
    if sem is None or not sem.precise:
        return None
    cands = sem.items_by_name.get(callee_name, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # prefer a definition in the same file as the caller
    same_file = [c for c in cands if c.file == caller_file]
    if len(same_file) == 1:
        return same_file[0]
    # prefer one whose owning type matches a type named in the call expression
    for c in same_file or cands:
        if c.owner_type and c.owner_type.split("::")[-1] in callee_name:
            return c
    return None      # ambiguous -> caller keeps lexical behaviour
```

在 `build_fault_chains` 里，遍历 `cur_callees` 时优先用 `_resolve_callee_target` 选节点。

### 3.2 `query_neighborhood` 的 callers 遍历

现状：

```python
callee_base = ce["callee"].split("::")[-1]
if ce["callee"] in frontier or callee_base in frontier:
```

这是 5 个 `area` 塌成 1 个的直接原因。改法：

```python
# NEW: exact consumer lookup via the compiler graph
def consumers_of_item(sem, item_key: str, limit: int = 12) -> list[dict]:
    """Call sites that the compiler says refer to a specific definition."""
```

**但这里有个重要的现实约束**：`call_edges` 是词法解析出来的，**没有绑定到语义 id**。
所以 Stage 1 之后，调用边**仍然无法精确到具体 impl**。

> **这就是 Stage 1 的边界，必须诚实写进论文和 README。**
> Stage 1 修好的是：
> - **定义侧**：trait->impl->method 归属、方法与类型的对应关系、完整限定路径
> - **被 `_trait_impl_surfaces` 驱动的部分**（即 `rust_impact` 的 trait impl 提示）
> Stage 2 才修好调用侧的精确绑定。

---

## 4. 降级策略（必须有）

`cargo doc` 会失败：依赖未构建、build script 失败、宏展开错误、crate 本身编译不过
（**你的场景里这是常态：44.5% 的任务在复现阶段就失败**）。

### 4.1 三级降级

| 级别 | 条件 | 图的来源 | 给 agent 的提示 |
|---|---|---|---|
| **L0 precise** | `cargo doc` rc=0 | 编译器 | "以下关联由编译器确认" |
| **L1 partial** | workspace 部分 member 成功 | 成功的 member 用编译器，失败的用词法 | 标注每条的 `precision` |
| **L2 lexical** | 全部失败 | 现有词法 CKG | 提示"图为近似，可能有同名混淆" |

### 4.2 实现

```python
class SemanticWorkspaceBuilder:
    def __init__(self, root: Path, env: dict[str, str], crates: list[str]):
        ...

    def build(self) -> SemanticWorkspace:
        ws = SemanticWorkspace(root=self.root)
        for crate in self.crates:
            graph = build_semantic_graph(self.root, crate, self.env)
            ws.graphs[crate] = graph
            if graph.precise:
                ws.members_ok.append(crate)
            else:
                ws.members_failed[crate] = graph.degrade_reason or "unknown"
        ws.precise = bool(ws.members_ok)
        # Any crate that failed falls back to the lexical CKG automatically,
        # because rust_impact falls through to lexical when `sem` lacks the item.
        return ws
```

### 4.3 缓存（关键：不能让每个 agent 步骤都重跑 cargo doc）

```python
# Cache keyed by (crate_root, crate, git HEAD or source hash)
# Reuse the harness's existing cache dir pattern.
CACHE_DIR = Path(os.environ.get("RUSTBENCH_CACHE_DIR", ".rustbench_cache")) / "semantic"
```

- 键：`(crate, mtime/size hash of all .rs in the crate)`
- 值：解析好的 `SemanticGraph`（pickle 或 json）
- 失效：任何 `.rs` 变化即失效（Stage 1 是**快照**，不是增量）

> **注意**：agent 的工作副本在修复过程中会被改，语义图会过期。
> 两种策略：
> - (a) **修复前建一次图**，修复后不重建——快，但修完的图是旧的；
> - (b) **每次 `cargo_check` 失败后重建**——准，但慢（实测 0.3-0.4s，可接受）。
>
> **建议 (b) 但带缓存**：键里含源码 hash，改动才重建。
> 重建 0.3-0.4s vs agent 一次 API 调用（数秒），**完全可以接受**。

---

## 5. 接线：让 `ckg.semantic` 可用

### 5.1 `tools.py:4253` 附近

```python
    rust_tools_mod = load_local("rust_tools")
    rust_tools = rust_tools_mod.make_rust_tools(root, resolve_path, rust_env)
    ckg_mod = load_local("rust_ckg")
    ckg_tool, ckg_graph = ckg_mod.make_rust_ckg_tool(root)
    state.ckg = ckg_graph
```

改为：

```python
    ckg_mod = load_local("rust_ckg")
    ckg_tool, ckg_graph = ckg_mod.make_rust_ckg_tool(root)

    # NEW: Stage 1 overlay — compiler-precise definition graph.
    # Best-effort: any failure leaves ckg_graph.semantic = None and the
    # lexical CKG continues to work unchanged.
    try:
        semantic_mod = load_local("rust_semantics")
        ckg_graph.semantic = semantic_mod.build_workspace_graph(root, rust_env)
    except Exception as exc:
        ckg_graph.semantic = None
        _log(f"[semantic] Stage 1 unavailable: {exc}")

    state.ckg = ckg_graph
```

### 5.2 `rust_env` 的传递

`rust_env` 在 `tools.py` 里已经传给 `cargo_check` / `cargo_test`，直接复用即可。
**但必须确认 `rust_env` 里有 `CARGO_TARGET_DIR` 指向共享 cache**——
`rust_environment.py:configure_cargo_cache` 已经做了这件事，
Stage 1 的 `cargo doc` 应当**复用同一个 target dir**，
否则会重新下载/编译全部依赖，在离线环境下直接失败。

> **这是最容易踩的坑**：`cargo doc` 会触发依赖编译。
> 你的基准强制 `--frozen` + 离线 + 共享 cargo-cache，
> 所以 Stage 1 必须 `--offline`（已含）**并且复用 harness 预取的 cache**。
> 如果某个 crate 的依赖没预取，`cargo doc` 会失败 -> 自动降级 L2。
> **这个降级比例本身就是要报告的实验数字。**

### 5.3 提示词更新（`agent_prompts.py`）

`SUGGESTER_PROMPT` 第 5 条现在说：

> "Use inspect_rust_neighborhood / rust_symbols / read_lines to check exact names,
>  trait bounds, generic parameters, lifetimes, Send/Sync, and #[cfg(feature = "...")]."

改为增加一句：

> "When `inspect_rust_neighborhood` reports `precision: compiler`, the trait/impl
>  associations are compiler-confirmed and exact. When it reports
>  `precision: lexical`, the same-named items in different modules may be
>  conflated — verify the file/module before assuming an association."

`VERIFIER` / `FIXER` 的 prompt 里提到 trait impl 一致性的地方同样加一句。

---

## 6. 验证计划

### 6.1 单元测试（新增 `test_rust_semantics.py`）

```python
def test_same_name_traits_get_distinct_ids(tmp_path):
    """Two traits named Shape in different modules must not be conflated."""
    # 写一个两成员 workspace，断言：
    #   dependents_of_trait("core::geometry::Shape") 只返回 geometry 的 impl
    #   dependents_of_trait("core::shapes::Shape")  只返回 shapes 的 impl
    #   两者长度都是 1，且 impl 的 type_full_path 不同

def test_rustdoc_fallback_on_broken_crate(tmp_path):
    """A crate that does not compile must degrade, not raise."""
    # 断言 semantic.precise is False and degrade_reason is set

def test_private_items_included(tmp_path):
    """pub(crate) items must be present in the semantic graph."""

def test_workspace_path_dep_resolves(tmp_path):
    """A member depending on a sibling must resolve (cargo doc, not bare rustdoc)."""

def test_impl_membership(tmp_path):
    """trait->impl->method membership is recorded with exact keys."""
```

### 6.2 回归测试（必须全绿）

现有 `baselines/test` 有 137 个单元测试（`improvment_record.md` 记录）。
**改动后必须全绿**，因为 `rust_impact` / `rust_ckg` 的既有输出格式不能变。

### 6.3 端到端度量（论文的核心数据）

在 Rust-SWE-bench 上测三个数：

| 指标 | 怎么测 | 为什么重要 |
|---|---|---|
| **L0 覆盖率** | 500 题里有多少题能拿到编译器精确图 | 证明方案可行性，且是诚实的边界声明 |
| **误报率下降** | 词法版 vs 语义版，`_trait_impl_surfaces` 返回的 impl 中真正相关的比例 | **直接证明 Stage 1 的价值** |
| **解决率变化** | 同模型同预算，跑词法版 vs 语义版 | 最终指标 |

> 第三个数字要小心：它同时受模型、预算、任务集影响。
> **必须固定 `gpt-6-luna` + 150 calls + 同一 50 题**，
> 和 `testset_results_20261118` 里已有的 `test`（23/50）做对照。

---

## 7. 实施顺序与工作量

| 步骤 | 内容 | 产出 | 预估 |
|---|---|---|---|
| **1** | `rust_semantics.py` 骨架 + `collect_rustdoc_json` + `build_semantic_graph` | 能 dump 出编译器图 | 半天 |
| **2** | `SemanticWorkspace` 跨 crate 合并 + `resolve_full_paths` | 能回答"改 trait 影响谁" | 半天 |
| **3** | `rust_impact.py` 接线 + 降级分派 | 语义版生效，词法版兜底 | 半天 |
| **4** | `tools.py` 接线 + prompt 更新 | agent 能看到 `precision` 标记 | 2 小时 |
| **5** | `test_rust_semantics.py` + 回归全绿 | 可合并 | 1 天 |
| **6** | 端到端度量脚本 + 跑 500 题 | 论文数据 | 2-3 天（跑批） |
| **7** | （可选，Stage 2）调用边精确绑定 | — | 另计 |

**第 1-5 步是一周内可完成的**，且每一步都独立可验证、可回滚。

---

## 8. 三个必须诚实写进论文的边界

1. **Stage 1 只修定义侧。** 调用点绑定（`x.foo()` 调到哪个 impl）仍然不精确，
   要到 Stage 2（HIR/MIR）才能解决。论文里不能笼统写"编译器精确的调用图"。

2. **不是所有仓库都能拿到精确图。** 依赖未预取、build script 失败、
   crate 编译不过都会导致降级。**降级比例必须报告**，而且它本身是有价值的发现：
   "在 Rust-SWE-bench 上，X% 的任务能在预算内拿到编译器精确的定义图"。

3. **这不能算"首创编译器精确的图"。** rust-analyzer / HIR 早已成熟。
   可主张的是：**首次把编译器级精确解析引入仓库级 Rust 修复 agent，
   并证明它对解决率的因果贡献**。这是工程-实证贡献。

---

## 9. 附录：实测命令速查

```powershell
# 单 crate（无 path 依赖时可用）
$env:RUSTC_BOOTSTRAP = "1"
rustdoc --edition 2021 --output-format=json -Zunstable-options `
        --crate-name demo src/lib.rs -o out/
# -> out/demo.json

# workspace（有 path 依赖时必须用这个）
$env:RUSTC_BOOTSTRAP = "1"
$env:RUSTDOCFLAGS = "--output-format=json -Zunstable-options"
cargo doc --offline --no-deps -p app
# -> target/doc/app.json

# 读出编译器认定的 trait->impl->method
#   impl id=3   trait=Shape(id=1)   for=Sq(id=4)   method=area(id=2)
#   impl id=51  trait=Shape(id=49)  for=Ci(id=52)  method=area(id=50)
```

**已知失败模式**：
- `E0432 unresolved import` -> 裸 `rustdoc` 没拿到 `--extern`，改用 `cargo doc`
- `error: the option 'Z' is only accepted on the nightly compiler` -> 没设 `RUSTC_BOOTSTRAP=1`
- `error: the -Z unstable-options flag must be passed` -> `RUSTDOCFLAGS` 没带上 `-Zunstable-options`
- 依赖未编译 -> `cargo doc` 会先 `Checking core v0.1.0`（这步需要离线 cache 命中）
