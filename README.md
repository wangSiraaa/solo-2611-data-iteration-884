# 可修订加权记录源与可重复加权采样

本仓库实现三个可独立运行的模块：

- `python -m wrsource.source`：SQLite 支撑的可修订加权记录源，稳定 `row_id`、
  分组、权重 1..100、正文、递增修订号、不可变源版本快照与固定版本分页。
- `python -m wrsource.dataset`：可重复的加权采样数据集生产器，全量构建、按变更
  清单增量刷新、真实 JSONL + manifest 输出与对照源 API 的校验。
- `python -m wrsource.tunestore`（纯领域逻辑在 `wrsource/tuning.py`）：两个编辑点
  从同一已发布源版本**离线**改动后带回的调权单（ticket）双分支合并、可独立预览的
  合后源、以及受“样本发布承诺”约束的配对签发。

另一个模块 `wrsource/sampling.py` 只负责**精确**抽样键算术与选择；`wrsource/reference.py`
是测试用的小集合独立参考实现。没有模型训练，也没有人工标注界面。

## 抽样规则（题目定义，照此实现）

对每个活跃行、给定 256 位 seed：

```
h = SHA-256(seed[32] || uint64_be(len(utf8(row_id))) || utf8(row_id))
a = int(h) + 1                         # u = a / D,  D = 2^256 + 1
priority = u ** (1 / weight)
```

按 priority 降序选 `quota` 行，完全相等时按 `row_id` 码点序。`quota <= 0` 给空
样本，活跃行少于 quota 时全选。不同 seed 是不同请求。抽签键只绑定
`(seed, row_id)`，因此正文修改、改组、移动分片都不会重抽该行的键；权重修改会
改变该行优先级。

### 精确比较（不能先转 float）

两个优先级的比较等价于整数比较：

```
p1 > p2  <=>  (a1/D)^(1/w1) > (a2/D)^(1/w2)
         <=>  a1^w2 * D^w1  >  a2^w1 * D^w2        (两边同取 w1*w2 次幂)
```

实现采用三级比较器（`sampling.compare_priority`）：

1. **float 对数筛选**：仅当 `|log p1 - log p2| > 1e-9` 时直接判定。阈值远大于
   `math.log` 在 `[1, 2^256]` 上的实际误差，只会给出必然正确的结论。
2. **60 位 Decimal 对数**：误差约 `1e-55`，仅当差值 `> 1e-50` 时判定。
3. **精确整数幂交叉比较**：`a1^w2 * D^w1` 与 `a2^w1 * D^w2` 直接比大整数，作为
   最终裁决。返回 0 只表示**实数意义下精确相等**，此时再按 `row_id` 码点序。

因此极近键不会被 float 压成相等。测试构造了 `a` 仅差 1、对数在 float 上完全
相等的情形，精确路径仍能区分；并用 300 位 Decimal 对随机与近邻键做交叉校验。
manifest 为每行保存 `hash_hex`、`a`、`weight`，并为每个超额组保存 `cut` 边界
证书（`a_sel^w_cut * D^w_sel` 与 `a_cut^w_sel * D^w_cut` 两个大整数），可离线
复核比较，无需相信浮点快路。

## 源服务设计

### 表结构

- `rows`：每行的**当前**可变状态，含稳定 `row_id`、group、weight、正文摘要、
  `rev`、`active`、`shard`。`row_id` 一旦分配终身不变。
- `bodies`：内容寻址、只追加的正文库。旧版本正文永不回收（本实现不清理），因此
  每个已发布源版本都保留**真实可读取**的正文。
- `versions` / `version_rows`：不可变整版快照。快照只存 group、weight、正文摘要、
  `rev`，**不存 shard**。
- `batch_log`：原始批次与结果，用于审计。

### 事务与修订

一个批次的每个操作先在内存中折叠为**每行最终状态**，然后在一个
`BEGIN IMMEDIATE` 事务里写当前状态、必要时发布新快照；任何错误整体回滚。因此
“批内按最终状态原子提交”。

- 逻辑内容（active/group/weight/body 摘要）改变 → `rev += 1`，并发布新源版本。
  同一批次对同一行多次修改只算一次修订。
- 只改 `shard`（物理位置）→ `rev` 不变，**不发布新版本**，旧的固定版本内容不变。
  这正是“分片只是物理存放位置，同一逻辑版本内改变分片不改变行身份或内容”。
- 批内先 `add` 后 `retract` 的新行视为从未存在（返回 `discarded_row_ids`）；
  先撤回后新增是合法的“最终状态为活跃”。

### 固定版本分页

分页键是 `(version_id, row_id)`，按 `row_id` 的 UTF-8 字节序（与码点序一致）。
因为版本是不可变快照，读者持有旧版本游标时，写者发布新版本不会让分页漏行或重复。
游标编码了版本号，用错版本会被拒绝。读者在分页过程中穿插写入当前状态，读取的仍
是同一快照。跨分片的行按 `row_id` 统一排序，分片不参与行身份或内容。

### 为何选 SQLite

单文件、WAL 并发读者/单写者、事务与 `PRIMARY KEY` 唯一性天然满足“稳定 row_id +
原子批次 + 不可变版本”。10 万行量级写入约 2s，整版快照读取与分页都在亚秒到数秒
级，远低于题目规模。

## 数据集生产器设计

### 合并：每分片产出全部候选，父进程按组统一选择

- 从**一次一致元数据读取** `all_rows_meta(version)` 取得所有活跃行及其当前 shard，
  在父进程按 shard 分组后分发。即使并发移动分片，也不会漏行或重复。
- 每个 worker 只计算该分片所有行的候选（`a`、`log_screen`）并**全部返回**。
- 父进程把所有分片候选按 `group` 合并成完整候选池，再对整组执行精确 top-quota。
- **绝不拼接各分片样本，也不只保留一个全局候选数**。因此无论分片数与遍历顺序，
  合并结果与单流参考完全一致（测试用随机改分片、并行/串行对照验证）。
  `--parallel` 使用进程池；缺省单进程。二者输出逐字节相同。

### 扫描 vs 候选持久索引

本实现对**需要的组**做一次全量扫描并精确排序，不在数据库里持久化全局候选顺序
索引。理由：

- 抽签键依赖 seed 与权重；权重可变，任何持久索引都要按 `(seed, weight)` 失效，
  维护成本高且易与源版本不一致。
- 当前规模（10 万行、每组 quota ≤ 1000）下，整库精确排序在亚秒到数秒级，重扫
  受影响组足够快。
- 重扫可证明与单流参考一致，避免持久索引的陈旧风险。

若规模继续增长，可缓存 `(row_id, seed) -> a`（与正文/分片无关，仅在 seed 变化时
失效），并用每组堆/树维护 top-quota；权重变化仍要求对应组重排序。当前实现刻意
选择简单且可验证的方案。

### 精确比较缓存

`Candidate` 缓存 `a`（精确键）与 `log_screen`（对数快路值），排序时大量比较走
O(1) 的 float 快路，只在极近时回退 Decimal 与大整数幂。`cut` 证书只在 manifest
生成时对每组边界计算一次，把昂贵的大整数运算限制在真正的边界与极近键上。

### 增量刷新与跨组修订

`refresh` 计算 `diff_versions(old, new)`，得到 added/removed/changed 行及其
**旧组与新组**。受影响的组（新增、撤回、改组的新旧两组、改权、改正文，以及 quota
变化的组）全部在**新版本上完整重扫**；未受影响组复用旧选择。因为未受影响组的行、
权重、分组、正文、quota 都没变，复用与重算在数学上完全相等。

关键点：撤回样本内一行时，是在**新版本的完整组候选**上重新取 top-quota，补入的
是真正的下一名，而不是从旧样本里重抽。改组同时进入旧组和新组的受影响集合，名额
在两组间正确迁移。测试覆盖：撤回首名后与同版本全量重建逐字节相同；改组后两组分别
与独立参考一致；增量刷新与同版本全量构建的 manifest 和 JSONL 完全相同。

### 输出内容

`samples.jsonl` 每行含：`group`、`rank`、`row_id`、`weight`、`hash_hex`、`a`、
`log_screen`、`body_sha256`、`body_bytes`、`source_version`、`seed_hex`、**真实
`body`**。`manifest.json` 含 `format`、`source_version`、`seed_hex`、`quotas`、
`totals`、`records_file`、`records_sha256`、以及每组 `eligible`、`quota`、
`selected_count`、`selected`（带精确比较材料）与 `cut` 边界证书。

`verify` 会重算 `records_sha256`、逐行回查源 API 的正文与摘要、按 seed 重算
`hash_hex`，并做一次同版本全量重建比对身份、顺序与真实正文。因此**每行都从源
API 回查验证**，而不是只写抽样元数据。

## 离线调权单双分支合并与承诺签发

### 票据：编辑点只携带改动

离线编辑点从某个**已发布源版本**检出，带回调权单（ticket）时只带字段级差异，
不带整库。每条行记录显式引用共同基线版本、原 `row_id` 与原行修订号 `base_rev`，
并内联它编辑时看到的原行快照（group/weight/body 摘要/active）。登记时逐条核对：

- 行必须仍在该基线快照中；
- `base_rev` 必须等于快照当前修订号，基线被推进导致的陈旧票据直接拒收；
- 内联原行必须与快照逐字段一致。

每个分支对同一行的同一字段至多发一项。权重可 `set`（设为 n）或 `add`（在基线
基础上增加 d，允许负数）；正文、分组只能 `set` 新值；撤回是 `active=retract`，
且不能与同行其它编辑并列。正文改动进入共享的内容寻址 `bodies` 库，但只追加，
既有版本引用的摘要永不改变。

### 合并规则（纯函数，全部冲突可列出）

`wrsource/tuning.py::merge_tickets` 是无 I/O 的纯函数，语义不合规不抛异常、而是
收集成带**两侧凭证**（票据 id、分支 id、原行修订、原行快照、具体操作）的冲突清单：

- 两边改**不相交字段** → 合；
- 并行权重增量 → 增量相加；两个相同的绝对设值 → 合（不同设值冲突）；
- **设值遇到增量即冲突**，哪怕两边各自单独做到的末态数字完全相同（凭证里同时
  记录 `branch_end_weight` 与 `baseline_weight`，证明这是规则冲突而非数值冲突）；
- 撤回遇到未改行 → 合；两边并行撤回同一行 → 合；撤回撞到另一侧实改 → 列冲突；
- 分组/正文两边设成不同值 → 冲突；相同值（正文按摘要）→ 合；
- 合成权重必须落在旧范围 `1..100`，越界列 `weight_out_of_range`；
- 只有 `row_id` 相同才比较，不同行同字段不冲突。

合并按票据 id 规范化左右，因此 `merge(a,b)` 与 `merge(b,a)` 得到同一 `merge_id`。
`merge_id` 是“票据对 + 合后行 + 冲突清单”的内容哈希；同一对票据重复合并幂等。
冲突合并可以持久化与审阅，但没有有效视图（`effective_view` 直接拒绝），更不能
预演或签发。

### 预演：先抽签，承诺只核对、永不补位

`run_rehearsal` 在**合后源的完整活跃组**上跑与主生产器完全相同的精确抽签
（复用 `sampling.compare_priority` 三级比较器与 `dataset.build_group_result`，
含 `cut` 边界证书）。承诺（promise）在分组里点名哪些 `row_id` 必须保留
（keep）、哪些不能出现（forbid），它只在确定性抽签结果出来后做核对：

- keep 的行不在组内 → `keep_missing`；在组内但未中签 → `keep_not_selected`；
- forbid 的行中签 → `forbid_present`；组不存在 → `unknown_group`。

承诺不能新增、替换或“补换”任何入选者；名额 1 而 keep 两行时，抽签仍只出一行。
`rehearsal_id` 哈希绑定 `(merge_id, seed, quotas, promises)`，并显式记录所用的
`merge_content_sha256`——同一合并被多个承诺分别试算时，每次预演都能解释自己复用
的是哪一版合并。预演无论承诺是否满足都持久化，只保留可复核的抽签材料与差异。

### 签发状态机与配对原子可见

签发号（issuance_no）是调用方给的人工单号，状态机为 `pending → approved|rejected`，
决策一次性、不可改判：

1. `request`：仅接受承诺满足的预演；在一个事务里落 `pending` 记录并把样本行以
   `source_ref_kind='preview'` 暂存（此时**没有**合后源版本号）。同一签发号重试
   且内容（合并摘要 + 预演 id）完全相同 → 幂等返回；内容不同 → 拒绝，即使主库
   基线已经推进，也绝不允许把两份不同合并结果挂到同一单号。请求与决策之间崩溃
   是安全的：`pending` 与预览样本已落盘，重启后重试即可。
2. `decide --approve`：同一事务内分配合后源版本（引用名 `merged-N`，与主库整数
   版本号命名空间不相交）并把样本引用翻转为 `merged-N`，因此**合后源版与关联样本
   成对可见**，不存在只看见一半的中间态。
3. `decide --reject` 或承诺不满足：不分配合后源版本，只保留可复核的预演、差异与
   预览样本。

### 与旧产品独立查询

调权表全部带 `tune_` 前缀，与主库 `rows/versions/version_rows` 同库不同表；本模块
对主表只读。主库旧版本、离线预演（`merged-preview:<merge_id>`）、已发布合后版本
（`merged-N`）三套视图互不影响：`MergedView` 实现了数据集生产器所需的最小接口，
未改动的 `wrsource.dataset.build_dataset` 可以直接在合并预览上产出真实正文的
JSONL/manifest，manifest 的 `source_version` 即合并预览标识。

## 代码结构

```
wrsource/
  sampling.py   精确键、比较器、top-quota、边界证书（无 I/O）
  source.py     版本化源服务 + 分页 + CLI（可独立运行）
  dataset.py    全量构建 / 增量刷新 / 校验 + CLI（可独立运行）
  tuning.py     票据校验、双分支合并、冲突凭证、预演与承诺判定（纯函数）
  tunestore.py  票据/合并/预演/签发的 SQLite 持久化 + MergedView + CLI
  reference.py  测试用独立精确参考实现
tests/
  test_sampling.py  键构造、双向一致性、极近键、边界证书、参考对照
  test_source.py    原子批次、折叠、修订号、分片移动、分页并发、重启
  test_dataset.py   全量/参考对照、零/超名额、正文版本、分片不变性、
                    并行一致、撤回首名补位、改组、增量==全量、篡改检测、重开
  test_tuning.py    票据校验、全部合并规则、任务 A/B 示例、承诺与预演身份
  test_tunestore.py 持久化、发布中断/重试幂等、同号防重绑、基线推进、
                    配对原子发布、旧产品与预演独立查询、合并视图上的生产器
demo/run_demo.py          端到端演示（含 --large 10 万行）
demo/run_tuning_demo.py   离线双分支合并 + 承诺签发端到端演示（含中断重试）
```

## 运行

```bash
# 测试
python3 -m unittest discover -s tests -v

# 演示（小规模 / 10 万行）
python3 demo/run_demo.py
python3 demo/run_demo.py --large --parallel

# 源服务 CLI
python3 -m wrsource.source init --db src.sqlite
python3 -m wrsource.source batch --db src.sqlite --ops ops.json
python3 -m wrsource.source versions --db src.sqlite
python3 -m wrsource.source page --db src.sqlite --version 1 --limit 100
python3 -m wrsource.source page --db src.sqlite --version 1 --cursor <CURSOR>
python3 -m wrsource.source diff --db src.sqlite --old 1 --new 2
python3 -m wrsource.source verify-bodies --db src.sqlite --version 1
python3 -m wrsource.source dump --db src.sqlite --version 1 --out dump.jsonl

# 数据集 CLI
python3 -m wrsource.dataset build --db src.sqlite --version 1 \
    --seed-hex <64 hex> --quota g0=1000 --quota g1=1000 --out ds_v1
python3 -m wrsource.dataset refresh --db src.sqlite --old ds_v1 \
    --new-version 2 --out ds_v2
python3 -m wrsource.dataset verify --db src.sqlite --dataset ds_v2
python3 -m wrsource.dataset compare --a ds_v2 --b ds_v2_full

# 离线调权单：票据登记 → 双分支合并 → 预演 → 承诺签发
python3 demo/run_tuning_demo.py
python3 -m wrsource.tunestore ticket   --db src.sqlite --ticket-file left.json
python3 -m wrsource.tunestore merge    --db src.sqlite --left <idL> --right <idR>
python3 -m wrsource.tunestore diff     --db src.sqlite --merge <merge_id>
python3 -m wrsource.tunestore baseline-status --db src.sqlite --merge <merge_id>
python3 -m wrsource.tunestore rehearse --db src.sqlite --merge <merge_id> \
        --seed-hex 00..00 --quota gAB=1 --promise-file promises.json
python3 -m wrsource.tunestore request  --db src.sqlite --no ISS-001 \
        --rehearsal <rehearsal_id>
python3 -m wrsource.tunestore decide   --db src.sqlite --no ISS-001 --approve
python3 -m wrsource.tunestore issuances --db src.sqlite
python3 -m wrsource.tunestore sample   --db src.sqlite --no ISS-001
python3 -m wrsource.tunestore merged-versions --db src.sqlite
```

规模验证：10 万行、100 组、每组 quota 1000（合计 10 万样本）时，写入约 2s，
并行全量构建约 2.5s，增量刷新受影响组约 3s，校验约 3s。
