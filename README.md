# 不确定身份关联下的服务受益覆盖估计

本项目保存业务分析所需的本地（合成）资料，并在不生成“不可争议统一ID”的前提下，
估计同一时期的受益人数、跨部门重叠和固定总体中的未覆盖比例。所有数据都是本地
合成 token，不请求任何外部身份查询。

## 这不是什么

* 不是对姓名做阈值并查集。弱相似链不会闭包成大团；团块内任意两条原子必须有
  直接候选证据，并同时满足出生桶、可信 person 锚点和时间约束。
* 不是把比较分数叫概率。比较分数只作为证据，先在校准样本上估计校准参数；
  缺失字段不提供负证据，跨期换号也不把号码差异当作异人证据。
* 不是只给一个覆盖数字。报告同时给出：一个明确标注假设的代表关联、由关联
  不确定性产生的区间/分布、以及资料不足时的部分识别范围。
* 不是各条边独立抽样。每个抽样世界都是对原子的一个合法划分，互斥身份约束
  在所有世界中都成立。

## 输入

`data/` 与 `scenarios/` 下每个资料包由清单（manifest）和四张 JSONL 表组成：

* `population`：固定总体名单（id、birth_bucket、area）。
* `records`：部门名册（id、department、period、name_token、phone_token、birth_bucket）。
* `bridges`：少量可信桥接（id、record、person、valid_from、valid_to、revoked_at）。
  `valid_from` 包含、`valid_to` 不包含；`revoked_at <= version` 时该版本不可见。
* `comparisons`：已登记属性比较（id、left、right、name_score、phone_score、
  birth_score、reviewed_same）。分数不是概率；`reviewed_same=null` 表示未复核。

JSONL 文件按清单中的 SHA256 与字节长度绑定，任何内容变化都必须重新登记。

## 流程

1. **候选生成**（`candidates.py`）：部门内重复上报、可信桥接先形成原子；候选只
   来自已登记比较，不按姓名批量连接。出生桶冲突、不同可信 person 是禁止边。
   候选之外属性相容但未登记比较的对进入漏配审计，并给出分层补抽样建议。
2. **校准**（`calibration.py`）：分数 → LLR → L2 logistic 校准；空值贡献为 0。
   标注不足时退回弱先验并传播参数不确定性。若标注集中在高分候选，报告明确
   声明选择偏差并给出补抽样方案（层内随机、按纳入率倒数加权）。
3. **受约束关联**（`worlds.py`）：精确枚举小连通分量的全部合法划分；较大分量
   用精确条件分布的 Gibbs。每个世界都满足弱链不闭包、互斥锚点、出生桶与时间约束。
4. **覆盖统计**（`coverage.py`）：每期受益人数、部门重叠人数/部门对、固定总体
   未覆盖比例；未锚定团块按部分识别报上下界。代表关联是权重最高的单个世界，
   明确标注为假设而非身份事实。
5. **复核队列**（`review.py`）：按“复核会改变哪个统计量、改变多少”排序，而不是
   只挑最难匹配的边。候选边、可裁决的桥接冲突、漏配补抽样分别列出。
6. **版本与复算**（`analyze.py`）：快照只保存输入摘要、配置和依据；用快照配置
   重跑会得到相同的校准签名、世界摘要和复核队列摘要。撤销只影响后续版本。

## 运行

```sh
python -m unittest discover -s tests -v
python -m compileall -q linkage_data tests
python -m linkage_data.cli data/normal.json                      # 完整性检查
python -m linkage_data.cli analyze data/normal.json --period 1    # 覆盖分析
python -m linkage_data.cli snapshot data/normal.json --period 1 --version 1 --out out/snapshots
python -m linkage_data.cli verify data/normal.json out/snapshots/<file>.json
python -m linkage_data.cli replay scenarios/bridge-revocation/manifest.json --period 1 --versions 1,2
python -m linkage_data.cli sensitivity scenarios/phone-change/manifest.json --period 1
```

`analyze` 的 JSON 包含 `coverage`（代表关联、后验分布、部分识别范围）、
`calibration`（权重、标注审计、偏差说明）、`candidates`（边际后验、禁止边、
冲突、漏配审计）、`review_queue` 和 `warnings`。

## 场景回放

`scenarios/` 为本地合成资料，覆盖：

| 场景 | 重点 |
| --- | --- |
| `phone-change` | 改号不拆人；跨期号码差异不作为异人证据 |
| `same-name-different-person` | 同名同生日但可信 person 互斥 |
| `duplicate-submission` | 部门内多次上报只计一次 |
| `weak-chain` | a-b、b-c 强相似且 a-c 未登记；不闭包，漏配审计 |
| `anchor-conflict` | a、c 锚定不同 person，b 不能同时并入两侧 |
| `bridge-revocation` | 版本 2 撤销桥接；旧版快照仍可复算 |
| `calibration-bias` | 标注集中于高分候选；声明偏差并给出补抽样 |
| `mixed-ops` | 改号、重复上报、同名、跨期与漏配并存 |

重新生成场景：

```sh
python tools/build_scenarios.py scenarios
```

## 运行环境

Python 3.11+，仅标准库。`docker build -t linkage-data . && docker run --rm linkage-data`
执行默认完整性检查。
