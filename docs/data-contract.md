# 业务资料约定

所有token均合成，不能反查真实身份。valid_from包含，valid_to不包含。revoked_at为后续资料版本获知撤销的时刻。comparisons为已登记属性比较，不等于匹配概率；reviewed_same=null表示没有复核结论。

每行id在该表内唯一，文件顺序不代表业务事件顺序。清单schema=1，各表登记文件名、字节长度、SHA256、行数及完整字段集合，未知版本或内容变化应重新登记。金额及测量数值均为独立资料，不代表任何现实机构的经营数据。

- population：id、birth_bucket、area。
- records：id、department、period、name_token、phone_token、birth_bucket。
- bridges：id、record、person、valid_from、valid_to、revoked_at。
- comparisons：id、left、right、name_score、phone_score、birth_score、reviewed_same。

normal与boundary是相互独立的资料包，不能直接拼接重复id。引用字段按本项目domain.py的REFERENCES登记。读取检查只检查完整性、身份及这些直接引用，不替代领域分析中的可识别性、时点可见性或可行性判断。

## 分析对资料的解释（不确定身份关联）

以下规则属于分析模型，不是资料字段本身；改变它们必须作为新配置重新登记并在
报告中说明。

- **时期**：只有 `records.period` 等于分析时期的记录进入覆盖统计；其他时期的
  记录仍可用于身份关联。桥接有效期按其所关联记录的时期检查：`valid_from <=
  record.period < valid_to` 时视为覆盖。可用 `bridge_period_scope=all` 做敏感性分析。
- **撤销**：`revoked_at <= version` 的桥接在该版本不可见；版本号更小时仍可见。
  同一份资料在版本 1 与版本 2 可以产生不同划分，旧版本快照用其原配置复算。
- **空值**：比较分数为 `null` 时该字段不提供证据（贡献恰为 0），不表示“不相同”。
- **比较分数**：`comparisons` 的分数只作为证据；报告中的同人概率是经独立标注
  校准后的模型量，不是分数本身。
- **重复上报**：同部门、同姓名、同号码、同出生桶的记录先合成一个原子，覆盖统计
  只计一次；合并若会违反可信 person 锚点，则不合并并登记冲突。
- **弱链**：团块内任意两条原子必须有直接候选边且达到内聚阈值；相似链不能闭包。
- **固定总体**：名单覆盖只认可信桥接锚点；未锚定团块按部分识别报告上下界，
  不给名单归属硬分配概率。
