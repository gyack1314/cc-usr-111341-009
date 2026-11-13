# 业务资料约定

所有token均合成，不能反查真实身份。valid_from包含，valid_to不包含。revoked_at为后续资料版本获知撤销的时刻。comparisons为已登记属性比较，不等于匹配概率；reviewed_same=null表示没有复核结论。

每行id在该表内唯一，文件顺序不代表业务事件顺序。清单schema=1，各表登记文件名、字节长度、SHA256、行数及完整字段集合，未知版本或内容变化应重新登记。金额及测量数值均为独立资料，不代表任何现实机构的经营数据。

- population：id、birth_bucket、area。
- records：id、department、period、name_token、phone_token、birth_bucket。
- bridges：id、record、person、valid_from、valid_to、revoked_at。
- comparisons：id、left、right、name_score、phone_score、birth_score、reviewed_same。

normal与boundary是相互独立的资料包，不能直接拼接重复id。引用字段按本项目domain.py的REFERENCES登记。读取检查只检查完整性、身份及这些直接引用，不替代领域分析中的可识别性、时点可见性或可行性判断。
