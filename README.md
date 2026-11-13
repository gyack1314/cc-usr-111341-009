# 不确定身份关联下的服务受益覆盖估计

本项目保存业务分析所需的本地资料，提供内容摘要、记录身份与登记关系检查。当前读取入口只返回表数量及身份检查结果，不进行统计拟合、分群、优化或经营判断。

资料表包含population、records、bridges、comparisons。字段单位和关系见 docs/data-contract.md。data/normal.json登记常规资料，data/boundary.json登记特殊业务情形，data/broken-digest.json用于检查资料损坏的拒绝行为。JSONL文件按清单内SHA256和字节长度绑定。

运行环境为 Python 3.11及以上，当前仅使用标准库，无需额外服务。包内io.py负责资料读取，domain.py负责表集合与引用检查，cli.py提供本地检查入口。

```sh
python -m unittest discover -s tests -v
python -m compileall -q linkage_data tests
python -m linkage_data.cli data/normal.json
python -m linkage_data.cli data/boundary.json
```

容器内使用同样的检查入口：

```sh
docker build -t linkage-data .
docker run --rm linkage-data
```
