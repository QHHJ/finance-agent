# Travel E2E Benchmark

这个目录用于差旅材料分类的端到端测评。

## 评测口径

- 文件级 `doc_type`：
  - `transport_ticket`
  - `transport_payment`
  - `flight_detail`
  - `hotel_invoice`
  - `hotel_payment`
  - `hotel_order`
  - `unknown`
- 批次级 `slot`（可选）：
  - `go_ticket`
  - `go_payment`
  - `go_detail`
  - `return_ticket`
  - `return_payment`
  - `return_detail`
  - `hotel_invoice`
  - `hotel_payment`
  - `hotel_order`
  - `unknown`

`doc_type` 是单文件分类结果，`slot` 是整批材料整理后的槽位分配结果。

## 数据集格式

使用 JSONL，一行一个样本。字段如下：

- `sample_id`: 样本唯一 ID，必填
- `file_path`: 文件路径，必填
- `expected_doc_type`: 金标 `doc_type`，必填
- `batch_id`: 同一次出差/同一批材料的分组 ID，可选；不填时默认等于 `sample_id`
- `batch_order`: 批次内顺序，可选；默认按数据集行号
- `expected_slot`: 金标槽位，可选；如果只测文件级分类可以不填
- `note`: 备注，可选

示例见 [dataset.template.jsonl](/E:/code documents/agent caiwu/finance-agent/benchmark/travel/dataset.template.jsonl)。

## 运行方式

```powershell
conda run -n finance-agent python scripts/travel_e2e_benchmark.py benchmark/travel/your_dataset.jsonl
```

自定义输出路径：

```powershell
conda run -n finance-agent python scripts/travel_e2e_benchmark.py benchmark/travel/your_dataset.jsonl --output benchmark/travel/reports/my_report.json
```

## 输出内容

报告会输出到 `benchmark/travel/reports/`，包含：

- 总样本数、总批次数、失败批次数
- 文件级 `doc_type` accuracy / macro F1
- 槽位级 `slot` accuracy / macro F1（如果提供 `expected_slot`）
- 文件级分类耗时 `mean / median / p95 / max`
- 批次级端到端耗时 `mean / median / p95 / max`
- confusion matrix
- 每条样本的预测结果、来源、置信度、耗时、错误信息

## 建议

- 第一轮先做 `30-50` 条试运行集，确保标签口径和文件路径都正确。
- 正式统计建议至少 `100` 条以上，并尽量覆盖机票发票、支付记录、机票明细、酒店发票、酒店支付、酒店订单截图。
- 如果要测整批整理效果，务必正确填写同一出差批次的 `batch_id` 和 `batch_order`。
