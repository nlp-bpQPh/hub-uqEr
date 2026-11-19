# BERT 全部实验（每个 1 epoch）

| 实验 | 池化 | Warmup | Loss | FGM | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---|---:|---:|---:|---:|
| focal | cls | 0.00 | focal | False | 0.5576 | 0.5421 | 0.5573 |
| warmup | cls | 0.10 | ce | False | 0.5589 | 0.5375 | 0.5585 |
| cls | cls | 0.00 | ce | False | 0.5571 | 0.5343 | 0.5569 |
| fgm | cls | 0.00 | ce | True | 0.5596 | 0.5334 | 0.5589 |
| max | max | 0.00 | ce | False | 0.5554 | 0.5327 | 0.5554 |
| mean | mean | 0.00 | ce | False | 0.5572 | 0.5302 | 0.5567 |
