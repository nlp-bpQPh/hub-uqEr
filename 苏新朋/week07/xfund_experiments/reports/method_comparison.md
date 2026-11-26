# Method comparison (official XFUND zh.val)

All scores classify the same OCR blocks. Entity micro-F1 includes header/question/answer; other is excluded from this metric. Question-answer links are not scored.

| Method | Blocks | Accuracy | 4-class macro-F1 | Entity micro P | Entity micro R | Entity micro F1 | Header F1 | Question F1 | Answer F1 | Other F1 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01_rules | 3629 | 0.5304 | 0.4764 | 0.6662 | 0.5744 | 0.6169 | 0.4545 | 0.5909 | 0.6378 | 0.2225 | completed |
| 02_bilstm_crf | 3629 | 0.6914 | 0.5784 | 0.7449 | 0.7522 | 0.7485 | 0.4359 | 0.6975 | 0.7947 | 0.3853 | completed |
| 03_bert_linear | 3629 | 0.7652 | 0.6493 | 0.7902 | 0.8554 | 0.8215 | 0.5735 | 0.7769 | 0.8688 | 0.3779 | completed |
| 04_bert_crf | 3629 | 0.7561 | 0.6669 | 0.7859 | 0.8432 | 0.8136 | 0.6733 | 0.7676 | 0.8519 | 0.3747 | completed |
| 05_bert_crf_hard_decode | 3629 | 0.7672 | 0.6711 | 0.7999 | 0.8524 | 0.8253 | 0.6522 | 0.7766 | 0.8646 | 0.3909 | completed |
| 06_llm_api_zero_shot | - | - | - | - | - | - | - | - | - | - | not run |
| 07_llm_api_few_shot | - | - | - | - | - | - | - | - | - | - | not run |
