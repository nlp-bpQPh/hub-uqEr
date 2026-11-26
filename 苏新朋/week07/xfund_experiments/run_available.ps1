$ErrorActionPreference = 'Stop'
python prepare_data.py
python methods/01_rules/run.py
python methods/02_bilstm_crf/run.py
python methods/03_bert_linear/run.py
python methods/04_bert_crf/run.py
python methods/05_bert_crf_hard_decode/run.py
python common/compare_methods.py
