# Phase 4 critical comparison: 15M vs 60M

## Raw leaderboards

### 15M params, 1 epoch, 150M unique tokens (CC val)
rid                    name  val_loss
 V0    random_eq150M_seed42    4.2390
 V6 ebm_within_topic_eq150M    4.2400
 V7    random_eq150M_seed43    4.2466
 V8    random_eq150M_seed44    4.2501
 V4       ebm_interp_eq150M    4.2582
 V9       ebm_eq150M_seed43    4.2711
 V3              ebm_eq150M    4.2720
 V2            fwedu_eq150M    4.3179
 V5            lasso_eq150M    4.3267
 V1             dclm_eq150M    4.3405

### 60M params, 2 epochs, 150M unique tokens (CC val)
rid                    name  val_loss
 V6 ebm_within_topic_eq150M    3.8734
 V0    random_eq150M_seed42    3.8839
 V7    random_eq150M_seed43    3.8865
 V8    random_eq150M_seed44    3.8963
 V3              ebm_eq150M    3.9136
 V9       ebm_eq150M_seed43    3.9183
 V4       ebm_interp_eq150M    3.9206
 V5            lasso_eq150M    3.9596
 V2            fwedu_eq150M    3.9672
 V1             dclm_eq150M    4.0064

Mean random:  15M = 4.2452   60M = 3.8889

## Family means (averaged across seeds where applicable)
| family | mean val_loss 15M | mean val_loss 60M | gap-vs-random 15M | gap-vs-random 60M | change with scale |
|---|---:|---:|---:|---:|---:|
| Random | 4.2452 | 3.8889 | +0.0000 | +0.0000 | +0.0000 ≈ unchanged |
| EBM within-topic | 4.2400 | 3.8734 | -0.0053 | -0.0155 | -0.0102 ✅ narrows |
| EBM (global top-k) | 4.2671 | 3.9175 | +0.0219 | +0.0286 | +0.0067 ❌ widens |
| Lasso | 4.3267 | 3.9596 | +0.0815 | +0.0707 | -0.0107 ✅ narrows |
| FineWeb-Edu | 4.3179 | 3.9672 | +0.0727 | +0.0783 | +0.0056 ❌ widens |
| DCLM | 4.3405 | 4.0064 | +0.0953 | +0.1175 | +0.0222 ❌ widens |
