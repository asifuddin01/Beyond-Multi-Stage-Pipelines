# Released results

All numbers are on the sealed 2,454-image test split (1,642 ODIR-5K, 640 RFMiD 1.0,
172 RFMiD 2.0) over 39 targets (38 named diseases + pooled `G_Rare`).

| Path | Content |
|---|---|
| `final_ensemble/test_pred_ensemble_flat_lc_w055_tta.csv` | binary predictions of the best configuration (flat + label-correlation, TTA x4, w_lc = 0.55) |
| `final_ensemble/test_prob_ensemble_flat_lc_w055_tta.csv` | blended probabilities, one `<code>_prob` column per target |
| `final_ensemble/test_per_disease_ensemble_flat_lc.csv` | per-target support, F1, AP, decision threshold and number of predicted positives |
| `ablation/ablation_table.csv` | the nine configurations of the ablation study |
| `pipeline/pipeline_summary.json` | cascade metrics, loss cascade and per-image outcomes |
| `pipeline/pipeline_per_disease.csv` | cascade per-disease precision / recall / F1 / AUC |
| `pipeline/stage2_loss_ablation.csv` | Stage-2 router: BCE vs focal vs ASL vs GeM (validation) |
| `pipeline/stage3_validation.csv` | Stage-3 group classifiers on validation, incl. the G5 source shortcut |
| `flat_baseline/` | flat model per-disease table and summary |

Binary predictions equal `prob >= thr` with the thresholds in the per-disease file;
`scripts/score_predictions.py` reproduces every metric from these files and the
test labels. Thresholds were tuned on validation for the 22 targets with at least
10 validation positives; the remaining targets use 0.5.
