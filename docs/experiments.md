# Competition Experiment Tracking Log

| Exp ID | Model Architecture | Input Channels | TTA | Conf Thr | NMS IoU | Local mAP50 | Local mAP@[0.50:0.95] | Kaggle Public Score | Notes / Decisions |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP01** | RGB 3-Band Baseline | 3 bands | No | 0.20 | 0.50 | — | — | — | Pseudo-RGB baseline (Bands 2, 7, 15) |
| **EXP02** | 16-Band CNN Baseline | 16 bands | No | 0.20 | 0.50 | — | — | — | Full spectral baseline without attention |
| **EXP03** | 16-Band + Spectral Attention | 16 bands | No | 0.15 | 0.50 | — | — | — | Adds wavelength weighting module |
| **EXP04** | HS-SAFD (Spectral + Spatial) | 16 bands | No | 0.15 | 0.50 | — | — | — | Full hybrid backbone with PAN neck |
| **EXP05** | HS-SAFD + TTA (Phase 1 Final) | 16 bands | Yes | 0.10 | 0.55 | — | — | — | Single-model multi-scale + flips TTA |
| **EXP06** | HS-SAFD + TTA (Phase 2 Final) | 16 bands | Yes | 0.10 | 0.55 | — | — | — | Phase 2 joint Test (1,000) + Ranking (1,000) |

---

## Submission Notes

### Submission 1 (Phase 1 Baseline)
- **Model**: HS-SAFD Single Model
- **Input**: 16 bands (460–600 nm)
- **TTA**: No
- **Confidence Threshold**: 0.20
- **NMS IoU**: 0.50
- **Description**: Initial Phase 1 baseline submission on test set.

### Submission 2 (Phase 1 Optimized)
- **Model**: HS-SAFD Single Model
- **Input**: 16 bands
- **TTA**: Yes (Scales: 0.8, 1.0, 1.2; Flips: H, V)
- **Confidence Threshold**: 0.10
- **NMS IoU**: 0.55
- **Description**: Optimized single-model with TTA and CIoU loss.
