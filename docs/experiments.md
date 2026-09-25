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

## Submission History & Strategy

### Submission #1
- **Status**: Submitted
- **Model**: HS-SAFD Single Model
- **Notes**: Initial upload.

### Submission #2 (Corrected int64 Schema)
- **Status**: Ready for Upload / Submitted
- **File**: `outputs/submissions/submission.csv`
- **Model**: HS-SAFD (Single Model + CIoU Loss + TTA)
- **Fix Applied**: `image_id` strictly converted to `int64` integer type.
- **Validation**: Passed all 12 Kaggle competition checks.
- **SHA256**: `CC3D091FE6BC7F24A82C9465B4CA99077DBB878313073E8F0068E21D6F018129`

---

## Key Rules for Scoring Selection

1. **Up to 2 Submissions**:
   - Kaggle allows selecting up to **2 submissions** for final leaderboard scoring.
   - For Phase 1, ensure your corrected submission is checked.
2. **Phase 2 Transition (Sept 25, 16:00 Beijing = 13:30 IST)**:
   - When the ranking set (1,000 scenes) is released, run single-model inference on both test (1,000) and ranking (1,000).
   - The Phase 2 submission must contain **both** test and ranking predictions.
   - Manually tick **"Use for final scoring"** on the combined Phase 2 submission on Kaggle.
