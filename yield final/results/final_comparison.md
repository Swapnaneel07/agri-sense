# Final comparison (test set, mean ± std over seeds)

| Model | Test RMSE | Test MAE | Test R² | Train time (s) | Params / trees |
|---|---|---|---|---|---|
| Baseline MLP (no reg) | 13,378.1 ± 205.9 | 7,780.5 ± 227.3 | 0.971 ± 0.001 | 10.0 | 25,089 |
| Best MLP (init + reg) | 13,711.4 ± 369.9 | 7,972.0 ± 277.7 | 0.970 ± 0.002 | 9.7 | 25,089 |
| XGBoost | 18,509.3 ± 270.0 | 11,610.7 ± 128.0 | 0.945 ± 0.002 | 1.7 | 500 trees |

## Conclusion
- Best MLP and Baseline MLP are statistically indistinguishable (RMSE gap 333.3 <= seed std 369.9).
- Best MLP is better than XGBoost by 4,797.9 RMSE (larger than the seed std 369.9).
- Tabular data with mixed categorical + numeric features often favors gradient-boosted trees; the point of this comparison is to measure and explain the gap, and the MLP is still shipped for the course.
