# SOC Estimation Full Report
**Generated on**: 2026-03-28 06:23:42

## 2. Dataset Partition (Zero-Shot Cross-Profile)
| Dataset | Driving Cycles | Academic Purpose Statement |
|---------|----------------|---------------------------|
| Training Set | DST (All Temperatures) | **Training Only**: The only source of gradient updates visible to the model |
| Validation Set | FUDS (All Temperatures) | **Validation**: Used solely for PSO fitness evaluation and early stopping (Seen distribution) |
| Test Set | US06 (All Temperatures) | **Strict Blind Test**: The sole metric for zero-shot cross-profile generalization performance |

## 3. PSO vs Grid Search Hyperparameter Optimization Comparison
*Note: PSO employs log-space search for precise localization of continuous minima.*
| Method | Learning Rate (LR) | CNN Kernel | LSTM Hidden Units | Validation (FUDS) RMSE (%) |
|--------|---------------------|------------|-------------------|----------------------------|
| PSO    | 0.000588 | 7 | 113 | 1.3909 |
| Grid   | 0.000100 | 11 | 64 | 1.6818 |

## 4. Detailed Performance Results Across All Temperatures
*Academic Statement: Since FUDS participates in the early stopping mechanism, its results are for reference and validation only. The true cross-profile generalization capability of the model is based on US06.*

| Temperature (°C) | Dataset Attribute | Cycle | RMSE (%) | MAE (%) |
|------------------|-------------------|-------|----------|---------|
| 0 | Validation (Seen) | FUDS | 1.3178 | 1.0249 |
| 0 | **Blind Test (Unseen)** | **US06** | **1.6299** | **1.2708** |
| 10 | Validation (Seen) | FUDS | 1.1117 | 0.8676 |
| 10 | **Blind Test (Unseen)** | **US06** | **1.3221** | **1.0631** |
| 20 | Validation (Seen) | FUDS | 0.9059 | 0.7112 |
| 20 | **Blind Test (Unseen)** | **US06** | **1.3767** | **1.1101** |
| 25 | Validation (Seen) | FUDS | 0.9211 | 0.7267 |
| 25 | **Blind Test (Unseen)** | **US06** | **1.2342** | **0.9677** |
| 30 | Validation (Seen) | FUDS | 0.8070 | 0.6353 |
| 30 | **Blind Test (Unseen)** | **US06** | **1.0722** | **0.8731** |
| 40 | Validation (Seen) | FUDS | 0.7954 | 0.6182 |
| 40 | **Blind Test (Unseen)** | **US06** | **1.0003** | **0.8163** |
| 50 | Validation (Seen) | FUDS | 0.8657 | 0.6648 |
| 50 | **Blind Test (Unseen)** | **US06** | **1.1516** | **0.9332** |
## 5. Key Performance Summary (Based on US06 Blind Test)
- **US06 Blind Test Average RMSE**: 1.2553%
- **US06 Blind Test Average MAE**: 1.0049%
*(Note: FUDS validation set average RMSE is 0.9607%)*

## 6. Final Conclusion
The proposed model completely abandons the traditional Coulomb counting accumulation. It solely relies on 5-dimensional transient measurable physical quantities (current, voltage, temperature, and voltage temporal derivatives). Under the rigorous zero-shot framework of "DST Training $
ightarrow$ FUDS Validation $
ightarrow$ US06 Blind Test", it achieves extremely high prediction accuracy across the full temperature range (0-50°C). Furthermore, by utilizing the log-scaled PSO algorithm, it successfully overcomes the limitations of discrete grid search and precisely locates the optimal hyperparameters in the continuous domain. The proposed method features rigorous physical logic and possesses complete practical value for edge deployment in real BMS.
