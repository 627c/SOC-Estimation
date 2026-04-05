# SOC-Estimation
Zero-shot SOC Estimation with CNN-BiLSTM and Channel Attention (CALCE Dataset)

## 📋 Project Structure
```
SOC-Estimation/
├── utils.py                # Core utilities (dataset, model, helper functions)
├── train.py                # Training script (for author only: PSO + Grid Search + full training)
├── test.py                 # One-click inference (for users: load pretrained model)
├── requirements.txt         # Dependencies
├── best_model_final.pth    # Pretrained model weights
├── feature_scaler.pkl      # Feature scaler
├── soc_scaler.pkl          # SOC scaler
└── README.md               # Usage documentation
```

# SOC-Estimation-CNN-BiLSTM
Zero-shot SOC Estimation with CNN-BiLSTM and Channel Attention, based on CALCE Dataset.

## 📋 Project Structure
```
SOC-Estimation-CNN-BiLSTM/
├── utils.py                # Core utilities (dataset, model, helper functions)
├── train.py                # Training script (for author only)
├── test.py                 # One-click inference (for users)
├── requirements.txt         # Dependencies
├── best_model_final.pth    # Pretrained weights
├── feature_scaler.pkl      # Feature scaler
├── soc_scaler.pkl          # SOC scaler
├── figs/                    # Figures for README
│   ├── pso_swarm_cool.gif
│   ├── loss_curve.png
│   └── prediction_example.png
└── README.md
```

## 📊 Visualisation
![step_voltage_analysis_0](https://raw.githubusercontent.com/627c/SOC-Estimation/main/figs/step_voltage_analysis_0.png)
![step_current_analysis_0](https://raw.githubusercontent.com/627c/SOC-Estimation/main/figs/step_current_analysis_0.png)


## 🚀 Quick Start (For Users – No Training Required)
### 1. Prerequisites
- Python 3.8+
- (Optional) CUDA 11.0+ for GPU acceleration

### 2. Clone Repository
```bash
git clone https://github.com/your-username/SOC-Estimation.git
cd SOC-Estimation
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Place CALCE Excel files in this exact path (create folders if missing):
```
/data/stu1/liuanqi/soc_calce/final/data/
```
File naming format:
```
CALCE_{working_condition}_Step_{temp}.xlsx
```
Example: `CALCE_DST_Step_0.xlsx`, `CALCE_US06_Step_25.xlsx`

### 5. Run Inference
```bash
python test.py
```

## 🔧 For Authors (Training)
```bash
python train.py
```
After training, update PSO/Grid Search hyperparameters in `test.py`.

## ⚠️ Important Notes
- No Git LFS required: all files < 100MB.
- Do NOT modify hardcoded paths.
- Do NOT delete `best_model_final.pth`, `feature_scaler.pkl`, `soc_scaler.pkl`.
- FileNotFoundError: check dataset path and filenames.

## 📊 Performance
Zero‑shot cross‑profile generalization:
- Train: DST | Validation: FUDS | Blind Test: US06
- Full temperature range: 0–50°C
- Detailed metrics in generated report
