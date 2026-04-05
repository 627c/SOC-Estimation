import torch
import numpy as np
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings('ignore')

from utils import *
import pickle

# ===================== Main Test Pipeline =====================
if __name__ == "__main__":
    print("="*80)
    print("🚀 SOC Estimation Test Pipeline (No Training)")
    print("="*80)
    
    all_temps = [0, 10, 20, 25, 30, 40, 50]
    results = {}
    
    results['pso_lr'] = 0.000588  
    results['pso_cnn_kernel'] = 7
    results['pso_lstm_hidden'] = 113
    results['pso_val_rmse'] = 1.5709
    results['grid_lr'] = 0.0001
    results['grid_cnn_kernel'] = 11
    results['grid_lstm_hidden'] = 64
    results['grid_val_rmse'] = 1.7118

    print("\n🔍 Loading pre-trained model and scalers...")
    with open(f"{DATA_SAVE_DIR}/feature_scaler.pkl", 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(f"{DATA_SAVE_DIR}/soc_scaler.pkl", 'rb') as f:
        soc_scaler = pickle.load(f)
    
    model = CNNBiLSTM_ChannelAttn(
        cnn_kernel_size=results['pso_cnn_kernel'],
        lstm_hidden=results['pso_lstm_hidden']
    ).to(device)
    model.load_state_dict(torch.load(f"{RESULT_DIR}/best_model_final.pth", map_location=device))
    print("✅ Model loaded successfully!")

    print("\n🧪 Starting temperature-wise testing...")
    test_results = {temp: {} for temp in all_temps}
    for temp in tqdm(all_temps, desc="Temperature-wise Testing"):
        fuds_rmse, fuds_mae = test_single_temp_model(model, feature_scaler, soc_scaler, temp, 'FUDS')
        us06_rmse, us06_mae = test_single_temp_model(model, feature_scaler, soc_scaler, temp, 'US06')
        test_results[temp]['FUDS'] = {'rmse': fuds_rmse, 'mae': fuds_mae}
        test_results[temp]['US06'] = {'rmse': us06_rmse, 'mae': us06_mae}
        print(f"✅ {temp}℃ Test completed | US06 RMSE:{us06_rmse:.4f}%")
    results.update(test_results)

    print("\n📄 Generating report...")
    generate_detail_report(results, all_temps)

    print("\n" + "="*80)
    print("🎉 All tests completed! Report & figures saved to results folder")
    print("="*80)
