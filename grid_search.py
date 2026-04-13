import numpy as np
import pandas as pd
from tqdm import tqdm

def run_grid_search(train_loader, val_loader, feature_scaler, soc_scaler, 
                    save_dir, light_train_eval_func, seed):
    """
    Core Grid Search function (Strictly 225 evaluations)
    """
    print("\n Starting Grid Search comparison (Equal Budget: 225 evaluations)...")
    
    lr_candidates = np.array([0.0001, 0.0002, 0.0004, 0.0008, 0.0016])  # 5 values, log-ish space
    kernel_candidates = [3, 5, 7, 9, 11]  # 5 odd numbers
    hidden_candidates = [32, 48, 64, 80, 96, 104, 112, 120, 128]  # 9 values
    
    best_rmse = float('inf')
    best_params = {}
    grid_results = []
    
    total_iter = len(lr_candidates) * len(kernel_candidates) * len(hidden_candidates)
    print(f"Total Grid Search evaluations: {total_iter}")
    
    for lr_idx, lr in enumerate(lr_candidates):
        for kernel_idx, kernel in enumerate(kernel_candidates):
            for hidden_idx, hidden in enumerate(hidden_candidates):
                current_iter = lr_idx * len(kernel_candidates) * len(hidden_candidates) + \
                               kernel_idx * len(hidden_candidates) + hidden_idx + 1
                
                print(f"\n[{current_iter}/{total_iter}] Grid: LR={lr:.6f}, Kernel={kernel}, Hidden={hidden}")
                
                # Call the shared training function from main_train.py
                rmse = light_train_eval_func(lr, kernel, hidden, train_loader, val_loader, feature_scaler, soc_scaler)
                
                grid_results.append({"lr": lr, "kernel": kernel, "hidden": hidden, "rmse": rmse})
                print(f"Grid RMSE: {rmse:.4f}%")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'lr': lr, 'cnn_kernel': kernel, 'lstm_hidden': hidden, 'rmse': rmse}
                    print(f"   New Grid Best!")
    
    # Save results
    pd.DataFrame(grid_results).to_csv(f"{save_dir}/grid_search_results.csv", index=False)
    print(f"\n Grid Search finished. Best: {best_params}")
    return best_params
