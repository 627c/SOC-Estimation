import numpy as np
import pandas as pd
from tqdm import tqdm

def run_random_search(train_loader, val_loader, feature_scaler, soc_scaler,
                      save_dir, light_train_eval_func, seed, n_evaluations=225):
    """
    Core Random Search function (Strictly 225 evaluations)
    """
    print("\n🎲 Starting Random Search comparison (Equal Budget: 225 evaluations)...")
    
    # Fix random state for reproducibility
    rs = np.random.RandomState(seed=seed)
    
    best_rmse = float('inf')
    best_params = {}
    random_results = []
    
    for i in tqdm(range(n_evaluations), desc="Random Search"):
        # 1. Learning rate: log-uniform in [10^-4, 10^-2.5]
        lr_log = rs.uniform(-4.0, -2.5)
        lr = 10 ** lr_log
        
        # 2. CNN kernel: random choice from odd numbers [3,5,7,9,11]
        kernel = rs.choice([3, 5, 7, 9, 11])
        
        # 3. LSTM hidden units: uniform integer in [32, 128]
        hidden = rs.randint(32, 129)
        
        print(f"\n[{i+1}/{n_evaluations}] Random: LR={lr:.6f}, Kernel={kernel}, Hidden={hidden}")
        
        # Call the shared training function from main_train.py
        rmse = light_train_eval_func(lr, kernel, hidden, train_loader, val_loader, feature_scaler, soc_scaler)
        
        random_results.append({"lr": lr, "kernel": kernel, "hidden": hidden, "rmse": rmse})
        print(f"Random RMSE: {rmse:.4f}%")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'lr': lr, 'cnn_kernel': kernel, 'lstm_hidden': hidden, 'rmse': rmse}
            print(f"  🎉 New Random Best!")
    
    # Save results
    pd.DataFrame(random_results).to_csv(f"{save_dir}/random_search_results.csv", index=False)
    print(f"\n✅ Random Search finished. Best: {best_params}")
    return best_params