import torch
import numpy as np
from tqdm import tqdm
import pyswarms as ps
import pandas as pd
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings('ignore')

# 导入公共代码
from utils import *

# ===================== Lightweight Training & Evaluation Function =====================
def light_train_and_evaluate(lr, cnn_kernel, lstm_hidden, train_loader, val_loader, feature_scaler, soc_scaler):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    model = CNNBiLSTM_ChannelAttn(cnn_kernel_size=cnn_kernel, lstm_hidden=lstm_hidden).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for _ in range(PSO_EVAL_EPOCHS):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            val_preds.extend(model(Xb).cpu().numpy())
            val_trues.extend(yb.cpu().numpy())
            
    val_preds_real = soc_scaler.inverse_transform(np.array(val_preds).reshape(-1, 1)).flatten()
    val_trues_real = soc_scaler.inverse_transform(np.array(val_trues).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(val_trues_real, val_preds_real))
    
    del model, optimizer
    torch.cuda.empty_cache()
    return rmse

# ===================== PSO Objective Function =====================
def pso_objective_function(swarm_pos, train_loader, val_loader, feature_scaler, soc_scaler):
    n_particles = swarm_pos.shape[0]
    costs = np.zeros(n_particles)
    for i in tqdm(range(n_particles), desc="PSO Particle Evaluation", leave=False):
        lr = 10 ** np.clip(swarm_pos[i, 0], -4.0, -2.5)
        cnn_kernel = int(np.clip(np.round(swarm_pos[i, 1]), 3, 11))
        cnn_kernel = cnn_kernel if cnn_kernel % 2 != 0 else cnn_kernel - 1
        lstm_hidden = int(np.clip(np.round(swarm_pos[i, 2]), 32, 128))
        rmse = light_train_and_evaluate(lr, cnn_kernel, lstm_hidden, train_loader, val_loader, feature_scaler, soc_scaler)
        costs[i] = rmse
        print(f"  Particle {i+1} | LR:{lr:.5f}, Kernel:{cnn_kernel}, Hidden:{lstm_hidden} -> RMSE:{rmse:.4f}%")
    return costs

# ===================== Grid Search Comparison =====================
def run_grid_search(train_loader, val_loader, feature_scaler, soc_scaler):
    print("\n🔍 Starting Grid Search comparison...")
    lr_candidates = [0.0001, 0.001, 0.003]
    kernel_candidates = [3, 7, 11]
    hidden_candidates = [32, 64, 128]
    
    best_rmse = float('inf')
    best_params = {}
    grid_results = []
    
    for lr in lr_candidates:
        for kernel in kernel_candidates:
            for hidden in hidden_candidates:
                print(f"\nTesting Grid params: LR={lr}, Kernel={kernel}, Hidden={hidden}")
                rmse = light_train_and_evaluate(lr, kernel, hidden, train_loader, val_loader, feature_scaler, soc_scaler)
                grid_results.append({"lr": lr, "kernel": kernel, "hidden": hidden, "rmse": rmse})
                print(f"Grid Validation RMSE: {rmse:.4f}%")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'lr': lr, 'cnn_kernel': kernel, 'lstm_hidden': hidden, 'rmse': rmse}
    
    pd.DataFrame(grid_results).to_csv(f"{DATA_SAVE_DIR}/grid_search_results.csv", index=False)
    print(f"\n✅ Grid Search completed, Best params: {best_params}")
    return best_params

# ===================== Hyperparameter Sensitivity Analysis =====================
def generate_hyperparam_sensitivity(train_loader, val_loader, feature_scaler, soc_scaler, kernel_fixed):
    print("\n📊 Generating hyperparameter sensitivity data...")
    lr_candidates = [0.0005, 0.001, 0.002, 0.003, 0.005]
    hidden_candidates = [32, 64, 80, 100, 128]
    
    sensitivity_matrix = np.zeros((len(hidden_candidates), len(lr_candidates)))
    
    for i, hidden in enumerate(tqdm(hidden_candidates, desc="Hidden Unit Iteration")):
        for j, lr in enumerate(lr_candidates):
            rmse = light_train_and_evaluate(lr, kernel_fixed, hidden, train_loader, val_loader, feature_scaler, soc_scaler)
            sensitivity_matrix[i, j] = rmse
    
    np.save(f"{DATA_SAVE_DIR}/hyperparam_sensitivity_matrix.npy", sensitivity_matrix)
    pd.DataFrame({
        "lr": lr_candidates,
        "hidden_units": hidden_candidates
    }).to_json(f"{DATA_SAVE_DIR}/hyperparam_params.json", orient="records")
    print("✅ Hyperparameter sensitivity data saved!")
    return sensitivity_matrix

# ===================== Full Training Function (returns loss history) =====================
def train_full_model(model, train_loader, val_loader, lr, soc_scaler, epochs=300):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7, verbose=True
    )
    
    best_val_rmse = float('inf')
    patience_counter = 0
    early_stop_patience = 30
    train_losses, val_losses, val_rmses = [], [], []
    
    for epoch in tqdm(range(epochs), desc="Full Training"):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_avg = train_loss / len(train_loader)
        train_losses.append(train_loss_avg)
        
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item()
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(yb.cpu().numpy())
        val_loss_avg = val_loss / len(val_loader)
        val_losses.append(val_loss_avg)
        
        val_preds_real = soc_scaler.inverse_transform(np.array(val_preds).reshape(-1, 1)).flatten()
        val_trues_real = soc_scaler.inverse_transform(np.array(val_trues).reshape(-1, 1)).flatten()
        val_rmse = np.sqrt(mean_squared_error(val_trues_real, val_preds_real))
        val_rmses.append(val_rmse)
        
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), f"{RESULT_DIR}/best_model.pth")
            print(f"\n✅ New best model! Validation RMSE: {val_rmse:.4f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n✅ Early stopping triggered! Training stopped at epoch {epoch+1}")
                model.load_state_dict(torch.load(f"{RESULT_DIR}/best_model.pth"))
                break
    
    np.savez(f"{DATA_SAVE_DIR}/training_history.npz",
             train_losses=train_losses,
             val_losses=val_losses,
             val_rmses=val_rmses)
    
    plt.figure(figsize=(10,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend()
    plt.title('Training & Validation Loss')
    plt.savefig(f"{RESULT_DIR}/loss_curve.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    return model, best_val_rmse, train_losses, val_losses, val_rmses

# ===================== Main Training Pipeline =====================
if __name__ == "__main__":
    print("="*80)
    print("🚀 SOC Estimation Training Pipeline")
    print("="*80)
    results = {}
    all_temps = [0, 10, 20, 25, 30, 40, 50]
    
    print("\n🔍 Loading data...")
    list_train = [load_real_data('DST', t) for t in all_temps]
    list_val   = [load_real_data('FUDS', t) for t in all_temps]
    
    df_train_concat_for_plot = pd.concat(list_train, ignore_index=True)
    df_train_concat_for_plot.to_csv(f"{DATA_SAVE_DIR}/df_train_for_correlation.csv", index=False)
    
    train_dataset = BatteryDataset(list_train, mode='train')
    val_dataset   = BatteryDataset(list_val, mode='test', feature_scaler=train_dataset.feature_scaler, soc_scaler=train_dataset.soc_scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Dataset constructed | Training set: {len(train_dataset)} pure sequence samples (DST)")

    with open(f"{DATA_SAVE_DIR}/feature_scaler.pkl", 'wb') as f:
        pickle.dump(train_dataset.feature_scaler, f)
    with open(f"{DATA_SAVE_DIR}/soc_scaler.pkl", 'wb') as f:
        pickle.dump(train_dataset.soc_scaler, f)
    print("✅ Scalers saved!")
    
    print("\n🔍 Starting PSO hyperparameter optimization...")
    PSO_N_PARTICLES = 15
    PSO_ITERS = 15
    w_max, w_min = 0.9, 0.4
    options = {'c1': 2.0, 'c2': 2.0, 'w': w_max}
    bounds = ([-4.0, 3, 32], [-2.5, 11, 128])
    
    optimizer = ps.single.GlobalBestPSO(
        n_particles=PSO_N_PARTICLES, dimensions=3, options=options, bounds=bounds
    )
    
    pso_best_cost = None
    pso_best_pos = None
    pso_convergence_history = []
    for i in tqdm(range(PSO_ITERS), desc="PSO Iteration"):
        current_w = w_max - (w_max - w_min) * (i / (PSO_ITERS - 1))
        optimizer.options['w'] = current_w
        iter_cost, iter_pos = optimizer.optimize(
            lambda x: pso_objective_function(x, train_loader, val_loader, train_dataset.feature_scaler, train_dataset.soc_scaler),
            iters=1, verbose=False
        )
        pso_convergence_history.append(iter_cost)
        if pso_best_cost is None or iter_cost < pso_best_cost:
            pso_best_cost = iter_cost
            pso_best_pos = iter_pos
        print(f"\nPSO Iteration {i+1}/{PSO_ITERS} | Current Best RMSE: {iter_cost:.4f}%")
    
    np.save(f"{DATA_SAVE_DIR}/pso_convergence_history.npy", np.array(pso_convergence_history))
    results['pso_lr'] = 10 ** pso_best_pos[0] 

    cnn_kernel_raw = int(np.round(pso_best_pos[1]))
    results['pso_cnn_kernel'] = cnn_kernel_raw if cnn_kernel_raw % 2 != 0 else cnn_kernel_raw - 1

    results['pso_lstm_hidden'] = int(np.round(pso_best_pos[2]))
    results['pso_val_rmse'] = pso_best_cost
    print(f"\n✅ PSO optimization completed, convergence data saved!")
    
    grid_best = run_grid_search(train_loader, val_loader, train_dataset.feature_scaler, train_dataset.soc_scaler)
    results['grid_lr'] = grid_best['lr']
    results['grid_cnn_kernel'] = grid_best['cnn_kernel']
    results['grid_lstm_hidden'] = grid_best['lstm_hidden']
    results['grid_val_rmse'] = grid_best['rmse']
    
    generate_hyperparam_sensitivity(train_loader, val_loader, train_dataset.feature_scaler, train_dataset.soc_scaler, results['pso_cnn_kernel'])
    
    print("\n🏋️ Full model training with PSO optimal params...")
    best_model = CNNBiLSTM_ChannelAttn(
        cnn_kernel_size=results['pso_cnn_kernel'],
        lstm_hidden=results['pso_lstm_hidden']
    ).to(device)
    best_model, final_val_rmse, _, _, _ = train_full_model(
        best_model, train_loader, val_loader, 
        lr=results['pso_lr'], soc_scaler=train_dataset.soc_scaler
    )
    torch.save(best_model.state_dict(), f"{RESULT_DIR}/best_model_final.pth")
    
    print("\n🎉 Training completed! Model saved to results folder")