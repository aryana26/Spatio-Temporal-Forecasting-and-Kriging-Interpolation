# train_stllm_custom.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model.st_llm_model import ST_LLM
from datetime import datetime
import json
import math


class PM25Dataset(Dataset):
    def __init__(self, X, Y, timestamps, sensor_count, input_len, device):
        self.X = X
        self.Y = Y
        self.timestamps = timestamps
        self.sensor_count = sensor_count
        self.input_len = input_len
        self.device = device


        all_timestamps = pd.to_datetime(timestamps)
        self.time_tokens = []
        for ts in all_timestamps:
            hour_idx = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
            day_idx = ts.dayofweek
            self.time_tokens.append((hour_idx, day_idx))
        self.time_tokens = np.array(self.time_tokens)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx] 
        y = self.Y[idx] 


        input_ts_indices = np.arange(idx, idx + self.input_len)
        input_time_tokens = self.time_tokens[input_ts_indices]
        
 
        input_time_tokens = np.expand_dims(input_time_tokens, axis=1)
        input_time_tokens = np.repeat(input_time_tokens, self.sensor_count, axis=1)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        time_tensor = torch.tensor(input_time_tokens, dtype=torch.long)

        return {
            'x': x_tensor,
            'y': y_tensor,
            'time': time_tensor
        }

def collate_fn(batch):
    xs = torch.stack([item['x'] for item in batch])
    ys = torch.stack([item['y'] for item in batch])
    times = torch.stack([item['time'] for item in batch])
    return xs, ys, times

def prepare_and_load_data(df, input_len=24, pred_len=24, batch_size=32,
                         val_ratio=0.1, test_ratio=0.2, device=torch.device('cpu')):

    loc = df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    loc['sensor_id'] = loc.index
    df = df.merge(loc, on=['latitude', 'longitude'])
    df = df.sort_values(['timestamp', 'sensor_id'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    pivoted = df.pivot(index='timestamp', columns='sensor_id', values='pm25').fillna(method='ffill').fillna(method='bfill').fillna(0)
    values = pivoted.values
    timestamps = pivoted.index.values


    X, Y = [], []
    for i in range(len(values) - input_len - pred_len + 1):
        X.append(values[i:i+input_len])
        Y.append(values[i+input_len:i+input_len+pred_len])
    X = np.array(X)  
    Y = np.array(Y) 


    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    n_samples = X.shape[0]
    val_size = int(n_samples * val_ratio)
    test_size = int(n_samples * test_ratio)
    train_size = n_samples - val_size - test_size

    X_train, Y_train = X[:train_size], Y[:train_size]
    ts_train = timestamps[:train_size + input_len]
    X_val, Y_val = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
    ts_val = timestamps[train_size:train_size + val_size + input_len]
    X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]
    ts_test = timestamps[train_size + val_size:]

    sensor_count = X.shape[2]

    train_ds = PM25Dataset(X_train, Y_train, ts_train, sensor_count, input_len, device)
    val_ds = PM25Dataset(X_val, Y_val, ts_val, sensor_count, input_len, device)
    test_ds = PM25Dataset(X_test, Y_test, ts_test, sensor_count, input_len, device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, sensor_count, loc

def calculate_metrics(predictions, targets):

    preds = predictions.detach().cpu()
    targets = targets.detach().cpu()
    

    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    

    epsilon = 1e-8
    
    mse = nn.MSELoss()(preds_flat, targets_flat).item()
    rmse = math.sqrt(mse)
    mae = nn.L1Loss()(preds_flat, targets_flat).item()
    

    mape = torch.mean(torch.abs((targets_flat - preds_flat) / (targets_flat + epsilon))) * 100
    
    return {
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'mape': mape.item()
    }

def save_inference_files(model, sensor_locations, model_config, metrics, save_dir='outputs'):
    """Save all files needed for inference"""
    os.makedirs(save_dir, exist_ok=True)
    

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
    

    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    

    sensor_locations.to_csv(os.path.join(save_dir, 'sensor_locations.csv'), index=False)
    

    with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"All inference files saved to {save_dir}")

def main():
    # Parameters
    input_len = 24
    pred_len = 24
    batch_size = 16
    epochs = 100  
    lr = 1e-3
    wd = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm_layers = 6
    unfrozen = 1
    patience = 10 
    best_val_loss = float('inf')
    patience_counter = 0


    df = pd.read_pickle('../bihar_meteo_era5_may_jan_iterative_imputed.pkl')
    print(f'Dataset size: {df.shape}')

    train_loader, val_loader, test_loader, num_nodes, sensor_locations = prepare_and_load_data(df,
                                                                                             input_len=input_len,
                                                                                             pred_len=pred_len,
                                                                                             batch_size=batch_size,
                                                                                             device=device)


    model = ST_LLM(
        input_len=input_len,
        output_len=pred_len,
        num_nodes=num_nodes,
        input_dim=1,
        llm_layer=llm_layers,
        U=unfrozen,
        device=device
    )

    model.to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_percentage:.2f}%")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xs, ys, time_tokens in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xs, ys, time_tokens = xs.to(device), ys.to(device), time_tokens.to(device)
            optimizer.zero_grad()
            predictions = model(xs, time_tokens)
            
            ys_reshaped = ys.permute(0, 3, 2, 1)
            loss = criterion(predictions, ys_reshaped)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xs, ys, time_tokens in val_loader:
                xs, ys, time_tokens = xs.to(device), ys.to(device), time_tokens.to(device)
                preds = model(xs, time_tokens)
                ys_reshaped = ys.permute(0, 3, 2, 1)
                loss = criterion(preds, ys_reshaped)
                val_loss += loss.item()
                

                all_preds.append(preds)
                all_targets.append(ys_reshaped)

        avg_val_loss = val_loss / len(val_loader)
        training_history['val_loss'].append(avg_val_loss)
        

        val_preds = torch.cat(all_preds)
        val_targets = torch.cat(all_targets)
        val_metrics = calculate_metrics(val_preds, val_targets)
        training_history['val_metrics'].append(val_metrics)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val RMSE: {val_metrics["rmse"]:.4f}')
        print(f'  Val MSE: {val_metrics["mse"]:.4f}')
        print(f'  Val MAE: {val_metrics["mae"]:.4f}')
        print(f'  Val MAPE: {val_metrics["mape"]:.2f}%')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            

            torch.save(model.state_dict(), 'outputs/best_model.pth')
            

            model_config = {
                'input_len': input_len,
                'output_len': pred_len,
                'num_nodes': num_nodes,
                'input_dim': 1,
                'llm_layer': llm_layers,
                'U': unfrozen,
                'device': str(device)
            }
            
            save_inference_files(model, sensor_locations, model_config, {
                'best_val_loss': best_val_loss,
                'best_epoch': epoch + 1,
                'val_metrics': val_metrics,
                'trainable_params': trainable_params,
                'total_params': total_params,
                'trainable_percentage': trainable_percentage
            })
            
            print(f'  ↳ New best model saved! (Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  ↳ Early stopping counter: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs!')
                break

    print("\n=== Final Test Evaluation ===")
    model.load_state_dict(torch.load('outputs/best_model.pth'))
    model.eval()
    
    test_loss = 0
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for xs, ys, time_tokens in test_loader:
            xs, ys, time_tokens = xs.to(device), ys.to(device), time_tokens.to(device)
            preds = model(xs, time_tokens)
            ys_reshaped = ys.permute(0, 3, 2, 1)
            loss = criterion(preds, ys_reshaped)
            test_loss += loss.item()
            
            all_test_preds.append(preds)
            all_test_targets.append(ys_reshaped)
    
    avg_test_loss = test_loss / len(test_loader)
    test_preds = torch.cat(all_test_preds)
    test_targets = torch.cat(all_test_targets)
    test_metrics = calculate_metrics(test_preds, test_targets)
    
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test RMSE: {test_metrics["rmse"]:.4f}')
    print(f'Test MSE: {test_metrics["mse"]:.4f}')
    print(f'Test MAE: {test_metrics["mae"]:.4f}')
    print(f'Test MAPE: {test_metrics["mape"]:.2f}%')
    

    final_metrics = {
        'test_loss': avg_test_loss,
        'test_metrics': test_metrics,
        'training_history': training_history
    }
    
    with open('outputs/final_test_results.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    main()