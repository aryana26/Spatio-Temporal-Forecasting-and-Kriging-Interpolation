# stllm_distilgpt2_peft.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from transformers import GPT2Model, GPT2Config, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv
import warnings
warnings.filterwarnings('ignore')

from stllm import haversine_distance,create_spatial_graph,cyclic_encode_hour,cyclic_encode_month,cyclic_encode_day_of_week,SpatioTemporalDataset,inverse_transform_pm25,STLLMWithGNN

def main():
    # config
    # sequence_length = 24 * 3
    # forecast_horizon = 24
    # num_neighbors = 3
    # node_feature_dim = 11
    # gnn_hidden_dim = 64
    # gnn_heads = 2
    # gnn_layers = 2
    # d_model = None  # will be set from distilgpt2 config
    # batch_size = 1
    # learning_rate = 1e-4
    # num_epochs = 2
    # grad_accum_steps = 2
    # use_bitsandbytes = False  # set True if you have bitsandbytes installed
    # use_checkpoint = False
    # lora_r=8
    # early_stopping_patience=5
    features_to_be_used=[
                'pm25','temp','rh',
                'hour_sin','hour_cos',
                'month_sin','month_cos',
                'latitude','longitude'
            ]
    model_config = {
    "sequence_length": 24 * 1,
    "forecast_horizon": 24,
    "num_neighbors": 3,
    "node_feature_dim": len(features_to_be_used),
    "gnn_hidden_dim": 64,
    "gnn_heads": 2,
    "gnn_layers": 2,
    "d_model": None,  # will be set from distilgpt2 config
    "batch_size": 3,
    "learning_rate": 1e-4,
    "num_epochs": 20,
    "grad_accum_steps": 2,
    "use_bitsandbytes": False, 
    "use_checkpoint": False,
    "lora_r": 8,
    "early_stopping_patience": 5
}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load data
    df = pd.read_pickle('bihar_meteo_era5_may_jan_iterative_imputed.pkl')
    # df=df[(df['pm25']<=500)&(df['pm25']>=1)].reset_index(drop=True)
    df=df[(df['timestamp'].dt.year==2023)&(df['timestamp'].dt.month.isin([7]))].reset_index(drop=True)
    # create sensor_id mapping based on unique lat/lon pairs
    sensor_map = df.groupby(['latitude','longitude'])['temp'].mean().reset_index().reset_index()[['index','latitude','longitude']]
    sensor_map.columns = ['sensor_id','latitude','longitude']
    
    # sensor_map=sensor_map[sensor_map.sensor_id.isin([1,2,3,4,5,6,7,8,9,10])].reset_index(drop=True) # just to quickly run and check. 
    
    df = pd.merge(sensor_map, df[['latitude','longitude','rh','temp','pm25','timestamp']], on=['latitude','longitude'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("Loaded data:", df.shape)
    print("Unique sensors:", df['sensor_id'].nunique())


    sensor_locs = sensor_map.sort_values('sensor_id').reset_index(drop=True)


    edge_index, edge_attr = create_spatial_graph(sensor_locs, num_neighbors=model_config["num_neighbors"])
    torch.save(edge_index, "models/edge_index.pt")
    torch.save(edge_attr, "models/edge_attr.pt")
    print("Saved graph tensors for inference.")
    print("Graph edges:", edge_index.shape[1])


    timestamps = sorted(df['timestamp'].unique())
    split_idx = int(0.8 * len(timestamps))
    train_ts = timestamps[:split_idx]
    test_ts = timestamps[split_idx:]
    train_df = df[df['timestamp'].isin(train_ts)].reset_index(drop=True)
    test_df = df[df['timestamp'].isin(test_ts)].reset_index(drop=True)
    print("Train rows:", len(train_df), "Test rows:", len(test_df))


    train_ds = SpatioTemporalDataset(train_df, model_config["sequence_length"], model_config["forecast_horizon"],feature_cols=features_to_be_used)
    test_ds = SpatioTemporalDataset(test_df, model_config["sequence_length"], model_config["forecast_horizon"],feature_cols=features_to_be_used)
    train_loader = DataLoader(train_ds, batch_size=model_config["batch_size"], shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=model_config["batch_size"], shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    print("Train sequences:", len(train_ds), "Test sequences:", len(test_ds))

    model_config['feature_cols']=train_ds.feature_cols
    

    print("Loading distilgpt2...")
    try:
        if model_config["use_bitsandbytes"]:
    
            gpt2 = AutoModel.from_pretrained("distilgpt2", low_cpu_mem_usage=True)
        else:
            gpt2_config = GPT2Config.from_pretrained("distilgpt2")
            gpt2 = GPT2Model.from_pretrained("distilgpt2", config=gpt2_config)
    except Exception as e:
        print("HF load fallback:", e)
        gpt2_config = GPT2Config.from_pretrained("distilgpt2")
        gpt2 = GPT2Model.from_pretrained("distilgpt2", config=gpt2_config)


    model_config["d_model"] = getattr(gpt2.config, "hidden_size", None)
    print("Transformer hidden size (d_model):", model_config["d_model"])


    model = STLLMWithGNN(
        input_dim=model_config["node_feature_dim"],
        gnn_hidden_dim=model_config["gnn_hidden_dim"],
        gnn_heads=model_config["gnn_heads"],
        gnn_layers=model_config["gnn_layers"],
        d_model=model_config["d_model"],
        forecast_horizon=model_config["forecast_horizon"],
        num_sensors=train_ds.num_sensors,
        use_checkpoint=model_config["use_checkpoint"]
    )

    for p in gpt2.parameters():
        p.requires_grad = False


    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=model_config["lora_r"],
        lora_alpha=model_config["lora_r"]*2,
        lora_dropout=0.05,
        target_modules=["c_attn"]  
    )
    try:
        gpt2 = get_peft_model(gpt2, lora_config)
        print("Applied LoRA to transformer.")
    except Exception as e:
        print("PEFT wrapper failed, continuing without PEFT:", e)


    model.set_transformer(gpt2)


    edge_index = edge_index.long().contiguous().to(device)
    if edge_attr is not None and edge_attr.numel() > 0:
        edge_attr = edge_attr.float().contiguous().to(device)
    else:
        edge_attr = None


    model.to(device)

    model.transformer.to(device)

    optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(optimizer_params, lr=model_config["learning_rate"], weight_decay=1e-4)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    percent_trainable = 100.0 * trainable_count / total_count if total_count > 0 else 0.0
    print(f"Trainable params: {trainable_count:,}")
    print(f"Total params: {total_count:,}")
    print(f"Percent trainable: {percent_trainable:.4f}%")


    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


    model.train()
    global_step = 0
    best_val_rmse = float("inf")
    patience = model_config["early_stopping_patience"]
    patience_counter = 0
    for epoch in range(model_config["num_epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config['num_epochs']}")
        optimizer.zero_grad()
        for step, (X, y) in enumerate(pbar):
            # X: (B, S, L, F), y: (B, S, H)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(X, edge_index, edge_attr)  
                loss = criterion(outputs, y) / model_config["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % model_config["grad_accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * model_config["grad_accum_steps"]
            pbar.set_postfix({'loss': f"{(epoch_loss / (step+1)):.6f}"})


            del X, y, outputs, loss
            torch.cuda.empty_cache()


        model.eval()
        val_loss_scaled=0.0
        with torch.no_grad():
            val_preds = []
            val_trues = []
            for Xv, yv in test_loader:
                Xv = Xv.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)
                outv = model(Xv, edge_index, edge_attr) 
        
                preds_np = outv.cpu().numpy()  
                trues_np = yv.cpu().numpy()   
                loss_scaled = criterion(outv, yv)
                val_loss_scaled += loss_scaled.item() * Xv.size(0)  

        

                for b in range(preds_np.shape[0]):
                    for s, sid in enumerate(test_ds.sensor_ids):
                        pred_abs = inverse_transform_pm25(
                            train_ds.scalers[sid], sid, preds_np[b, s], len(train_ds.feature_cols)
                        )
                        true_abs = inverse_transform_pm25(
                            train_ds.scalers[sid], sid, trues_np[b, s], len(train_ds.feature_cols)
                        )
                        val_preds.append(pred_abs)
                        val_trues.append(true_abs)
        
                del Xv, yv, outv

        
            if len(val_preds) > 0:
                val_preds = np.concatenate(val_preds, axis=0)
                val_trues = np.concatenate(val_trues, axis=0)
                mae = mean_absolute_error(val_trues, val_preds)
                rmse = mean_squared_error(val_trues, val_preds, squared=False)
                mse = mean_squared_error(val_trues, val_preds, squared=True)
                mape = np.mean(np.abs((val_trues - val_preds) / (val_trues + 1e-6))) * 100
            else:
                mae = rmse = mape = mse = float('nan')
        val_loss_scaled /= len(test_loader.dataset) 

        print(f"Epoch {epoch+1} -> "
              f"Train loss (avg): {(epoch_loss/len(train_loader)):.6f}, "
              f"Scaled Val Loss (MSE): {val_loss_scaled:.6f} Val MAE: {mae:.4f}, Val RMSE: {rmse:.4f}, Val MAPE: {mape:.2f}%, Val MSE: {mse:.4f}")
        
        if val_loss_scaled < best_val_rmse:
            best_val_rmse = val_loss_scaled
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scalers': {sid: {'mean': train_ds.scalers[sid].mean_, 'scale': train_ds.scalers[sid].scale_}
                            for sid in train_ds.sensor_ids},
                'config':model_config
            }, "models/stllm_best.pth")
            print(" Saved new best model (improved val_loss_scaled).")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(" Early stopping triggered.")
                break



    # torch.save({'model_state_dict': model.state_dict(), 'scalers': {sid: {'mean': train_ds.scalers[sid].mean_, 'scale': train_ds.scalers[sid].scale_} for sid in train_ds.sensor_ids}}, "stllm_final.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
