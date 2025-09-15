# inference_driver.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
import matplotlib.pyplot as plt

from stllm import STLLMWithGNN  # keep names exactly the same

from transformers import GPT2Model, GPT2Config, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

def inverse_transform_pm25_from_scalerinfo(scaler_info, data_np, num_features):
    """
    scaler_info: {'mean': array_like (F,), 'scale': array_like (F,)}
    data_np: 1D array of pm25 values in scaled space (H,)
    num_features: F (number of features used during training)
    returns: 1D array of unscaled pm25 values (H,)
    """
    mean = np.asarray(scaler_info['mean'], dtype=np.float32)
    scale = np.asarray(scaler_info['scale'], dtype=np.float32)
    # build dummy (H, F)
    H = len(data_np)
    dummy = np.zeros((H, num_features), dtype=np.float32)
    dummy[:, 0] = data_np
    unscaled = dummy * scale + mean
    return unscaled[:, 0]


def run_inference_with_saved_scalers(
    df,
    model,
    scaler_info_dict,
    edge_index,
    edge_attr,
    sequence_length,
    forecast_horizon,
    feature_cols=None,
    device='cpu',
    batch_size_seq=1
):
    """
      sequence windows start at i=0 .. max_seq-1 where max_seq = T - sequence_length - forecast_horizon + 1
    needs:
      df: df with cols['sensor_id','timestamp','pm25','temp','rh','latitude','longitude']
      model: STLLMWithGNN instance
      scaler_info_dict
      edge_index, edge_attr
      feature_cols
      sequence_length
      forecast_horizon
      batch_size_seq
    """
    model.eval()
    model.to(device)

    if feature_cols is None:
        feature_cols = [
            'pm25','temp','rh',
                'hour_sin','hour_cos',
                'month_sin','month_cos',
                'latitude','longitude'
        ]
    F = len(feature_cols)

    sensor_ids = sorted(df['sensor_id'].unique())
    sensor_to_index = {sid: i for i,sid in enumerate(sensor_ids)}
    S = len(sensor_ids)

    per_sensor_vals = {}
    per_sensor_timestamps = {}
    per_sensor_features = {} 

    if 'hour_sin' not in df.columns:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        df['dow'] = df['timestamp'].dt.dayofweek
        df['hour_sin'], df['hour_cos'] = zip(*df['hour'].map(lambda h: (np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24))))
        df['month_sin'], df['month_cos'] = zip(*df['month'].map(lambda m: (np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12))))
        # df['dow_sin'], df['dow_cos'] = zip(*df['dow'].map(lambda d: (np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7))))
        df.drop(columns=['hour','month','dow'], inplace=True)

    # pivot per sensor
    for sid in sensor_ids:
        sub = df[df['sensor_id']==sid].sort_values('timestamp').reset_index(drop=True)
        per_sensor_timestamps[sid] = sub['timestamp'].values
        arr = sub[feature_cols].values.astype(np.float32)  # shape (T_s, F)
        per_sensor_features[sid] = arr
        per_sensor_vals[sid] = sub['pm25'].values.astype(np.float32)

    min_T = min(arr.shape[0] for arr in per_sensor_features.values())
    max_sequences = min_T - sequence_length - forecast_horizon + 1
    if max_sequences < 1:
        raise ValueError("Not enough timepoints per sensor for sequence_length + forecast_horizon")

    results = []  

    # sliding windows 
    for seq_start in tqdm(range(0, max_sequences, batch_size_seq), desc="Inference windows"):
        seq_batch = list(range(seq_start, min(seq_start + batch_size_seq, max_sequences)))
        batch_size_actual = len(seq_batch)

        B = batch_size_actual
        Xbat = np.zeros((B, S, sequence_length, F), dtype=np.float32)
        Ybat_true = np.zeros((B, S, forecast_horizon), dtype=np.float32)  
        tsbat = [[None]*S for _ in range(B)]  

        for b_idx, start_idx in enumerate(seq_batch):
            for s_idx, sid in enumerate(sensor_ids):
                arr = per_sensor_features[sid]  # raw features (T_s, F)
                if sid not in scaler_info_dict:
                    raise KeyError(f"Scaler info for sensor {sid} not found in saved scalers")
                scaler_info = scaler_info_dict[sid]
                mean = np.asarray(scaler_info['mean'], dtype=np.float32)
                scale = np.asarray(scaler_info['scale'], dtype=np.float32)
                window_raw = arr[start_idx:start_idx+sequence_length]  # (L, F)
                scaled = (window_raw - mean[None, :]) / (scale[None, :] + 1e-12)
                Xbat[b_idx, s_idx] = scaled


                pm25_vals = per_sensor_vals[sid]
                true_window = pm25_vals[start_idx+sequence_length : start_idx+sequence_length+forecast_horizon]
                if len(true_window) != forecast_horizon:
                    true_window = np.pad(true_window, (0, max(0, forecast_horizon - len(true_window))), 'constant', constant_values=np.nan)
                Ybat_true[b_idx, s_idx] = true_window


                ts_window = per_sensor_timestamps[sid][start_idx+sequence_length : start_idx+sequence_length+forecast_horizon]
                tsbat[b_idx][s_idx] = ts_window


        X_t = torch.from_numpy(Xbat).to(device) 
        with torch.no_grad():
            out_t = model(X_t, edge_index.to(device), edge_attr.to(device) if edge_attr is not None else None)  # (B, S, H)
        out_np = out_t.cpu().numpy()


        for b_idx, start_idx in enumerate(seq_batch):
            for s_idx, sid in enumerate(sensor_ids):
                pred_scaled = out_np[b_idx, s_idx, :]  

                pred_raw = inverse_transform_pm25_from_scalerinfo(scaler_info_dict[sid], pred_scaled, F)
                true_raw = Ybat_true[b_idx, s_idx, :]  
                ts_seq = tsbat[b_idx][s_idx]
                for h in range(forecast_horizon):
                    results.append({
                        'sensor_id': sid,
                        'timestamp': pd.to_datetime(ts_seq[h]),
                        'pm25_pred': float(pred_raw[h]),
                        'pm25_true': float(true_raw[h])  
                    })

        # too much load on a30, need to free up
        del X_t, out_t
        torch.cuda.empty_cache()


    df_pred = pd.DataFrame(results)
    return df_pred

def driver(
    df,
    ckpt_path = "models/stllm_best.pth",
    edge_index_path = "models/edge_index.pt",
    edge_attr_path = "models/edge_attr.pt",
    
    sequence_length = None,
    forecast_horizon = None,
    feature_cols = None,
    device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print('Using device:',device)


    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_config=ckpt['config']
    sequence_length=model_config["sequence_length"]
    forecast_horizon=model_config["forecast_horizon"]


    if 'scalers' not in ckpt:
        raise KeyError("Checkpoint does not include 'scalers'. you messed up dawg ")
    scaler_info_dict = ckpt['scalers']

    # number of sensors -> use scaler keys
    sensor_ids = sorted([int(k) for k in scaler_info_dict.keys()]) if isinstance(list(scaler_info_dict.keys())[0], (str,)) else sorted(scaler_info_dict.keys())
    num_sensors = len(sensor_ids)

    # load transformer backbone and wrap with LoRA (same hyperparams as training)
    print("Loading transformer backbone (distilgpt2) and wrapping with LoRA...")
    gpt2_config = GPT2Config.from_pretrained("distilgpt2")
    gpt2 = GPT2Model.from_pretrained("distilgpt2", config=gpt2_config)
    # freeze base weights (optional)
    for p in gpt2.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=model_config["lora_r"],
        lora_alpha=model_config["lora_r"]*2,
        lora_dropout=0.05,
        target_modules=["c_attn"]
    )
    try:
        gpt2 = get_peft_model(gpt2, lora_cfg)
        print("PEFT/LoRA wrapper applied to transformer.")
    except Exception as e:
        print("Warning: PEFT wrapping failed or not installed:", e)

    # derive d_model from transformer config
    d_model = getattr(gpt2.config, "hidden_size", None)
    if d_model is None:
        raise RuntimeError("Could not determine transformer hidden size (d_model).")


    input_dim = model_config["node_feature_dim"]
    gnn_hidden_dim = model_config["gnn_hidden_dim"]
    gnn_heads = model_config["gnn_heads"]
    gnn_layers = model_config["gnn_layers"]

    model = STLLMWithGNN(
        input_dim=input_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_heads=gnn_heads,
        gnn_layers=gnn_layers,
        d_model=d_model,
        forecast_horizon=forecast_horizon,
        num_sensors=num_sensors,
        use_checkpoint=False
    )

    model.set_transformer(gpt2)


    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Model state_dict loaded (strict=False).")


    model.to(device)
    try:
        model.transformer.to(device)
    except Exception:
        pass


    edge_index = torch.load(edge_index_path).to(device)
    edge_attr = torch.load(edge_attr_path).to(device) if edge_attr_path is not None else None


    df_pred = run_inference_with_saved_scalers(
        df=df,
        model=model,
        scaler_info_dict=scaler_info_dict,
        edge_index=edge_index,
        edge_attr=edge_attr,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_cols=model_config["feature_cols"],
        device=device,
        batch_size_seq=1
    )

    return df_pred


if __name__ == "__main__":
    df_new = pd.read_pickle('bihar_meteo_era5_may_jan_iterative_imputed.pkl')  # new data to forecast
    df_new=df_new[(df_new['timestamp'].dt.year==2023)&(df_new['timestamp'].dt.month.isin([7]))].reset_index(drop=True)

    sensor_map = df_new.groupby(['latitude','longitude'])['temp'].mean().reset_index().reset_index()[['index','latitude','longitude']]
    sensor_map.columns = ['sensor_id','latitude','longitude']
    
    # sensor_map=sensor_map[sensor_map.sensor_id.isin([1,2,3,4,5,6,7,8,9,10])].reset_index(drop=True) # just to quickly run and check. 
    
    df_new = pd.merge(sensor_map, df_new[['latitude','longitude','rh','temp','pm25','timestamp']], on=['latitude','longitude'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])


    ckpt = torch.load("models/stllm_best.pth", map_location='cpu')
    model_config=ckpt['config']

    df_pred = driver(
        df=df_new,
        ckpt_path="models/stllm_best.pth",
        edge_index_path="models/edge_index.pt",
        edge_attr_path="models/edge_attr.pt",
        sequence_length = model_config["sequence_length"],
    forecast_horizon = model_config["forecast_horizon"]
    )
    # from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error, mean_squared_error
    print('mean_absolute_percentage_error:',mean_absolute_percentage_error(df_pred['pm25_true'],df_pred['pm25_pred']))
    print('mean_absolute_error:',mean_absolute_error(df_pred['pm25_true'],df_pred['pm25_pred']))
    print('root_mean_squared_error:',root_mean_squared_error(df_pred['pm25_true'],df_pred['pm25_pred']))
    print('mean_squared_error:',mean_squared_error(df_pred['pm25_true'],df_pred['pm25_pred']))
    
    df_pred.to_parquet('models/df_pred_forecasted_stllm.parquet')
    df_pred['timestamp']=pd.to_datetime(df_pred['timestamp'])


    future_steps = 24  # next 24 hours
    sensor_ids = sorted(df['sensor_id'].unique())
    per_sensor_features = {sid: df[df['sensor_id']==sid].sort_values('timestamp')[model_config["feature_cols"]].values.astype(np.float32) 
                           for sid in sensor_ids}
    per_sensor_timestamps = {sid: df[df['sensor_id']==sid].sort_values('timestamp')['timestamp'].values 
                             for sid in sensor_ids}
    

    last_windows = {sid: per_sensor_features[sid][-sequence_length:].copy() for sid in sensor_ids}
    last_timestamps = {sid: per_sensor_timestamps[sid][-1] for sid in sensor_ids}

    future_results = []
    for step in range(future_steps):
        Xbat = np.zeros((1, len(sensor_ids), sequence_length, len(model_config["feature_cols"])), dtype=np.float32)
        for s_idx, sid in enumerate(sensor_ids):
            scaler_info = scaler_info_dict[sid]
            mean = np.asarray(scaler_info['mean'], dtype=np.float32)
            scale = np.asarray(scaler_info['scale'], dtype=np.float32)
            scaled_window = (last_windows[sid] - mean[None,:]) / (scale[None,:]+1e-12)
            Xbat[0, s_idx] = scaled_window
        
        X_t = torch.from_numpy(Xbat).to(device)
        with torch.no_grad():
            out_t = model(X_t, edge_index.to(device), edge_attr.to(device) if edge_attr is not None else None)
        out_np = out_t.cpu().numpy()
        
        for s_idx, sid in enumerate(sensor_ids):
            pred_scaled = out_np[0, s_idx, 0]  # take first step only
            pred_raw = inverse_transform_pm25_from_scalerinfo(scaler_info_dict[sid], np.array([pred_scaled]), len(model_config["feature_cols"]))[0]
            
            next_ts = pd.to_datetime(last_timestamps[sid]) + pd.Timedelta(hours=1)
            last_timestamps[sid] = next_ts
            
            future_results.append({
                'sensor_id': sid,
                'timestamp': next_ts,
                'pm25_pred': float(pred_raw),
                'pm25_true': np.nan
            })
            
            new_row = last_windows[sid][-1].copy()
            new_row[0] = pred_raw
            last_windows[sid] = np.vstack([last_windows[sid][1:], new_row])
            
    df_future = pd.DataFrame(future_results)
    df_all = pd.concat([df_pred, df_future]).reset_index(drop=True)

    
    df_all.groupby('timestamp')[['pm25_true','pm25_pred']].mean().plot(figsize=(15,6))
    plt.savefig("models/forecast_vs_true_extended.png", dpi=150)
    
