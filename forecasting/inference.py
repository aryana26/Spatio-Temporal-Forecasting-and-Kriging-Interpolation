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
# import your model class from the same file where you defined it
# from stllm_distilgpt2_peft import STLLMWithGNN, SpatioTemporalDataset, cyclic_encode_...  # if needed
# If the classes are in the same file, just run this script from that directory so it can import.
from stllm import STLLMWithGNN  # keep names exactly the same
# Note: We do not use SpatioTemporalDataset inside the inference loop to avoid refitting scalers.
# We'll create windows manually using saved scaler info.

from transformers import GPT2Model, GPT2Config, AutoModel
from peft import get_peft_model, LoraConfig, TaskType

# ---------------------------
# helper: inverse transform using saved mean/scale dict
# ---------------------------
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

# ---------------------------
# inference function (windowed sliding sequences)
# ---------------------------
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
    Produces predictions for every valid sliding input window:
      sequence windows start at i=0 .. max_seq-1 where max_seq = T - sequence_length - forecast_horizon + 1
    Args:
      df: DataFrame with columns at least ['sensor_id','timestamp','pm25','temp','rh','latitude','longitude']
      model: STLLMWithGNN instance with transformer already attached and state_dict loaded
      scaler_info_dict: {sensor_id: {'mean': <F-array>, 'scale': <F-array>}, ...}
      edge_index, edge_attr: loaded tensors (on device)
      feature_cols: list of feature names in order used during training (default below)
      sequence_length: L
      forecast_horizon: H (must match training horizon typically)
      batch_size_seq: how many sliding windows to forward at once (keeps memory small; default 1)
    Returns:
      df_pred: DataFrame with columns ['sensor_id','timestamp','pm25_pred','pm25_true']
      metrics: dict with MAE, RMSE, MAPE, MSE computed over all predictions
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

    # Build per-sensor arrays (sorted by sensor_id)
    sensor_ids = sorted(df['sensor_id'].unique())
    sensor_to_index = {sid: i for i,sid in enumerate(sensor_ids)}
    S = len(sensor_ids)

    # build per-sensor raw arrays and timestamps
    per_sensor_vals = {}
    per_sensor_timestamps = {}
    per_sensor_features = {}  # raw features (unscaled) -> so we can scale using saved mean/scale

    # Add cyclic features to df if missing
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
        # ensure feature columns exist
        arr = sub[feature_cols].values.astype(np.float32)  # shape (T_s, F)
        per_sensor_features[sid] = arr
        per_sensor_vals[sid] = sub['pm25'].values.astype(np.float32)

    # compute max sequences (we only loop until all sensors can provide full windows)
    min_T = min(arr.shape[0] for arr in per_sensor_features.values())
    max_sequences = min_T - sequence_length - forecast_horizon + 1
    if max_sequences < 1:
        raise ValueError("Not enough timepoints per sensor for sequence_length + forecast_horizon")

    results = []  # will collect (sid, timestamp, pred, true)

    # We'll process sliding windows in small batches of windows for thriftiness
    for seq_start in tqdm(range(0, max_sequences, batch_size_seq), desc="Inference windows"):
        seq_batch = list(range(seq_start, min(seq_start + batch_size_seq, max_sequences)))
        batch_size_actual = len(seq_batch)

        # Build batch X of shape (B, S, L, F)
        # B = batch_size_actual
        B = batch_size_actual
        Xbat = np.zeros((B, S, sequence_length, F), dtype=np.float32)
        Ybat_true = np.zeros((B, S, forecast_horizon), dtype=np.float32)  # ground truth pm25 (raw)
        tsbat = [[None]*S for _ in range(B)]  # to store forecast timestamps for each sensor

        for b_idx, start_idx in enumerate(seq_batch):
            for s_idx, sid in enumerate(sensor_ids):
                arr = per_sensor_features[sid]  # raw features (T_s, F)
                # Scale using saved scaler info for this sensor
                if sid not in scaler_info_dict:
                    raise KeyError(f"Scaler info for sensor {sid} not found in saved scalers")
                scaler_info = scaler_info_dict[sid]
                mean = np.asarray(scaler_info['mean'], dtype=np.float32)
                scale = np.asarray(scaler_info['scale'], dtype=np.float32)
                window_raw = arr[start_idx:start_idx+sequence_length]  # (L, F)
                # scale: (val - mean)/scale  -> but your training used StandardScaler.fit_transform which is (x - mean)/scale
                scaled = (window_raw - mean[None, :]) / (scale[None, :] + 1e-12)
                Xbat[b_idx, s_idx] = scaled

                # true raw pm25
                pm25_vals = per_sensor_vals[sid]
                true_window = pm25_vals[start_idx+sequence_length : start_idx+sequence_length+forecast_horizon]
                if len(true_window) != forecast_horizon:
                    # this shouldn't happen but guard
                    true_window = np.pad(true_window, (0, max(0, forecast_horizon - len(true_window))), 'constant', constant_values=np.nan)
                Ybat_true[b_idx, s_idx] = true_window

                # forecast timestamps for mapping
                ts_window = per_sensor_timestamps[sid][start_idx+sequence_length : start_idx+sequence_length+forecast_horizon]
                tsbat[b_idx][s_idx] = ts_window

        # convert Xbat to torch and run model
        X_t = torch.from_numpy(Xbat).to(device)  # shape (B, S, L, F)
        with torch.no_grad():
            out_t = model(X_t, edge_index.to(device), edge_attr.to(device) if edge_attr is not None else None)  # (B, S, H)
        out_np = out_t.cpu().numpy()

        # unscale predictions per sensor & append to results
        for b_idx, start_idx in enumerate(seq_batch):
            for s_idx, sid in enumerate(sensor_ids):
                pred_scaled = out_np[b_idx, s_idx, :]  # (H,)
                # inverse transform pm25 scaled -> raw
                pred_raw = inverse_transform_pm25_from_scalerinfo(scaler_info_dict[sid], pred_scaled, F)
                true_raw = Ybat_true[b_idx, s_idx, :]  # already raw
                ts_seq = tsbat[b_idx][s_idx]
                for h in range(forecast_horizon):
                    results.append({
                        'sensor_id': sid,
                        'timestamp': pd.to_datetime(ts_seq[h]),
                        'pm25_pred': float(pred_raw[h]),
                        'pm25_true': float(true_raw[h])  # may be nan if missing
                    })

        # free tensors
        del X_t, out_t
        torch.cuda.empty_cache()

    # create df
    df_pred = pd.DataFrame(results)

    # # compute global metrics on non-nan true values
    # mask = ~df_pred['pm25_true'].isna()
    # if mask.sum() > 0:
    #     y_true = df_pred.loc[mask, 'pm25_true'].values
    #     y_pred = df_pred.loc[mask, 'pm25_pred'].values
    #     mae = mean_absolute_error(y_true, y_pred)
    #     rmse = mean_squared_error(y_true, y_pred, squared=False)
    #     mse = mean_squared_error(y_true, y_pred, squared=True)
    #     mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100.0
    # else:
    #     mae = rmse = mse = mape = float('nan')

    # metrics = {'MAE': float(mae), 'RMSE': float(rmse), 'MSE': float(mse), 'MAPE': float(mape)}
    return df_pred

# ---------------------------
# full driver to load everything and call inference
# ---------------------------
def driver(
    df,
    ckpt_path = "models/stllm_best.pth",
    edge_index_path = "models/edge_index.pt",
    edge_attr_path = "models/edge_attr.pt",
    # df_path = "new_data.csv",   # path to csv/pickle with the data you want to run inference on
    
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

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_config=ckpt['config']
    sequence_length=model_config["sequence_length"]
    forecast_horizon=model_config["forecast_horizon"]

    # extract scalers info
    if 'scalers' not in ckpt:
        raise KeyError("Checkpoint does not include 'scalers'. You saved scalers at training as dict? check ckpt.")
    scaler_info_dict = ckpt['scalers']  # {sid: {'mean':..., 'scale':...}}

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
        # proceed with plain gpt2 (may still work if checkpoint contains plain weights)

    # derive d_model from transformer config
    d_model = getattr(gpt2.config, "hidden_size", None)
    if d_model is None:
        raise RuntimeError("Could not determine transformer hidden size (d_model).")

    # instantiate STLLMWithGNN with same hyperparams used in training
    # IMPORTANT: If you used different gnn_hidden_dim / gnn_heads / gnn_layers / input_dim in training,
    # change them here to match training exactly. Here I'm using the defaults you used earlier.
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

    # set transformer on the model BEFORE loading state_dict
    model.set_transformer(gpt2)

    # load saved model_state_dict (strict=False to allow small name differences from PEFT)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Model state_dict loaded (strict=False).")

    # move model/transformer to device
    model.to(device)
    try:
        model.transformer.to(device)
    except Exception:
        pass

    # load graph
    edge_index = torch.load(edge_index_path).to(device)
    edge_attr = torch.load(edge_attr_path).to(device) if edge_attr_path is not None else None

    # load input dataframe
    # support csv or pickle
    # if df_path.endswith(".csv"):
    #     df = pd.read_csv(df_path, parse_dates=['timestamp'])
    # else:
    #     df = pd.read_pickle(df_path) if df_path.endswith('.pkl') or df_path.endswith('.pickle') else pd.read_csv(df_path, parse_dates=['timestamp'])
    # df=pd.read_parquet(df_path)
    # run inference
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

    # print("Inference done. Metrics:", metrics)
    # save results
    # df_pred.to_csv("inference_predictions.csv", index=False)
    # print("Saved inference_predictions.csv")
    return df_pred

# # If running as script:
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

    # -----------------------------
    # recursive forecasting beyond last available timestamp
    # -----------------------------
    future_steps = 24  # next 24 hours
    sensor_ids = sorted(df['sensor_id'].unique())
    per_sensor_features = {sid: df[df['sensor_id']==sid].sort_values('timestamp')[model_config["feature_cols"]].values.astype(np.float32) 
                           for sid in sensor_ids}
    per_sensor_timestamps = {sid: df[df['sensor_id']==sid].sort_values('timestamp')['timestamp'].values 
                             for sid in sensor_ids}
    
    # last window per sensor
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
    
