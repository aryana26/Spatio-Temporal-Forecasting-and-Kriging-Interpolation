# inference.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


from model.st_llm_model import ST_LLM

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

class STLLMInference:
    def __init__(self, model_dir='outputs', device=None):
        self.model_dir = model_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model_and_config()
        
    def load_model_and_config(self):
        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            self.config = json.load(f)
        

        self.sensor_locations = pd.read_csv(os.path.join(self.model_dir, 'sensor_locations.csv'))
        

        self.model = ST_LLM(
            input_len=self.config['input_len'],
            output_len=self.config['output_len'],
            num_nodes=self.config['num_nodes'],
            input_dim=self.config['input_dim'],
            llm_layer=self.config['llm_layer'],
            U=self.config['U'],
            device=self.device
        )
        

        model_path = os.path.join(self.model_dir, 'model_weights.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Input length: {self.config['input_len']}, Output length: {self.config['output_len']}")
        print(f"Number of sensors: {self.config['num_nodes']}")
    
    def prepare_temporal_tokens(self, timestamps):

        timestamps = pd.to_datetime(timestamps)
        time_tokens = []
        for ts in timestamps:
            hour_idx = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
            day_idx = ts.dayofweek
            time_tokens.append((hour_idx, day_idx))
        return np.array(time_tokens)
    
    def create_input_sequence(self, df, start_idx, input_len):


        pivoted = df.pivot(index='timestamp', columns='sensor_id', values='pm25').fillna(method='ffill').fillna(method='bfill').fillna(0)
        values = pivoted.values
        

        sequence = values[start_idx:start_idx + input_len]
        sequence = sequence[..., np.newaxis]  
        

        timestamps = pivoted.index.values[start_idx:start_idx + input_len]
        time_tokens = self.prepare_temporal_tokens(timestamps)
        

        time_tokens = np.expand_dims(time_tokens, axis=1)
        time_tokens = np.repeat(time_tokens, self.config['num_nodes'], axis=1)
        
        return sequence, time_tokens, timestamps
    
    def forecast(self, df, forecast_horizon=24, batch_size=1):

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        

        df = df.merge(self.sensor_locations, on=['latitude', 'longitude'])
        df = df.sort_values(['timestamp', 'sensor_id'])
        

        pivoted = df.pivot(index='timestamp', columns='sensor_id', values='pm25').fillna(method='ffill').fillna(method='bfill').fillna(0)
        values = pivoted.values
        timestamps = pivoted.index.values
        
        input_len = self.config['input_len']
        total_timesteps = len(values)
        
        results = []
        

        for start_idx in tqdm(range(0, total_timesteps - input_len - forecast_horizon + 1, batch_size), 
                             desc="Generating forecasts"):
            
            batch_indices = list(range(start_idx, min(start_idx + batch_size, total_timesteps - input_len - forecast_horizon + 1)))
            
            batch_sequences = []
            batch_time_tokens = []
            batch_targets = []
            batch_timestamps = []
            
            for idx in batch_indices:
                # Create input sequence
                sequence, time_tokens, seq_timestamps = self.create_input_sequence(df, idx, input_len)
                
                # Get target values
                target_start = idx + input_len
                target_end = target_start + forecast_horizon
                target_values = values[target_start:target_end]
                target_timestamps = timestamps[target_start:target_end]
                
                batch_sequences.append(sequence)
                batch_time_tokens.append(time_tokens)
                batch_targets.append(target_values)
                batch_timestamps.append(target_timestamps)
            

            batch_sequences = torch.tensor(np.array(batch_sequences), dtype=torch.float32).to(self.device)
            batch_time_tokens = torch.tensor(np.array(batch_time_tokens), dtype=torch.long).to(self.device)
            

            with torch.no_grad():
                predictions = self.model(batch_sequences, batch_time_tokens)
                predictions = predictions.cpu().numpy()
            

            for i, idx in enumerate(batch_indices):
                preds = predictions[i]  
                targets = batch_targets[i]  
                pred_timestamps = batch_timestamps[i]
                

                preds = preds[0].T
                
                for t in range(forecast_horizon):
                    for sensor_idx in range(self.config['num_nodes']):
                        sensor_id = self.sensor_locations.iloc[sensor_idx]['sensor_id']
                        results.append({
                            'sensor_id': sensor_id,
                            'timestamp': pd.to_datetime(pred_timestamps[t]),
                            'pm25_pred': float(preds[t, sensor_idx]),
                            'pm25_true': float(targets[t, sensor_idx])
                        })
        
        return pd.DataFrame(results)
    
    def forecast_next_hours(self, df, hours=24):

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        

        last_timestamp = df['timestamp'].max()
        input_len = self.config['input_len']
        

        recent_data = df[df['timestamp'] > (last_timestamp - timedelta(hours=input_len))]
        
        if len(recent_data) < input_len * self.config['num_nodes']:
            raise ValueError(f"Not enough recent data. Need {input_len} hours for all sensors.")
        

        pivoted = recent_data.pivot(index='timestamp', columns='sensor_id', values='pm25').fillna(method='ffill').fillna(method='bfill').fillna(0)
        sequence = pivoted.values[-input_len:]
        sequence = sequence[..., np.newaxis]
        

        timestamps = pivoted.index.values[-input_len:]
        time_tokens = self.prepare_temporal_tokens(timestamps)
        time_tokens = np.expand_dims(time_tokens, axis=1)
        time_tokens = np.repeat(time_tokens, self.config['num_nodes'], axis=1)
        

        sequence_tensor = torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32).to(self.device)
        time_tokens_tensor = torch.tensor(time_tokens[np.newaxis, ...], dtype=torch.long).to(self.device)
        

        with torch.no_grad():
            predictions = self.model(sequence_tensor, time_tokens_tensor)
            predictions = predictions.cpu().numpy()[0, 0]  
        

        results = []
        for h in range(hours):
            forecast_time = last_timestamp + timedelta(hours=h+1)
            for sensor_idx in range(self.config['num_nodes']):
                sensor_id = self.sensor_locations.iloc[sensor_idx]['sensor_id']
                results.append({
                    'sensor_id': sensor_id,
                    'timestamp': forecast_time,
                    'pm25_pred': float(predictions[sensor_idx, h])
                })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, df_results):

        mask = ~df_results['pm25_true'].isna()
        if mask.sum() > 0:
            y_true = df_results.loc[mask, 'pm25_true'].values
            y_pred = df_results.loc[mask, 'pm25_pred'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            
            return {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MSE': float(mse),
                'MAPE': float(mape)
            }
        else:
            return {'MAE': np.nan, 'RMSE': np.nan, 'MSE': np.nan, 'MAPE': np.nan}

def main():

    df = pd.read_pickle('../bihar_meteo_era5_may_jan_iterative_imputed.pkl')
    

    df = df[(df['timestamp'].dt.year == 2023) & (df['timestamp'].dt.month.isin([7]))].reset_index(drop=True)
    

    inference = STLLMInference(model_dir='outputs')
    
  
    print("Generating forecasts for entire dataset...")
    df_pred = inference.forecast(df, forecast_horizon=24, batch_size=16)
    

    metrics = inference.calculate_metrics(df_pred)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    

    df_pred.to_parquet('outputs/forecast_results.parquet', index=False)
    print("Forecast results saved to forecast_results.parquet")
    

    print("\nGenerating future forecasts...")
    df_future = inference.forecast_next_hours(df, hours=24)
    df_future.to_parquet('outputs/future_forecasts.parquet', index=False)
    print("Future forecasts saved to future_forecasts.parquet")
    

    plt.figure(figsize=(15, 8))
    

    avg_pred = df_pred.groupby('timestamp')['pm25_pred'].mean()
    avg_true = df_pred.groupby('timestamp')['pm25_true'].mean()
    
    plt.plot(avg_pred.index, avg_pred.values, label='Predicted', alpha=0.7)
    plt.plot(avg_true.index, avg_true.values, label='True', alpha=0.7)
    
    plt.title('PM2.5 Forecast vs True Values')
    plt.xlabel('Timestamp')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/forecast_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to forecast_plot.png")
    
    return df_pred, df_future, metrics

if __name__ == "__main__":
    df_pred, df_future, metrics = main()