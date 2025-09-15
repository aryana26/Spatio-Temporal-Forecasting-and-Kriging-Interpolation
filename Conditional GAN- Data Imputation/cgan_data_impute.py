import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# Step 1. Preprocessing
# =========================
def preprocess_dataframe(df):
    # Assign sensor_id
    unique_sensors = df[["latitude","longitude"]].drop_duplicates().reset_index(drop=True)
    unique_sensors["sensor_id"] = range(len(unique_sensors))
    df = df.merge(unique_sensors, on=["latitude","longitude"], how="left")
    
    
    # BallTree for neighbors
    coords = np.radians(unique_sensors[["latitude","longitude"]].values)
    tree = BallTree(coords, metric="haversine")
    
    # Get nearest 3 neighbors (exclude self)
    dists, idxs = tree.query(coords, k=4)
    neighbor_idxs = idxs[:, 1:4]  # 3 nearest neighbors
    
    # Ensure full grid
    full_index = pd.MultiIndex.from_product(
        [df["sensor_id"].unique(), df["timestamp"].unique()],
        names=["sensor_id", "timestamp"]
    )
    df_full = df.set_index(["sensor_id","timestamp"]).reindex(full_index).reset_index()
    df_full = df_full.merge(unique_sensors, on="sensor_id", how="left")
    
    del df_full['latitude_x']
    del df_full['longitude_x']
    print(df_full.isna().sum())
    # Create neighbor mapping
    neighbor_mapping = {}
    mean_df=pd.DataFrame()
    for i, sensor_id in enumerate(unique_sensors["sensor_id"]):
        print(i,end='\r')
        neighbor_ids = unique_sensors.iloc[neighbor_idxs[i]]["sensor_id"].values
        neighbor_mapping[sensor_id] = neighbor_ids
        # times_when_sensor_missing=df_full[(df_full.sensor_id==sensor_id)]['timestamp']
        missing=df_full[(df_full.sensor_id.isin(neighbor_ids))].groupby('timestamp')['pm25'].mean().reset_index()
        missing.columns=['timestamp','neighbor_pm25']
        missing['sensor_id']=sensor_id
        mean_df=pd.concat([mean_df,missing])
    print(mean_df.isna().sum())
    print('befire merge',df_full.shape)
    df_full2=pd.merge(df_full,mean_df,on=['timestamp','sensor_id'])
    print('afetr merge',df_full2.shape)
    df_full2.neighbor_pm25=np.where(df_full2.neighbor_pm25.isna(),df_full2.pm25,df_full2.neighbor_pm25)
    df_full2['all_area_mean']=df_full2.groupby('timestamp')['pm25'].transform('mean')
    df_full2.neighbor_pm25=np.where(df_full2.neighbor_pm25.isna(),df_full2.all_area_mean,df_full2.neighbor_pm25)
    del df_full2['all_area_mean']
    df_full2.columns=['sensor_id', 'timestamp', 'pm25', 'temp', 'rh', 'latitude',
           'longitude', 'neighbor_pm25']
    print(df_full2.isna().sum())
    
    return df_full2

# =========================
# Step 2. Dataset Class
# =========================
class ImputationDataset(Dataset):
    def __init__(self, df):
        """
        df has [latitude, longitude, timestamp, neighbor_pm25, pm25, temp, rh]
        Missing targets will be NaN (to impute later).
        """
        # Extract month and hour
        df = df.copy()
        df["month"] = df["timestamp"].dt.month
        df["hour"] = df["timestamp"].dt.hour
        
        # Cyclic encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        self.df = df
        
        # Condition array
        self.cond = df[[
            "latitude", "longitude",
            "neighbor_pm25",
            "month_sin", "month_cos",
            "hour_sin", "hour_cos"
        ]].values.astype("float32")
        self.target = df[["pm25","temp","rh"]].values.astype("float32")

        # Store mask for missing values
        self.mask = ~np.isnan(self.target)
        
        # Normalize target - only use non-NaN values for statistics
        valid_target = self.target[self.mask]
        self.mean = np.nanmean(self.target, axis=0, keepdims=True)
        self.std = np.nanstd(self.target, axis=0, keepdims=True) + 1e-6
        self.target_norm = (self.target - self.mean) / self.std

        # Replace NaN with 0 in training
        self.target_norm = np.nan_to_num(self.target_norm)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.cond[idx], dtype=torch.float32),
            torch.tensor(self.target_norm[idx], dtype=torch.float32),
            torch.tensor(self.mask[idx], dtype=torch.bool)  # Return mask
        )


# =========================
# Step 3. Generator and Discriminator
# =========================
class Generator(nn.Module):
    def __init__(self, cond_dim, target_dim, z_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, cond, z):
        x = torch.cat([cond, z], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, cond_dim, target_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, cond, target):
        x = torch.cat([cond, target], dim=1)
        return self.net(x)


# =========================
# Step 4. Training Function
# =========================
def train_cgan(
    dataloader,
    cond_dim,
    target_dim,
    device="cuda",
    num_epochs=50,
    z_dim=16,
    hidden_dim=128,
    lr=1e-3,
    lambda_recon=10.0,
):
    G = Generator(cond_dim, target_dim, z_dim, hidden_dim).to(device)
    D = Discriminator(cond_dim, target_dim, hidden_dim).to(device)

    # Use larger batch sizes for GPU
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    opt_G = optim.Adam(G.parameters(), lr=lr, weight_decay=1e-5)
    opt_D = optim.Adam(D.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {
        'D_loss': [], 'G_loss': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': []
    }
    best_mae = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_mae, epoch_mse, epoch_rmse, epoch_mape = [], [], [], []
        epoch_D_loss, epoch_G_loss = [], []

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (cond, real, mask) in enumerate(loop):
            cond, real, mask = cond.to(device), real.to(device), mask.to(device)
            bs = cond.size(0)

            # -------------------
            # Train Discriminator
            # -------------------
            z = torch.randn(bs, z_dim, device=device)  # Create on GPU directly
            fake = G(cond, z).detach()

            real_out = D(cond, real)
            fake_out = D(cond, fake)

            loss_D = bce(real_out, torch.ones_like(real_out)) + \
                     bce(fake_out, torch.zeros_like(fake_out))

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            epoch_D_loss.append(loss_D.item())

            # -------------------
            # Train Generator
            # -------------------
            z = torch.randn(bs, z_dim, device=device)  # Create on GPU
            fake = G(cond, z)

            fake_out = D(cond, fake)
            adv_loss = bce(fake_out, torch.ones_like(fake_out))
            
            # Only compute reconstruction loss on non-missing values
            recon_loss = mse(fake[mask], real[mask]) if mask.any() else 0

            loss_G = adv_loss + lambda_recon * recon_loss

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            epoch_G_loss.append(loss_G.item())

            # -------------------
            # Metrics - Calculate less frequently to reduce CPU transfers
            # -------------------
            if batch_idx % 10 == 0 and mask.any():  # Every 10 batches
                mae_val = mean_absolute_error(real[mask].cpu().numpy(), fake[mask].detach().cpu().numpy())
                mse_val = mean_squared_error(real[mask].cpu().numpy(), fake[mask].detach().cpu().numpy())
                rmse_val = np.sqrt(mse_val)
                mape_val = np.mean(np.abs((real[mask].cpu().numpy() - fake[mask].detach().cpu().numpy()) / 
                                        (real[mask].cpu().numpy() + 1e-6))) * 100
                
                epoch_mae.append(mae_val)
                epoch_mse.append(mse_val)
                epoch_rmse.append(rmse_val)
                epoch_mape.append(mape_val)

            # Update progress bar less frequently
            if batch_idx % 5 == 0:
                loop.set_postfix(
                    D_loss=np.mean(epoch_D_loss[-10:]),
                    G_loss=np.mean(epoch_G_loss[-10:]),
                    MAE=np.mean(epoch_mae[-5:]) if epoch_mae else 0,
                )
        current_mae = np.mean(epoch_mae) if epoch_mae else 0
        # Save best model based on MAE (but continue training)
        if current_mae < best_mae:
            best_mae = current_mae
            patience_counter = 0
            # Save best model
            torch.save(G.state_dict(), 'best_generator.pth')
        else:
            patience_counter += 1
        # Monitor training progress

        if epoch % 5 == 0:  # Check every 5 epochs
            # Generate sample images to visually inspect quality
            with torch.no_grad():
                sample_z = torch.randn(16, z_dim, device=device)
                sample_cond = cond[:16]  # First 16 conditions
                samples = G(sample_cond, sample_z).cpu().numpy()
                # Visualize or save samples to check quality
            
            # Check for mode collapse (all samples look similar)
            sample_std = samples.std(axis=0).mean()
            if sample_std < 0.1:  # Very low diversity
                print("Warning: Possible mode collapse detected!")
            

        # Store history
        history['D_loss'].append(np.mean(epoch_D_loss))
        history['G_loss'].append(np.mean(epoch_G_loss))
        history['MAE'].append(np.mean(epoch_mae) if epoch_mae else 0)
        history['MSE'].append(np.mean(epoch_mse) if epoch_mse else 0)
        history['RMSE'].append(np.mean(epoch_rmse) if epoch_rmse else 0)
        history['MAPE'].append(np.mean(epoch_mape) if epoch_mape else 0)

        print(
            f"Epoch {epoch+1}: "
            f"D_loss={history['D_loss'][-1]:.4f}, "
            f"G_loss={history['G_loss'][-1]:.4f}, "
            f"MAE={history['MAE'][-1]:.4f}"
        )

    return G, D, history


# =========================
# Step 5. Imputation Function
# =========================
def impute_missing(G, dataset, device="cpu"):
    df = dataset.df.copy()
    cond = torch.tensor(dataset.cond, dtype=torch.float32).to(device)
    z = torch.randn(len(cond), 16).to(device)  # 16 is the noise_dimension
    
    with torch.no_grad():
        pred_norm = G(cond, z).cpu().numpy()
    
    pred = pred_norm * dataset.std + dataset.mean
    
    # Only replace missing values
    for i, col in enumerate(["pm25", "temp", "rh"]):
        mask = df[col].isna()
        df.loc[mask, col] = pred[mask, i]
    
    return df


# =========================
# Step 6. Save Model and Metadata
# =========================
def save_imputation_model(G, dataset, filepath="imputation_model"):
    """
    Save the trained model and all necessary metadata for future imputation
    """
    # Create directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    
    # Save the generator model
    torch.save(G.state_dict(), os.path.join(filepath, "generator.pth"))
    
    # Save dataset normalization parameters
    norm_params = {
        'mean': dataset.mean.tolist(),
        'std': dataset.std.tolist(),
        'cond_columns': list(dataset.df[["latitude", "longitude", "neighbor_pm25",
                                       "month_sin", "month_cos", "hour_sin", "hour_cos"]].columns)
    }
    
    with open(os.path.join(filepath, "norm_params.json"), 'w') as f:
        json.dump(norm_params, f)
    
    # Save model architecture parameters
    model_config = {
        'cond_dim': dataset.cond.shape[1],
        'target_dim': dataset.target.shape[1],
        'z_dim': 16,
        'hidden_dim': 128
    }
    
    with open(os.path.join(filepath, "model_config.json"), 'w') as f:
        json.dump(model_config, f)
    
    print(f"Model saved successfully in {filepath}/")



# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_parquet("../main_data_cleaned_aryan2.parquet")
    df = df[df.timestamp.dt.year == 2024][["pm25", "latitude", "longitude", "temp", "rh", "timestamp"]].reset_index(drop=True)
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    print("Filtered shape:", df.shape)
    
    df_full = preprocess_dataframe(df)
    print("Full grid shape:", df_full.shape)
    print("Missing values:", df_full[["pm25", "temp", "rh"]].isna().sum())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = ImputationDataset(df_full)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    cond_dim = dataset.cond.shape[1]   # latitude, longitude, neighbor_pm25, cyclic encodings
    out_dim = dataset.target.shape[1]  # should be 3 (pm25, temp, rh)

    print(f"Condition dimension: {cond_dim}, Output dimension: {out_dim}")

    # Train the model
    G, D, history = train_cgan(
        dataloader=loader, 
        cond_dim=cond_dim,
        target_dim=out_dim, 
        num_epochs=50, 
        device=device,
        hidden_dim=128,
        z_dim=16,
        lr=1e-3,
        lambda_recon=10.0
    )

    # Impute missing values
    df_imputed = impute_missing(G, dataset, device=device)
    df_imputed.to_parquet('df_imputed_cgan.parquet')
    save_imputation_model(G, dataset)
    print("Imputation done. Final shape:", df_imputed.shape)
    print("Missing values after imputation:", df_imputed[["pm25", "temp", "rh"]].isna().sum())