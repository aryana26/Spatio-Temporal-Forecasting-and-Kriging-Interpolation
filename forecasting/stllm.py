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

# ---------------------------
# Utility functions
# ---------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371.0
    return c * r

def create_spatial_graph(sensor_locs, num_neighbors=3):
    """
    sensor_locs: DataFrame with columns ['sensor_id','latitude','longitude'] sorted by sensor_id
    returns edge_index (2,E) tensor and edge_attr (E,1) tensor
    """
    coords = sensor_locs[['latitude','longitude']].values
    n = len(sensor_locs)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_distance(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
            dist_matrix[i,j] = d
            dist_matrix[j,i] = d

    edges = []
    edge_attrs = []
    for i in range(n):
        nearest = np.argsort(dist_matrix[i])[1:num_neighbors+1]  # skip self
        for j in nearest:
            edges.append((i, j))
            edge_attrs.append(1.0 / (dist_matrix[i, j] + 1e-6))
            # add reverse edge for undirected behavior
            edges.append((j, i))
            edge_attrs.append(1.0 / (dist_matrix[i, j] + 1e-6))

    if edges:
        edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)  # (2, E)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, 1)  # (E,1)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,1), dtype=torch.float32)
    return edge_index, edge_attr

def cyclic_encode_hour(hour):
    rad = 2 * np.pi * hour / 24
    return np.sin(rad), np.cos(rad)
def cyclic_encode_month(month):
    rad = 2 * np.pi * month / 12
    return np.sin(rad), np.cos(rad)
def cyclic_encode_day_of_week(day):
    rad = 2 * np.pi * day / 7
    return np.sin(rad), np.cos(rad)

# ---------------------------
# Dataset
# ---------------------------
class SpatioTemporalDataset(Dataset):
    def __init__(self, df, sequence_length, forecast_horizon, feature_cols=None,scalers=None):
        """
        df must contain columns: ['sensor_id','timestamp','latitude','longitude','pm25','temp','rh', ...]
        returns items: X (S, L, F) and y (S, H)
        """
        self.df = df.copy()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # ensure sorted
        self.df = self.df.sort_values(['sensor_id','timestamp']).reset_index(drop=True)

        # sensors ordered
        self.sensor_ids = sorted(self.df['sensor_id'].unique())
        self.num_sensors = len(self.sensor_ids)

        # add cyclic time features
        self.df['hour_sin'], self.df['hour_cos'] = zip(*self.df['timestamp'].dt.hour.apply(cyclic_encode_hour))
        self.df['month_sin'], self.df['month_cos'] = zip(*self.df['timestamp'].dt.month.apply(cyclic_encode_month))
        self.df['dow_sin'], self.df['dow_cos'] = zip(*self.df['timestamp'].dt.dayofweek.apply(cyclic_encode_day_of_week))

        if feature_cols is None:
            self.feature_cols = [
                'pm25','temp','rh',
                'hour_sin','hour_cos',
                'month_sin','month_cos',
                'dow_sin','dow_cos',
                'latitude','longitude'
            ]
        else:
            self.feature_cols = feature_cols

        # Scale per sensor
        if scalers is None:
            self.scalers = {}
        # fit new scalers
        else: # used in inference
            self.scalers = scalers
        # use given scalers
        
        self.scaled_data = {}  # key: sensor_id -> numpy array (T, F)
        for sid in self.sensor_ids:
            sensor_df = self.df[self.df['sensor_id'] == sid].sort_values('timestamp')
            arr = sensor_df[self.feature_cols].values.astype(np.float32)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(arr)
            self.scaled_data[sid] = scaled
            self.scalers[sid] = scaler

        # compute max sequences safely (min length across sensors)
        minlen = min(len(self.scaled_data[sid]) for sid in self.sensor_ids)
        self.max_sequences = minlen - sequence_length - forecast_horizon + 1
        if self.max_sequences < 1:
            raise ValueError(f"Not enough length across all sensors. minlen={minlen}, seq_len={sequence_length}, horizon={forecast_horizon}")

    def __len__(self):
        return self.max_sequences

    def __getitem__(self, idx):
        Xs = []
        Ys = []
        for sid in self.sensor_ids:
            arr = self.scaled_data[sid]
            X = arr[idx:idx+self.sequence_length]   # (L, F)
            y = arr[idx+self.sequence_length: idx+self.sequence_length+self.forecast_horizon, 0]  # pm25 only
            Xs.append(X)
            Ys.append(y)
        Xs = np.stack(Xs, axis=0).astype(np.float32)  # (S, L, F)
        Ys = np.stack(Ys, axis=0).astype(np.float32)  # (S, H)
        return torch.from_numpy(Xs), torch.from_numpy(Ys)

    # def inverse_transform_pm25(self, sensor_id, data_np):
    #     dummy = np.zeros((len(data_np), len(self.feature_cols)), dtype=np.float32)
    #     dummy[:,0] = data_np
    #     return self.scalers[sensor_id].inverse_transform(dummy)[:,0]
def inverse_transform_pm25(scaler, sensor_id, data_np, num_features):
    """
    scaler: StandardScaler or dict {'mean','scale'}
    data_np: (N,) array of pm25 values in scaled space
    """
    if isinstance(scaler, dict):  # when loaded from checkpoint
        mean = scaler['mean']
        scale = scaler['scale']
    else:  # StandardScaler instance
        mean = scaler.mean_
        scale = scaler.scale_

    dummy = np.zeros((len(data_np), num_features), dtype=np.float32)
    dummy[:, 0] = data_np
    unscaled = dummy * scale + mean
    return unscaled[:, 0]


class STLLMWithGNN(nn.Module):
    """
    GNN -> projection -> fuse(raw_features_proj + gnn_proj) -> LLM(inputs_embeds) -> output head

    This version **adds** a feat_projection that projects the raw input features (F)
    to d_model and **adds** it to the GNN projection (also projected to d_model).
    Everything else (names, forward signature) is kept the same so it is a drop-in.
    """
    def __init__(self, input_dim, gnn_hidden_dim, gnn_heads, gnn_layers, d_model,
                 forecast_horizon, num_sensors, use_checkpoint=False):
        super().__init__()
        self.input_dim = input_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_heads = gnn_heads
        self.gnn_layers = gnn_layers
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.num_sensors = num_sensors
        self.use_checkpoint = use_checkpoint

        # GNN stack (unchanged)
        self.gnns = nn.ModuleList()
        self.gnns.append(GATv2Conv(self.input_dim, gnn_hidden_dim, heads=gnn_heads, concat=True, edge_dim=1))
        for _ in range(gnn_layers - 1):
            self.gnns.append(
                GATv2Conv(gnn_hidden_dim * gnn_heads, gnn_hidden_dim, heads=gnn_heads, concat=True, edge_dim=1)
            )
        self.gnn_final_dim = gnn_hidden_dim * gnn_heads  # Dg

        # projection of GNN output to transformer dim (unchanged)
        self.input_projection = nn.Linear(self.gnn_final_dim, d_model)

        # NEW: project raw features (F) -> d_model so we can add with GNN projection
        # this is the minimal additive fusion.
        self.feat_projection = nn.Linear(self.input_dim, d_model)

        # placeholder for transformer; set via set_transformer()
        self.transformer = None

        # small head to map transformer last hidden state to forecast horizon
        self.output_head = nn.Linear(d_model, forecast_horizon)

    def set_transformer(self, transformer_model):
        # transformer_model must accept inputs_embeds and return last_hidden_state
        self.transformer = transformer_model

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: (B, S, L, F)
        edge_index: (2, E)
        edge_attr: (E,1)
        returns: (B, S, H)
        """
        B, S, L, F = x.shape
        device = x.device

        # Keep original 'x' available for feat_projection (shape: B,S,L,F)
        # Flatten/reshape to nodes for GNN processing across B*L graphs:
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * L, S, F)  # (B*L, S, F)
        x_nodes = x_flat.view(B * L * S, F)  # (B*L*S, F)  -- nodes as batch for message passing

        # Repeat edge_index for the B*L graphs (create a big graph)
        if edge_index.numel() > 0:
            # offsets for each graph copy
            offsets = (torch.arange(B * L, device=device) * S).repeat_interleave(edge_index.shape[1])
            big_edge_index = edge_index.repeat(1, B * L) + offsets[None, :]
            big_edge_attr = edge_attr.repeat(B * L, 1) if (edge_attr is not None) else None
        else:
            big_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            big_edge_attr = None

        # Run GNN stack on the big graph (same as before)
        out = x_nodes  # (B*L*S, F)
        for layer in self.gnns:
            if self.use_checkpoint:
                out = checkpoint(layer, out, big_edge_index, big_edge_attr)
            else:
                out = layer(out, big_edge_index, big_edge_attr)
            out = torch.relu(out)

        # Reshape back: (B*L*S, Dg) -> (B, L, S, Dg) -> permute -> (B, S, L, Dg)
        out = out.view(B, L, S, -1).permute(0, 2, 1, 3).contiguous()  # (B, S, L, Dg)

        # Project GNN output to d_model
        gnn_proj = self.input_projection(out)  # (B, S, L, d_model)

        # Project raw input features to d_model (NEW)
        # x has shape (B, S, L, F) -> feat_proj: (B, S, L, d_model)
        feat_proj = self.feat_projection(x)  # (B, S, L, d_model)

        # Additive fusion: elementwise sum (same shapes)
        fused = gnn_proj + feat_proj  # (B, S, L, d_model)

        # Prepare for transformer: merge B*S as batch dimension
        BS = B * S
        projected_flat = fused.view(BS, L, -1)  # (B*S, L, d_model)

        if self.transformer is None:
            raise RuntimeError("Transformer not set. Call set_transformer(transformer_model) before training/inference.")

        # Single call to transformer for all sensor series in batch
        transformer_out = self.transformer(inputs_embeds=projected_flat).last_hidden_state  # (BS, L, d_model)
        last_hidden = transformer_out[:, -1, :].view(B, S, -1)  # (B, S, d_model)

        return self.output_head(last_hidden)  # (B, S, H)
