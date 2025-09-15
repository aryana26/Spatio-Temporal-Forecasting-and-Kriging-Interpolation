# model/st_llm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time_day = nn.Embedding(time, features)
        self.time_week = nn.Embedding(7, features)
        
        nn.init.xavier_uniform_(self.time_day.weight)
        nn.init.xavier_uniform_(self.time_week.weight)

    def forward(self, x):

        day_emb = x[..., 0] 
        week_emb = x[..., 1]
        
        time_day = self.time_day(day_emb.long()) 
        time_week = self.time_week(week_emb.long()) 
        return time_day + time_week

class ST_LLM(nn.Module):
    def __init__(self, input_dim=1, channels=64, num_nodes=511, input_len=24,
                 output_len=24, llm_layer=6, U=1, device="cuda:0"):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.device = device

        time = 48
        gpt_channel = 256
        self.to_gpt_channel = 768


        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)
        

        self.start_conv = nn.Conv2d(input_dim * input_len, gpt_channel, kernel_size=(1, 1))


        self.gpt = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
        self.gpt.h = self.gpt.h[:llm_layer]
        

        for layer_ix, layer in enumerate(self.gpt.h):
            for param in layer.parameters():
                if layer_ix < llm_layer - U:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


        self.feature_fusion = nn.Conv2d(gpt_channel * 3, self.to_gpt_channel, kernel_size=(1, 1))
        self.regression_layer = nn.Conv2d(self.to_gpt_channel, output_len, kernel_size=(1, 1))

    def forward(self, x, time_tokens):
        batch_size = x.shape[0]
        
        x = x.permute(0, 3, 1, 2)
        B, C, T, N = x.shape
        x = x.reshape(B, C * T, N, 1)
        x = self.start_conv(x) 
        tem_emb = self.Temb(time_tokens) 
        tem_emb = tem_emb.permute(0, 3, 2, 1) 
        tem_emb = torch.mean(tem_emb, dim=-1, keepdim=True)

        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  
        node_emb = node_emb.permute(0, 2, 1).unsqueeze(-1)  
        data = torch.cat([x, tem_emb, node_emb], dim=1)  
        data = self.feature_fusion(data)  

        data = data.squeeze(-1).permute(0, 2, 1) 
        data = data.reshape(B * N, 1, self.to_gpt_channel)        
        out = self.gpt(inputs_embeds=data).last_hidden_state 
        out = out.reshape(B, N, 1, -1) 
        out = out.permute(0, 3, 1, 2) 
        pred = self.regression_layer(out)

        return pred.permute(0, 3, 2, 1)

    def param_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params