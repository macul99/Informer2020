import torch
import torch.nn as nn
#import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    '''
    Encode position index to d_model dimension feature
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # buffer placeholder
        pe = torch.zeros((max_len, d_model), requires_grad=False).float()

        position = torch.arange(max_len).unsqueeze(-1).float() # (max_len, 1)
        div_term = (torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model)).exp() # = 1/(10000)^(i/d_model)

        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model/2), sth like sin(nx) where n is the position index
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.register_buffer('pe', pe) # add pe as the state of the class instead of parameter

    def forward(self, x):
        return self.pe[:, :x.size(1)] # (1, actual_len, d_model)

class TokenEmbedding(nn.Module):
    '''
    Use Conv1D to embed c_in dimension to d_model dimention feature.
    Initialize weights using kaiming_normal()
    '''
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        v1,v2 = torch.__version__.split('.')[0:2]
        if (int(v1)==1 and int(v2)>=5) or int(v1)>1:
            padding = 1 
        else:
            padding = 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, 
                                   out_channels=d_model, 
                                   kernel_size=3, 
                                   padding=padding, 
                                   padding_mode='circular')

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, x):
        # x shape (B,L,D)
        # permute and transpose are similar, transpose can only swith two dims
        x = self.tokenConv(x.permute(0,2,1)).transpose(1,2) # permute to make D into channel dim for Conv
        return x

class FixedEmbedding(nn.Module):
    '''
    Fix nn.Embedding weights which are initialized based on Positional Embedding
    '''
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros((c_in, d_model), requires_grad=False).float()

        # Positional Embedding
        position = torch.arange(c_in).unsqueeze(-1).float()
        div_term = (torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)).exp()

        w[:,0::2] = torch.sin(position * div_term)
        w[:,1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model) # Embedding is just a lookup table
        self.emb.weight = nn.Parameter(w, requires_grad=False) # Fixed embedding, no need to do back propagation

    def forward(self, x):
        return self.emb(x).detach() # detach to make the output a leave node since no back propagation required

class TemporalEmbedding(nn.Module):
    '''
    Encode temporal info based on FixedEmbedding or normal Embedding layer

    Order of temporal info is [month, day, weekday, hour, minute(optional)]
    '''
    def __init__(self, d_model, embed_type='fixed', freq='h', minute_size=4):
        # freq: h or t
        super(TemporalEmbedding, self).__init__()

        #minute_size = 4 # 15min interval
        hour_size = 24
        weekday_size = 7 
        day_size = 32
        month_size = 13

        self.month_idx = 0
        self.day_idx = 1
        self.weekday_idx = 2
        self.hour_idx = 3
        self.minute_idx = 4

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding

        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,self.minute_idx]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,self.hour_idx])
        weekday_x = self.weekday_embed(x[:,:,self.weekday_idx])
        day_x = self.day_embed(x[:,:,self.day_idx])
        month_x = self.month_embed(x[:,:,self.month_idx])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    '''
    Use nn.Linear to do embedding.

    freq refer to utils.timefeatues.py

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    '''
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3} # refer to utils.timefeatues.py
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)