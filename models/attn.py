import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import numpy as np

from math import sqrt
from ..utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=None, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()

        self.mask_flag = mask_flag
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # Note: factor is used for compatibility of ProbAttention

    def forward(self, queries, keys, values, attn_mask=None):
        device = queries.device
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum('blhe,bshe->bhls', queries, keys) # Q*K' (le*es=ls)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=device) # mask shape [B, 1, L, L]
            scores.masked_fill_(attn_mask.mask, -np.Inf)

        # perform softmax on scores with scale, followed by dropout
        A = self.dropout(torch.softmax(scale*scores, dim=-1)) # A has the same shape as scores

        V = torch.einsum('bhls,bshd->blhd', A, values) # A*V (ls*sd=ld)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=None, output_attention=False):
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor # used to define sample size
        self.scale = scale
        # attention_dropout is for compatibility only, not used in this class
        self.output_attention = output_attention


    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: factor*ln(L_q)
        device = Q.device
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape
        # sample_k<=L_K
        # n_top<=L_Q

        # sample L_K for each L_Q
        K_expand = repeat(K, 'b h lk e -> b h lq lk e', lq=L_Q) # shape is [B,H,L_Q,L_K,E]
        index_sample = torch.randint(L_K, (L_Q, sample_k)).to(device)
        K_sample = K_expand[:,:,torch.arange(L_Q).unsqueeze(-1),index_sample,:] # shape is [B,H,L_Q,sample_k,E]
        Q_K_sample = torch.einsum('bhqe,bhqke->bhqk', Q, K_sample) # [B,H,L_Q,sample_k], (1,e)*(e,k)->(1,k)->(k,)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) # [0] of max gives values, (B,H,L_Q)
        M_top = M.topk(n_top, sorted=False)[1] # [1] of topk gives index, (B,H,n_top)

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:,None,None],
                     torch.arange(H)[None,:,None],
                     M_top,
                     :] # (B,H,n_top,E)
        Q_K = torch.einsum('bhqe,bhke->bhqk', Q_reduce, K) # (B,H,n_top,L_K)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            assert L_V==L_Q, 'L_V and L_Q must be the same with masking!'
            context = V.cumsum(dim=-2) # along L_V dim
        else:
            context = reduce(V, 'b h l d -> b h d', 'mean')
            context = repeat(context, 'b h d -> b h l d', l=L_Q).clone()

        return context # (B,H,L_V,D)

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask=None):
        device = V.device
        B, H, L_V, D = V.shape
        # context_in.shape (B,H,L_Q,D), L_V==L_Q if mask_flag is True
        _, _, n_top, _ = scores.shape # (B,H,n_top,L_V), L_V==L_K
        # index shape (B,H,n_top) 

        # prepare mask
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=device)
            scores.masked_fill_(attn_mask.mask, -np.Inf)

        # softmax on scores, no dropout needed since scores are already randomly sampled
        attn = torch.softmax(scores, dim=-1)

        A = torch.einsum('bhal,bhld->bhad', attn, V).type_as(context_in)
        context_in[torch.arange(B)[:,None,None],
                   torch.arange(H)[None,:,None],
                   index, 
                   :] = A
        if self.output_attention:
            attention = (torch.ones([B,H,L_V,L_V])/L_V).type_as(attn).to(attn.device)
            attention[torch.arange(B)[:,None,None],
                      torch.arange(H)[None,:,None],
                      index,
                      :] = attn
            return (context_in, attention)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1) # (B,H,L_Q,D)
        keys = keys.transpose(2,1) # (B,H,L_K,D)
        values = values.transpose(2,1) #(B,H,L_K,D)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u      = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = min(U_part, L_K)
        u = min(u, L_Q)
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) # scores_top has shape of (B,H,n_top,L_K)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q) #(B,H,L_Q,D), if mask_flag: L_Q == L_V
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        context = rearrange(context, 'B H L D -> B L H D')
        
        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    # wrapper for both of FullAttention and ProbAttention
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, queries, keys, values, attn_mask):
        '''
        B, L, _ = queries.shape # (B, L, D)
        _, S, _ = keys.shape # (B, S, D)
        H = self.n_heads
        '''
        
        queries = self.query_projection(queries)
        queries = rearrange(queries, 'B L (H D) -> B L H D') # (B,L,H,d_keys)
        keys = self.key_projection(keys)
        keys = rearrange(keys, 'B S (H D) -> B S H D') # (B,S,H,d_keys)
        values = self.value_projection(values)
        values = rearrange(values, 'B S (H D) -> B S H D') # (B,S,H,d_values)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ) # out shape is (B,L,H,D)
        out = rearrange(out, 'B L H D -> B L (H D)') # (B,L,d_values*H)

        return self.out_projection(out), attn # output is (B,L,d_model)
