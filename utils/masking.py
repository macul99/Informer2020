import torch

class TriangularCausalMask():
    def __init__(self, B, L, S=None, device="cpu"):
        S = S or L
        mask_shape = [B, 1, L, S] # most of time L==S

        # by default, torch.ones create a tensor with requires_grad=False
        self._mask = torch.ones(mask_shape).triu(1).bool().to(device) 

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        B1, H1, n_top = index.shape # index shape (B,H,n_top), n_top out of L (n_top<=L)
        B2, H2, n_top1, S = scores.shape # scores shape (B,H,n_top,S), see attn.py
        assert (B==B1==B2) and (H==H1==H2) and (n_top==n_top1), 'Input size does not match!'
        mask_shape = [B, H, L, S] 
        _mask = torch.ones(mask_shape).triu(1).bool() # (B, H, L, S)
        indicator = _mask[torch.arange(B)[:, None, None],
                          torch.arange(H)[None, :, None],
                          index, :]
        # indicator shape is (B,H,n_top,S), (B,1,1),(1,H,1) and (B,H,n_top) broadcast to (B,H,n_top)
        self._mask = indicator.view(scores.shape).to(device) # will check whether the shape is correct or not
    
    @property
    def mask(self):
        return self._mask