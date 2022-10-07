import torch
import torch.nn.functional as F

def random_masking(x, masking_percentage):
    return F.dropout(x, p=masking_percentage, training=True, inplace=False)

def channel_wise_masking(x, masking_percentage):
    return F.dropout2d(x.unsqueeze(-1), p=masking_percentage, training=True, inplace=False).squeeze()

def temporal_masking(x, masking_percentage):
    masked_share = 0.
    masked_indices = torch.zeros(size=(x.shape[-1],), dtype=torch.bool)
    while masked_share < masking_percentage:
        first_unmasked_idx = min((masked_indices == 0).nonzero(as_tuple=True)[0])
        last_unmasked_idx = max((masked_indices == 0).nonzero(as_tuple=True)[0])
        start_idx = torch.randint(first_unmasked_idx, last_unmasked_idx, size=(1,))[0]
        masked_length = torch.randint(1, int(x.shape[-1]*masking_percentage), size=(1,))[0]
        masked_direction = torch.randint(0,1,(1,)) # 1: right, 0: left
        if masked_direction:
            masked_indices[start_idx:min(start_idx+masked_length, x.shape[-1])] = 1
        else:
            masked_indices[max(0,start_idx-masked_length):start_idx] = 1
        masked_share = sum(masked_indices) / len(masked_indices)
    x_masked = torch.clone(x.detach())
    x_masked[:,:,masked_indices] = 0
    return x_masked