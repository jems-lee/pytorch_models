import torch.nn as nn

class BPR(nn.Module):
    def __init__(self, n_users, n_items, n_factors, sparse=True):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
    
    def forward(self, user, pos_idx, neg_idx):
        pos_pred = (self.user_factors(user) * self.item_factors(pos_idx)).sum(-1).unsqueeze(-1) + self.item_biases(pos_idx)
        neg_pred = (self.user_factors(user) * self.item_factors(neg_idx)).sum(-1).unsqueeze(-1) + self.item_biases(neg_idx)
        return pos_pred, neg_pred
