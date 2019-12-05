import torch
import torch.nn as nn
import torch.nn.functional as F


class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']
        
        self.embedding_item_idx = torch.nn.Embedding(
            num_items=self.num_item,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_idx):
        item_emb = self.embedding_item_idx(item_idx)
        return item_emb


class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        self.num_user = config['num_user']
        self.embedding_dim = config['embedding_dim']
        
        self.embedding_user_idx = torch.nn.Embedding(
            num_users=self.num_user,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_idx):
        user_emb = self.embedding_user_idx(user_idx)
        return user_emb
