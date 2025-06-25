# src/main/python/relcore/pretraininglosses/my_weighted_infonce_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyWeightedInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        # Hardest Positive (level 1) should have the highest weight.
        # Hardest Negative (level 6) should have the highest weight.
        # Easiest Negative (level 4) should have the lowest weight.
        self.pos_weights = torch.tensor([0.0, 1.0, 0.44, 0.11, 0.0, 0.0, 0.0], dtype=torch.float) # [dummy, L1, L2, L3, L4, L5, L6]
        self.neg_weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.11, 0.44, 1.0], dtype=torch.float) # [dummy, L1, L2, L3, L4, L5, L6]

    def forward(self, anchor_emb, candidate_embs, levels):
        device = anchor_emb.device
        anchor_emb = F.normalize(anchor_emb, p=2, dim=0)
        candidate_embs = F.normalize(candidate_embs, p=2, dim=1)
        sim_scores = torch.matmul(anchor_emb.unsqueeze(0), candidate_embs.T) / self.temperature
        sim_scores = sim_scores.squeeze(0)
        w_pos = self.pos_weights[levels.long()].to(device)
        w_neg = self.neg_weights[levels.long()].to(device)
        log_w_pos = torch.log(w_pos.clamp(min=1e-9))
        log_w_neg = torch.log(w_neg.clamp(min=1e-9))
        numerator = torch.logsumexp(sim_scores + log_w_pos, dim=0)
        denominator = torch.logsumexp(sim_scores + log_w_neg, dim=0)
        loss = -(numerator - denominator)
        return loss