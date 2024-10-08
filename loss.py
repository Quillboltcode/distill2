import torch
import torch.nn as nn
import torch.nn.functional as F

# other loss put into loss folder is better
# import logit_calibration

class CEDistill(nn.Module):
    """Cross entropy loss between two set of outputs"""
    def __init__(self, temperature):
        super(CEDistill, self).__init__()
        self.temperature = temperature
    
    def forward(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs/self.temperature, dim=1)
        softmax_targets = F.softmax(targets/self.temperature, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

class DistilKL(nn.Module):
    """Distil knowlegde in neural network"""
    def __init__(self,T:float):
        super(DistilKL, self).__init__()
        self.T = T
    def forward(self, y_s:torch.Tensor, y_t:torch.Tensor):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t)*self.T*self.T
        return loss
    
