import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

    
class JaccardLoss(_Loss):
    def __init__(self, eps=1e-7, ignore_index=None):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        
        
    def forward(self, logits, label):            
        classes = set(label.unique().tolist())

        if self.ignore_index in classes:
            classes.remove(self.ignore_index)
        
        if len(classes) < 1:
            return 0. * logits.sum()
        
        batch_size, num_classes, _, _ = logits.shape
            
        if num_classes == 1:
            prob = F.logsigmoid(logits).exp()
            
            prob = prob.view(-1)
            label = label.view(-1)    
            
            not_ignore = label != self.ignore_index 
            
            prob = prob[not_ignore]
            label = label[not_ignore]
            
            iou = self.compute_iou(prob, label)
            losses = 1.0 - iou
            
            return losses
        
        else:
            losses = []
            
            prob = logits.log_softmax(dim=1).exp()
        
            prob = prob.view(batch_size, num_classes, -1)
            label = label.view(batch_size, -1)    
            not_ignore = label != self.ignore_index 
                
            label = F.one_hot(label, num_classes + 1)  
            label = label.permute(0, 2, 1)
            
            for j in classes:
                prob_j = prob[:, j, :][not_ignore]
                label_j = label[:, j, :][not_ignore]
                iou = self.compute_iou(prob_j, label_j)
                losses.append(1.0 - iou)
            
            return sum(losses) / len(losses)
    
    
    def compute_iou(self, prob, label):
        intersection = torch.sum(prob * label)
        difference = torch.sum(torch.abs(prob - label))
        union = difference + intersection
        
        return intersection / union.clamp_min(self.eps)