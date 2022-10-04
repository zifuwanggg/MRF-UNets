import torch

from metric import metric
from metric.confusion_matrix import ConfusionMatrix


class IoUDice(metric.Metric):
    def __init__(self, num_classes, device, dataset, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.confusion_matrix = ConfusionMatrix(num_classes=self.num_classes, device=device)
        self.dataset = dataset
        self.ignore_index = ignore_index


    def add(self, pred, label): 
        not_ignore = label != self.ignore_index
        pred_filtered = pred[not_ignore]
        label_filtered = label[not_ignore]

        self.confusion_matrix.add(pred_filtered.reshape(-1), label_filtered.reshape(-1))


    def value(self):
        confusion_matrix = self.confusion_matrix.value()

        true_positive = torch.diag(confusion_matrix)
        false_positive = torch.sum(confusion_matrix, 0) - true_positive
        false_negative = torch.sum(confusion_matrix, 1) - true_positive

        total = true_positive + false_positive + false_negative
        
        if self.num_classes == 2 or self.dataset == 'chaos':
            total[0] = 0
            
        true_positive = true_positive[total!=0]
        false_positive = false_positive[total!=0]
        false_negative = false_negative[total!=0]

        iou = true_positive / (true_positive + false_positive + false_negative)
        dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
        
        return 100 * torch.mean(iou), 100 * torch.mean(dice)