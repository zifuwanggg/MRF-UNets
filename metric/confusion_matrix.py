import torch

from metric import metric


class ConfusionMatrix(metric.Metric):
    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64).to(device)


    def add(self, pred, label):
        assert pred.shape == label.shape, f"pred.shape: {pred.shape} != label.shape: {label.shape}"
        assert (pred.max() < self.num_classes) and (pred.min() >= 0), f"pred.max: {pred.max()}, pred.min: {pred.min()}"
        assert (label.max() < self.num_classes) and (label.min() >= 0), f"label.max: {label.max()}, label.min: {label.min()}"

        x = pred + self.num_classes * label
        bincount = torch.bincount(x.long(), minlength=self.num_classes**2)
        conf = bincount.reshape((self.num_classes, self.num_classes))
        self.matrix += conf


    def value(self):
        return self.matrix