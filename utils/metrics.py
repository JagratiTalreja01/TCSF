"""
metrics.py

Evaluation metrics for flood segmentation.
"""

import torch


class SegmentationMetrics:
    """
    Computes common binary segmentation metrics.

    Metrics:
        IoU
        Dice
        Precision
        Recall
        F1-score
        Pixel Accuracy
    """

    def __init__(self, threshold=0.5, eps=1e-6):
        self.threshold = threshold
        self.eps = eps

    def _binarize(self, logits):
        probs = torch.sigmoid(logits)
        return (probs > self.threshold).float()

    def _confusion_matrix(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        tp = (preds * targets).sum()

        fp = (preds * (1 - targets)).sum()

        fn = ((1 - preds) * targets).sum()

        tn = ((1 - preds) * (1 - targets)).sum()

        return tp, fp, fn, tn

    def iou(self, logits, targets):
        preds = self._binarize(logits)

        tp, fp, fn, _ = self._confusion_matrix(preds, targets)

        return (tp + self.eps) / (tp + fp + fn + self.eps)

    def dice(self, logits, targets):
        preds = self._binarize(logits)

        tp, fp, fn, _ = self._confusion_matrix(preds, targets)

        return (2 * tp + self.eps) / (
            2 * tp + fp + fn + self.eps
        )

    def precision(self, logits, targets):
        preds = self._binarize(logits)

        tp, fp, _, _ = self._confusion_matrix(preds, targets)

        return (tp + self.eps) / (tp + fp + self.eps)

    def recall(self, logits, targets):
        preds = self._binarize(logits)

        tp, _, fn, _ = self._confusion_matrix(preds, targets)

        return (tp + self.eps) / (tp + fn + self.eps)

    def f1(self, logits, targets):
        precision = self.precision(logits, targets)
        recall = self.recall(logits, targets)

        return (
            2 * precision * recall
        ) / (precision + recall + self.eps)

    def pixel_accuracy(self, logits, targets):
        preds = self._binarize(logits)

        preds = preds.view(-1)
        targets = targets.view(-1)

        correct = (preds == targets).float().sum()

        return correct / (targets.numel() + self.eps)

    def evaluate(self, logits, targets):
        """
        Returns a dictionary of all metrics.
        """

        return {
            "IoU": self.iou(logits, targets).item(),
            "Dice": self.dice(logits, targets).item(),
            "Precision": self.precision(logits, targets).item(),
            "Recall": self.recall(logits, targets).item(),
            "F1": self.f1(logits, targets).item(),
            "PixelAcc": self.pixel_accuracy(logits, targets).item(),
        }