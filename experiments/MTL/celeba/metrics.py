import torch


class CelebaMetrics:
    """Multi-task CelebA per-attribute F1 accumulator."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def incr(self, y_preds, ys):
        y_preds = torch.stack(y_preds).detach()
        ys = torch.stack(ys).detach()
        y_preds = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1, 2])
        self.fp += (y_preds * (1 - ys)).sum([1, 2])
        self.fn += ((1 - y_preds) * ys).sum([1, 2])

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"f1": f1.cpu().numpy()}
