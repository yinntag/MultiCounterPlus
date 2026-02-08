import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        加权多分类交叉熵损失函数
        :param alpha: 加权损失的权重
        :param beta: 原始交叉熵损失的权重
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, true_labels, true_periods, pred_periods):
        """
        :param pred: 模型预测的类别概率 [batch_size, num_classes]
        :param true_labels: 真实类别标签 [batch_size]
        :param true_periods: 真实周期长度 [batch_size]
        :param pred_periods: 模型预测周期长度 [batch_size]
        :return: 总损失
        """
        # 1. 原始交叉熵损失
        original_loss = F.cross_entropy(pred, true_labels, reduction='mean')

        # 2. 周期误差计算
        period_error = torch.abs(1.0 / true_periods - 1.0 / pred_periods)  # [batch_size]

        # 3. 计算权重
        weights = period_error / period_error.sum()  # 归一化 [batch_size]

        # 4. 加权交叉熵损失
        log_probs = F.log_softmax(pred, dim=1)  # 转为 log 概率
        weighted_loss = -torch.sum(weights * log_probs[range(len(true_labels)), true_labels]) / len(true_labels)

        # 5. 总损失
        total_loss = self.alpha * weighted_loss + self.beta * original_loss
        return total_loss


# 示例训练流程
if __name__ == "__main__":
    # 假设一个批次的输入
    batch_size = 3
    num_classes = 5
    pred = torch.randn(batch_size, num_classes)  # [batch_size, num_classes]
    true_labels = torch.tensor([1, 3, 4])  # [batch_size]
    true_periods = torch.tensor([10.0, 30.0, 50.0])  # 真实周期长度 [batch_size]
    pred_periods = torch.tensor([12.0, 28.0, 55.0])  # 模型预测周期长度 [batch_size]

    # 初始化加权损失函数
    loss_fn = WeightedCrossEntropyLoss(alpha=1.0, beta=1.0)

    # 计算损失
    loss = loss_fn(pred, true_labels, true_periods, pred_periods)
    print(f"总损失: {loss.item():.4f}")
