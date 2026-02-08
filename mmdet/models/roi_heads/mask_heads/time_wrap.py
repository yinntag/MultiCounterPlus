import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiRateFeatureEnhancement(nn.Module):
    """
    多速率特征增强模块（MRFE）：不同感受野的时间卷积路径。
    """
    def __init__(self, embed_dim, kernel_sizes=(3, 5, 7), num_filters=128):
        super(MultiRateFeatureEnhancement, self).__init__()
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes

        # 多种感受野的卷积路径
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.attention = nn.Linear(len(kernel_sizes) * num_filters, embed_dim)

    def forward(self, x):
        """
        输入: x -> [batch, seq_len, embed_dim]
        输出: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # 将维度调整为 [batch, embed_dim, seq_len] 以适配 Conv1d
        x = x.permute(0, 2, 1)

        # 不同感受野的卷积
        multi_scale_features = [conv(x) for conv in self.convs]

        # 将多尺度特征拼接在 channel 维度
        concat_features = torch.cat(multi_scale_features, dim=1)  # [batch, num_filters * len(kernel_sizes), seq_len]

        # 恢复时间维度 [batch, seq_len, num_filters * len(kernel_sizes)]
        concat_features = concat_features.permute(0, 2, 1)

        # 自注意力权重计算并映射回原始 embed_dim
        enhanced_features = self.attention(concat_features)  # [batch, seq_len, embed_dim]
        return F.relu(enhanced_features)


class PeriodicAdaptationModule(nn.Module):
    """
    周期适配模块（PAM）：动态调整对象的周期性特征。
    """
    def __init__(self, embed_dim):
        super(PeriodicAdaptationModule, self).__init__()
        self.scaling_factor = nn.Linear(embed_dim, 1)  # 每个时间步生成一个动态调整因子

    def forward(self, x):
        """
        输入: x -> [batch, seq_len, embed_dim]
        输出: 调整后的周期特征 [batch, seq_len, embed_dim]
        """
        # 生成动态调整因子 [batch, seq_len, 1]
        scale = torch.sigmoid(self.scaling_factor(x))

        # 使用动态调整因子重新加权特征
        adjusted_features = x * scale
        return adjusted_features


class AdaptiveSamplingModule(nn.Module):
    """
    整体模块：整合 MRFE 和 PAM。
    """
    def __init__(self, embed_dim, kernel_sizes=(3, 5, 7), num_filters=128):
        super(AdaptiveSamplingModule, self).__init__()
        self.mrfe = MultiRateFeatureEnhancement(embed_dim, kernel_sizes, num_filters)
        self.pam = PeriodicAdaptationModule(embed_dim)

    def forward(self, x):
        """
        输入: x -> [batch, seq_len, embed_dim]
        输出: 优化后的特征 [batch, seq_len, embed_dim]
        """

        num_proposals, _ = x.shape  #
        # x = self.tsm_label(xx)
        x = x.reshape(-1, 64, 256)
        batch_size, seq_len, _ = x.shape  #

        # 多速率特征增强
        enhanced_features = self.mrfe(x)

        # 周期特征动态适配
        adjusted_features = self.pam(enhanced_features)

        return adjusted_features


# # 测试代码
# if __name__ == "__main__":
#     # 输入特征: [batch, seq_len, embed_dim]
#     x = torch.randn(192, 256)  # 模拟输入特征
#
#     # 定义模块
#     model = AdaptiveSamplingModule(embed_dim=256, kernel_sizes=(3, 5, 7), num_filters=128)
#
#     # 前向计算
#     output = model(x)
#
#     print(f"输入特征形状: {x.shape}")
#     print(f"输出特征形状: {output.shape}")
