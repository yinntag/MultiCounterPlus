import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class MemoryInducedTracking(nn.Module):
    def __init__(self, alpha=0.7):
        super(MemoryInducedTracking, self).__init__()
        self.alpha = alpha  # 控制更新比率的因子

    def match_from_embds(self, memory_embds, current_embds):
        # 计算相似度矩阵
        sim_matrix = torch.matmul(current_embds, memory_embds.t())

        # 使用匈牙利算法找到最大匹配
        row_ind, col_ind = linear_sum_assignment(-sim_matrix.detach().cpu().numpy())

        return col_ind

    def forward(self, pred_logits, pred_obj_logits, pred_masks, pred_embds):
        # 初始化输出列表
        out_logits = []
        out_obj_logits = []
        out_masks = []
        out_embds = [pred_embds[0]]  # 初始的记忆嵌入为第一帧的目标嵌入

        # 循环遍历每个帧
        for i in range(1, len(pred_logits)):
            # 匹配当前帧的嵌入与上一帧的记忆嵌入
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            # 更新输出结果
            out_logits.append(pred_logits[i][indices, :])
            out_obj_logits.append(pred_obj_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])

            # 更新记忆嵌入
            tmp_pred_embds = self.alpha * pred_embds[i][indices, :] + (1 - self.alpha) * out_embds[-1]
            out_embds.append(tmp_pred_embds)

        return out_logits, out_obj_logits, out_masks, out_embds


# 示例数据
d_model = 256
num_frames = 5
num_queries = 10

# 生成随机数据
pred_logits = [torch.randn(num_queries, d_model) for _ in range(num_frames)]
pred_obj_logits = [torch.randn(num_queries, d_model) for _ in range(num_frames)]
pred_masks = [torch.randn(num_queries, 10, 10) for _ in range(num_frames)]
pred_embds = [torch.randn(num_queries, d_model) for _ in range(num_frames)]

# 创建模型实例
model = MemoryInducedTracking(d_model)

# 运行模型
out_logits, out_obj_logits, out_masks, out_embds = model(pred_logits, pred_obj_logits, pred_masks, pred_embds)

# 打印输出结果
print("Output Logits:", out_logits.shape)
print("Output Object Logits:", out_obj_logits.shape)
print("Output Masks:", out_masks.shape)
print("Output Embeddings:", out_embds.shape)