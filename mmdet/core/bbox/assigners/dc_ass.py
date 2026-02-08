class TeViTHungarianAssigner(BaseAssigner):
    def __init__(self, num_sub_clips=4, eps=1e-7,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.num_sub_clips = num_sub_clips
        self.eps = eps

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               gt_ids=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        clip_length = len(bbox_pred)
        total_gt_ids = torch.unique(torch.cat(gt_ids))
        num_gts = total_gt_ids.numel()

        num_bboxes = bbox_pred[0].size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred[0].new_full((num_bboxes,),
                                                 -1,
                                                 dtype=torch.long)
        assigned_labels = bbox_pred[0].new_full((num_bboxes,),
                                                -1,
                                                dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return [
                AssignResult(
                    num_gts, assigned_gt_inds, None, labels=assigned_labels)
                for _ in range(clip_length)
            ]
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes[0].new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        costs = []
        for i in range(clip_length):  # for each frame
            cls_cost = self.cls_cost(cls_pred[i], gt_labels[i])
            # regression L1 cost
            normalize_gt_bboxes = gt_bboxes[
                                      i] / factor  # gt is in the form of [x_1, y_1, x_2, y_2], divided by the image resolution for normalization
            reg_cost = self.reg_cost(bbox_pred[i], normalize_gt_bboxes)
            # regression iou cost, defaultly giou is used in official DETR.
            bboxes = bbox_cxcywh_to_xyxy(bbox_pred[i]) * factor
            iou_cost = self.iou_cost(bboxes, gt_bboxes[i])
            # weighted sum of above three costs
            cost = cls_cost + reg_cost + iou_cost
            costs.append(cost)

        # 3. build bi-directional one-to-one correspondance
        ims_to_total, ims_to_total_weights, totals_to_ims = [], [], []
        total_gt_ids_list = total_gt_ids.tolist()
        for gt_id in gt_ids:
            per_ims_to_total = []
            per_ims_to_total_weights = []
            totals_to_per_ims = []

            for gid in total_gt_ids_list:
                if gid in gt_id.tolist():
                    per_ims_to_total.append(gt_id.tolist().index(gid))
                    per_ims_to_total_weights.append(1.)
                    totals_to_per_ims.append(gt_id.tolist().index(gid))
                else:  # If an id is missing from the current frame, the cost weight of the match for this id in that frame will be 0
                    per_ims_to_total.append(-1)
                    per_ims_to_total_weights.append(0.)
            per_ims_to_total = gt_id.new_tensor(
                per_ims_to_total, dtype=torch.int64)  # list to tensor
            per_ims_to_total_weights = gt_id.new_tensor(
                per_ims_to_total_weights, dtype=torch.float32)
            ims_to_total.append(per_ims_to_total)
            ims_to_total_weights.append(per_ims_to_total_weights)

            totals_to_per_ims = gt_id.new_zeros(num_gts + 1,
                                                dtype=torch.int64)  # num_gt+1 the first dimension is for none-object, others are the true gts
            totals_to_per_ims[1:] = per_ims_to_total + 1
            totals_to_ims.append(totals_to_per_ims)

        costs_ = [
            cost[:, indices] * weights
            for (cost, indices,
                 weights) in zip(costs, ims_to_total, ims_to_total_weights)
        ]
        cost = sum(costs_) / sum(
            ims_to_total_weights)  # summarize to form the video-level cost between each spatio-temporal query and gt

        # 4. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(
            cost)  # row_inds: the index of the matched proposal, col_inds: the matched gt index corresponding to those proposals
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred[0].device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred[0].device)  # numpy to torch.tensor

        # 5. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assign_results = []
        for _ in range(clip_length):  # for each frame
            assigned_gt_inds = bbox_pred[0].new_full((num_bboxes,),
                                                     0,
                                                     dtype=torch.long)  # shape: num_proposal, matched gt index for each proposal. Initialize to 0 first
            assigned_labels = bbox_pred[0].new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)  # shape: num_proposal, the matched label for each proposal. Initialize to -1 first
            # assign foregrounds based on matching results
            matched_col_inds_per_im = totals_to_ims[_][matched_col_inds + 1]  # none-object: 0 gt objects: >0
            matched_row_inds_per_im = matched_row_inds[
                matched_col_inds_per_im > 0]
            matched_col_inds_per_im = matched_col_inds_per_im[
                matched_col_inds_per_im > 0]
            assigned_gt_inds[
                matched_row_inds_per_im] = matched_col_inds_per_im  # record matched gt index to each proposal, 0 means none-object
            assigned_labels[matched_row_inds_per_im] = gt_labels[_][
                matched_col_inds_per_im - 1]  # give each proposal the label of the matched gt. 0 for face and -1 for none-objects
            assign_results.append(
                AssignResult(
                    gt_ids[_].numel(),
                    assigned_gt_inds,
                    None,
                    labels=assigned_labels))
        # print('assign_results:', assign_results)
        return assign_results

    def assign_with_overlap(self,
                            bbox_pred,
                            cls_pred,
                            gt_bboxes,
                            gt_labels,
                            img_meta,
                            gt_bboxes_ignore=None,
                            gt_ids=None):
        """
        引入窗口交叠的分治方法进行局部匹配并合并结果
        Args:
            bbox_pred: 预测的边界框
            cls_pred: 预测的类别分数
            gt_bboxes: GT 边界框
            gt_labels: GT 类别标签
            img_meta: 图像元信息
            gt_bboxes_ignore: 忽略的 GT 边界框
            gt_ids: GT 的 ID
        Returns:
            global_assign_results: 全局匹配结果
        """
        clip_length = len(bbox_pred)
        if clip_length <= self.num_sub_clips:  # 如果序列较短，无需分治
            return self.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore, gt_ids)

        # Step 1: 划分子区间 (带 50% 窗口交叠)
        sub_clip_size = (clip_length + self.num_sub_clips - 1) // self.num_sub_clips
        sub_assign_results = []
        overlaps = int(sub_clip_size // 2)  # 50% 的窗口交叠大小

        for i in range(self.num_sub_clips):
            start_idx = max(0, i * sub_clip_size - overlaps)
            end_idx = min(clip_length, (i + 1) * sub_clip_size + overlaps)

            # 获取局部帧范围内的预测和 GT
            local_bbox_pred = bbox_pred[start_idx:end_idx]
            local_cls_pred = cls_pred[start_idx:end_idx]
            local_gt_bboxes = gt_bboxes[start_idx:end_idx]
            local_gt_labels = gt_labels[start_idx:end_idx]
            local_gt_ids = gt_ids[start_idx:end_idx] if gt_ids is not None else None

            # Step 2: 对每个子区间执行局部匹配
            local_result = self.assign(local_bbox_pred, local_cls_pred, local_gt_bboxes,
                                       local_gt_labels, img_meta, gt_bboxes_ignore, local_gt_ids)
            sub_assign_results.append((start_idx, end_idx, local_result))

        # Step 3: 合并局部结果
        global_assign_results = [None] * clip_length
        for start_idx, end_idx, sub_result in sub_assign_results:
            for frame_idx, result in enumerate(sub_result):
                global_idx = start_idx + frame_idx
                if global_idx < clip_length:
                    # 若结果已有值，则优先保留靠后的匹配结果
                    if global_assign_results[global_idx] is None:
                        global_assign_results[global_idx] = result

        return global_assign_results