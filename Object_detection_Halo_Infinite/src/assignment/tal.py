from torchvision.ops import box_iou

def TAL_assigner(
    anchors,
    anchor_centers,
    pred_cls_logits,
    pred_boxes,
    gt_boxes,
    gt_labels,
    alpha=6.0,
    beta=1.0,
    top_k=6
):
    B, N, C = pred_cls_logits.shape
    device = pred_cls_logits.device

    cls_targets = torch.zeros((B, N), dtype=torch.long, device=device)  # 0 = background
    box_targets = torch.zeros((B, N, 4), dtype=torch.float, device=device)
    foreground_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

    for b in range(B):
        boxes = gt_boxes[b]  # [M, 4]
        labels = gt_labels[b]  # [M]
        M = boxes.shape[0]

        if M == 0:
            continue

        cls_probs = torch.softmax(pred_cls_logits[b], dim=-1)  # [N, C+1]
        ious = box_iou(pred_boxes[b], boxes)  # [N, M]

        assigned_gt_inds = torch.full((N,), -1, dtype=torch.long, device=device)  # -1 = not assigned yet

        for m in range(M):
            gt_box = boxes[m]
            gt_label = labels[m]

            x1, y1, x2, y2 = gt_box
            cx, cy = anchor_centers[0, :, 0], anchor_centers[0, :, 1]

            inside_mask = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
            if inside_mask.sum() == 0:
                continue

            scores = cls_probs[inside_mask, gt_label + 1]  # +1 because 0 = background
            iou_vals = ious[inside_mask, m]

            t_metric = (scores ** alpha) * (iou_vals ** beta)

            topk_idx = torch.topk(t_metric, k=min(top_k, t_metric.numel()), dim=0).indices
            anchor_idx = inside_mask.nonzero(as_tuple=False).squeeze(1)[topk_idx]

            for idx in anchor_idx:
                if assigned_gt_inds[idx] == -1:
                    # Если anchor ещё свободен, просто назначаем
                    assigned_gt_inds[idx] = m
                else:
                    # Если anchor уже занят другим GT — выбираем лучший по IoU
                    prev_m = assigned_gt_inds[idx]
                    if ious[idx, m] > ious[idx, prev_m]:
                        assigned_gt_inds[idx] = m

        # Теперь окончательно присваиваем метки и боксы
        pos_inds = (assigned_gt_inds >= 0)
        assigned_m = assigned_gt_inds[pos_inds]

        cls_targets[b, pos_inds] = labels[assigned_m] + 1  # +1 потому что фон = 0
        box_targets[b, pos_inds] = boxes[assigned_m]
        foreground_mask[b, pos_inds] = True

    return {
        "cls_targets": cls_targets,
        "box_targets": box_targets,
        "foreground_mask": foreground_mask
    }
