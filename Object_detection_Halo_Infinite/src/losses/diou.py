import torch
import torch.nn.functional as F


def diou_loss(pred_boxes, gt_boxes, eps=1e-6):
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = F.relu(x2 - x1) * F.relu(y2 - y1)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = (area_pred + area_gt - inter).clamp(min=eps)
    iou = inter / union

    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gt_center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    center_dist = (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2

    enc_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    diag = ((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2).clamp(min=eps)

    return (1 - (iou - center_dist / diag)).mean()
