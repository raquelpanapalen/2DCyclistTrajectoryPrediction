import torch
import torchvision

def get_iou(bboxes1, bboxes2):
    """
    torchvision.ops.box_iou:

    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    iou = torchvision.ops.box_iou(bboxes1, bboxes2)
    return torch.diagonal(iou)


