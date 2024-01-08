import torch
from torch import Tensor
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def Jaccard_coeff(y_pred, y_true, last=False):
    assert y_true.size(0) == y_pred.size(0)
    smooth = 0
    eps = 1e-7

    y_pred = y_pred.log_softmax(dim=1).exp()
    bs = y_true.size(0)
    num_classes = y_pred.size(1)
    dims = 2

    y_true = y_true.view(bs, -1)
    y_pred = y_pred.view(bs, num_classes, -1).argmax(dim=1)
    y_pred = F.one_hot(y_pred, num_classes)  # N,H*W -> N,H*W, C
    y_pred = y_pred.permute(0, 2, 1)  # N, C, H*W

    y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # N, C, H*W

    scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=smooth, eps=eps, dims=dims)
    
    if last == True:
        cm = torch.zeros([num_classes, num_classes]).to(y_true.device)
        for i in range(num_classes):
            for j in range(num_classes):
                cm[i, j] += (y_true[:, i, :] * y_pred[:, j, :]).sum()
    else:
        cm = torch.zeros([num_classes, num_classes])

    mask = y_true.sum(dims) > 0
    scores *= mask.float()
    scores = scores.sum(0) / mask.sum(0).clamp_min(eps)
    mask = mask.sum(0) > 0

    return scores.cpu(), cm.cpu(), mask.cpu()

def soft_jaccard_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    
    assert output.size() == target.size()

    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score
