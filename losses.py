import numpy as np
import torch
import torch.nn as nn

class Dice_loss(nn.Module):
    def __init__(self):  
        super(Dice_loss, self).__init__()

    def forward(self, pred, mask):
        '''
        :param pred: predict result, shape is [b, c, x, y]
        :param mask: shape is [b, c, x, y]
        :return:
        '''

        intersection = pred * mask 
        dice_per_class = 1. - (2.0 * torch.sum(intersection, dim=(1, 2, 3)) + 1.0) / (
            torch.sum(pred, dim=(1, 2, 3)) + torch.sum(mask, dim=(1, 2, 3)) + 1.0)
        dice_loss = torch.sum(dice_per_class, dim=0) / pred.size(0)
        return dice_loss


