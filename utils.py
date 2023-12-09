import numbers
import random
import time

import numpy as np
import torch
from PIL import Image
from torch import nn


class RandomSwap(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return self.swap(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def swap(self, img, crop):
        def crop_image(image, cropnum):
            width, high = image.size
            crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
            crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
            im_list = []
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    im_list.append(
                        image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
            return im_list

        widthcut, highcut = img.size
        img = img.crop((10, 10, widthcut - 10, highcut - 10))
        images = crop_image(img, crop)
        pro = 5
        if pro >= 5:
            tmpx = []
            tmpy = []
            count_x = 0
            count_y = 0
            k = 1
            RAN = 2
            for i in range(crop[1] * crop[0]):
                tmpx.append(images[i])
                count_x += 1
                if len(tmpx) >= k:
                    tmp = tmpx[count_x - RAN:count_x]
                    random.shuffle(tmp)
                    tmpx[count_x - RAN:count_x] = tmp
                if count_x == crop[0]:
                    tmpy.append(tmpx)
                    count_x = 0
                    count_y += 1
                    tmpx = []
                if len(tmpy) >= k:
                    tmp2 = tmpy[count_y - RAN:count_y]
                    random.shuffle(tmp2)
                    tmpy[count_y - RAN:count_y] = tmp2
            random_im = []
            for line in tmpy:
                random_im.extend(line)

            # random.shuffle(images)
            width, high = img.size
            iw = int(width / crop[0])
            ih = int(high / crop[1])
            toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
            x = 0
            y = 0
            for i in random_im:
                i = i.resize((iw, ih), Image.ANTIALIAS)
                toImage.paste(i, (x * iw, y * ih))
                x += 1
                if x == crop[0]:
                    x = 0
                    y += 1
        else:
            toImage = img
        toImage = toImage.resize((widthcut, highcut))
        return toImage


class DCLLoss(nn.Module):

    def __init__(self, abg):
        super(DCLLoss, self).__init__()
        # Alpha, beta and gamma in Eq.(10), balancing different loss.
        self.alpha = abg[0]
        self.beta = abg[1]
        self.gamma = abg[2]
        self.ce_loss = nn.CrossEntropyLoss()
        # self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.add_loss = nn.L1Loss()

    def __call__(self, outputs, labels, labels_swap, swap_law):
        loss_ce = self.ce_loss(outputs[0], labels)
        loss_swap = self.ce_loss(outputs[1], labels_swap)
        loss_law = self.add_loss(outputs[2], swap_law)
        loss = self.alpha * loss_ce + self.beta * loss_swap + self.gamma * loss_law
        return loss


class PerformanceMeter(object):
    """Record the performance metric during training
    """

    def __init__(self, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        self.current_value = self.values[-1]
        self.best_value = self.best_function(self.values)
        self.best_epoch = self.values.index(self.best_value)

    @property
    def value(self):
        return self.values[-1]


class AverageMeter(object):
    """Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


class Timer(object):

    def __init__(self):
        self.start = time.time()
        self.last = time.time()

    def tick(self, from_start=False):
        this_time = time.time()
        if from_start:
            duration = this_time - self.start
        else:
            duration = this_time - self.last
        self.last = this_time
        return duration


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
