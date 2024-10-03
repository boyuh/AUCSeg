import warnings

import torch
import torch.nn as nn
from abc import abstractmethod
from ..builder import LOSSES

@LOSSES.register_module()
class AUCLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo',
                 loss_name='loss_auc',
                 loss_weight=1.0, *kwargs):
        super(AUCLoss, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.transform = transform
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def _check_input(self, pred, target):
        assert pred.max() <= 1 and pred.min() >= 0
        assert target.min() >= 0
        assert pred.shape[0] == target.shape[0]

    def _ignore_index(self, pred, target, ignore_index):
        unmask = torch.where(target != ignore_index)[0]
        new_pred = pred[unmask,:]
        new_target = target[unmask]
        return new_pred, new_target

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):

        pred = torch.sigmoid(pred)
        pred = pred.transpose(0, 1).reshape(self.num_classes, -1).transpose(0, 1)
        target = target.reshape(-1)

        pred, target = self._ignore_index(pred, target, ignore_index)
        self._check_input(pred, target)

        Y = torch.stack(
            [target.eq(i).float() for i in range(self.num_classes)],
            1).squeeze()

        N = Y.sum(0)
        D = 1 / N[target.squeeze().long()]

        batch_num_classes = self.num_classes - torch.where(N == 0)[0].size()[0]

        if self.num_classes == 2:
            Y = target.float()
            numPos = torch.sum(Y.eq(1))
            numNeg = torch.sum(Y.eq(0))
            Di = 1.0 / numPos / numNeg
            return self.calLossPerCLass(pred.squeeze(1), Y, Di, numPos)
        else:
            if self.transform == 'ovo':
                factor = batch_num_classes * (batch_num_classes - 1)
            else:
                factor = 1

            loss = torch.tensor([0.], device=pred.device)
            if self.transform == 'ova':
                ones_vec = torch.ones_like(D, device=pred.device)

            for i in range(self.num_classes):
                if N[i] == 0:
                    continue
                if self.transform == 'ovo':
                    Di = D / N[i]
                else:
                    fac = torch.tensor([1.0], device=pred.device) / (N[i] * (N.sum() - N[i]))
                    Di = fac * ones_vec
                Yi, predi = Y[:, i], pred[:, i]
                loss += self.calLossPerCLass(predi, Yi, Di, N[i], batch_num_classes)

            return (loss / factor) * self.loss_weight

    def calLossPerCLass(self, predi, Yi, Di, Ni, batch_num_classes):

        return self.calLossPerCLassNaive(predi, Yi, Di, Ni, batch_num_classes)

    @abstractmethod
    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        pass

    @property
    def loss_name(self):
        return self._loss_name

@LOSSES.register_module()
class SquareAUCLoss(AUCLoss):
    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo',
                 loss_name='loss_square_auc',
                 loss_weight=1.0, **kwargs):
        super(SquareAUCLoss, self).__init__(num_classes, gamma, transform, loss_name, loss_weight)

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni, batch_num_classes):
        diff = predi - self.gamma * Yi
        nD = Di.mul(1 - Yi)
        fac = (batch_num_classes -
               1) if self.transform == 'ovo' else torch.tensor(1.0, device=predi.device)
        S = Ni * nD + (fac * Yi / Ni)
        diff = diff.reshape((-1, ))
        S = S.reshape((-1, ))
        A = diff.mul(S).dot(diff)
        nD= nD.reshape((-1, ))
        Yi= Yi.reshape((-1, ))
        B = diff.dot(nD) * Yi.dot(diff)
        return 0.5 * A - B

    @property
    def loss_name(self):
        return self._loss_name

@LOSSES.register_module()
class HingeAUCLoss(AUCLoss):
    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo',
                 loss_name='loss_hinge_auc',
                 loss_weight=1.0, **kwargs):
        super(HingeAUCLoss, self).__init__(num_classes, gamma, transform, loss_name, loss_weight)

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni, batch_num_classes):
        fac = 1 if self.transform == 'ova' else (batch_num_classes - 1)
        delta1 = (fac / Ni) * Yi * predi
        delta2 = Di * (1 - Yi) * predi
        return fac * self.gamma - delta1.sum() + delta2.sum()

    @property
    def loss_name(self):
        return self._loss_name

@LOSSES.register_module()
class ExpAUCLoss(AUCLoss):
    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo',
                 loss_name='loss_epx_auc',
                 loss_weight=1.0, **kwargs):
        super(ExpAUCLoss, self).__init__(num_classes, gamma, transform, loss_name, loss_weight)

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni, batch_num_classes):
        C1 = Yi * torch.exp(-self.gamma * predi)
        C2 = (1 - Yi) * torch.exp(self.gamma * predi)
        C2 = Di * C2
        return C1.sum() * C2.sum()

    @property
    def loss_name(self):
        return self._loss_name