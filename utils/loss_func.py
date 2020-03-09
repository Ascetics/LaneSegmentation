import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegLoss(nn.Module):
    def __init__(self, loss_type, weight=None, ignore_index=-100, reduction='mean'):
        """
        :param loss_type: loss函数类型决定forward，可以取值
            'cross_entropy'、'dice'、'cross_entropy+dice'
        :param weight: 从F.cross_entropy抄过来的注释
            weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        :param ignore_index: 从F.cross_entropy抄过来的注释
            ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
          ``True``, the loss is averaged over non-ignored targets. Default: -100
        :param reduction: 从F.cross_entropy抄过来的注释
            reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        super(SemanticSegLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.loss_type = loss_type
        pass

    def forward(self, output, label):
        """
        语义分割loss计算
        :param output: [N,C,H,W]，没有经过softmax的模型输出
        :param label: [N,H,W]，label就是grand truth，非ont-hot编码
        :return: loss
        """
        if self.loss_type == 'cross_entropy':
            return self.cross_entropy(output, label)
        elif self.loss_type == 'dice':
            return self.dice(output, label)
        elif self.loss_type == 'cross_entropy+dice':
            loss = self.cross_entropy(output, label) + self.dice(output, label)
            return loss
        else:
            raise NotImplementedError

    def dice(self, output, label):
        """
                    2|A*B|
        dice 系数 = --------   表示A和B的相似程度，越接近1越相似
                    |A|+|B|

                          2|A*B|
        dice loss = 1 -  -------- 表示A和B越相似，loss就应该越小
                          |A|+|B|

                          1+2|A*B|
        dice loss = 1 -  ---------- 有效防止分母为0
                          1+|A|+|B|

        :param output: [N,C,H,W]，没有经过softmax的模型输出
        :param label: [N,H,W]，label就是grand truth，非ont-hot编码
        :return:
        """
        assert output.shape[0] == label.shape[0]  # N相同
        assert output.shape[2] == label.shape[1]  # H相同
        assert output.shape[3] == label.shape[2]  # W相同

        probs = F.softmax(output, dim=1)  # 输出变成概率形式，表示对该类分类的概率
        # probs = F.sigmoid(output)
        one_hot = torch.zeros(output.shape).scatter_(1, label.unsqueeze(1), 1)  # label[N,H,W]变成one-hot形式[N,C,H,W]

        numerator = (probs * one_hot).sum(dim=(2, 3))  # [N,C] 计算分子|AB|
        denominator = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))  # [N,C] 计算分母|A|+|B|

        if self.weight:  # 如果类别有权重
            numerator = numerator * self.weight.view(1, -1)  # 从[C]看成维度[1,C]
            denominator = denominator * self.weight.view(1, -1)  # 从[C]看成维度[1,C]
            pass
        smooth = 1.
        loss = (smooth + 2. * numerator.sum(dim=1)) / (smooth + denominator.sum(dim=1))
        loss = 1. - loss

        if self.reduction == 'mean':  # N个平均
            loss = loss.mean()
            pass
        return loss

    def cross_entropy(self, output, label):
        """
        :param output: [N,C,H,W]，没有经过softmax的模型输出
        :param label: [N,H,W]，label就是grand truth，非ont-hot编码
        :return: 交叉熵
        """
        return F.cross_entropy(output, label,
                               weight=self.weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction)

    pass


if __name__ == '__main__':
    # a = torch.Tensor([[1., 2., 3.],
    #                   [4., 5., 6.]])
    # b = torch.Tensor([0.2, 0.7, 0.1])
    # c = a * b
    # print(c)
    a = torch.randint(0, 5, (1, 2, 2, 2), dtype=torch.float, requires_grad=True) / 5
    print(a)
    b = torch.randint(0, 2, (1, 2, 2))
    print(b)

    # dice = SemanticSegLoss(loss_type='dice')
    # dloss = dice(a, b)
    # print('dice', dloss.item())
    # print('dice back', dloss.backward())

    cd = SemanticSegLoss(loss_type='cross_entropy+dice')
    cdloss = cd(a, b)
    print('cd', cdloss.item())
    print('cd back', cdloss.backward())

    pass
