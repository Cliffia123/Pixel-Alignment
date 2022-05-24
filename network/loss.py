import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
from torch.autograd import Function
from torch.autograd import Variable

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

sys.path.append(
    join(root_dir,
         "crf/crfwrapper/bilateralfilter/build/lib.linux-x86_64-3.7")
)
from bilateralfilter import bilateralfilter, bilateralfilter_batch
# from network.core import entropy


"""
Area Loss
"""


# class AreaLoss(nn.Module):
#     def __init__(self, topk=25):
#         super(AreaLoss, self).__init__()
#         self.topk = topk
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, p, main_out, features): #p:激活图,main_out:vgg16的输出，features:激活图之前的特征,
#         core = torch.sum(p) / (p.shape[0] * p.shape[2] * p.shape[3])
#         if self.topk != 0:
#             pred_idx = torch.topk(self.softmax(main_out), self.topk, dim=1)[1]
#             for j in range(self.topk):
#                 feat = features[[k for k in range(p.size(0))], pred_idx[:, j], :, :]
#                 core += (torch.sum(feat) / (p.shape[0] * p.shape[2] * p.shape[3]))
#
#         return core

class AreaLoss(nn.Module):
    def __init__(self, topk=25):
        super(AreaLoss, self).__init__()
        self.topk = topk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, p, main_out, features):
        loss = torch.sum(p) / (p.shape[0] * p.shape[2] * p.shape[3])
        if self.topk != 0:
            pred_idx = torch.topk(self.softmax(main_out), self.topk, dim=1)[1]
            for j in range(self.topk):
                feat = features[[k for k in range(p.size(0))], pred_idx[:, j], :, :]
                loss += (torch.sum(feat) / (p.shape[0] * p.shape[2] * p.shape[3]))
        return loss
#
# class BackgroundAreaLoss(nn.Module):
#     def __init__(self, topk=120):
#         super(AreaLoss, self).__init__()
#         self.topk = topk
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, p, main_out, features):
#
#         n = p.shape[0]
#         bl = p[:, 0].view(n, -1).sum(dim=-1).view(-1, )
#         loss = -bl
#         return loss


class BackgroundLoss(nn.Module):
    def __init__(self, topk=80):
        super(BackgroundLoss, self).__init__()
        self.topk = topk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, score_bg, score, target):  # p:激活图,main_out:vgg16的输出，features:激活图之前的特征,
        batch, _= score_bg.size()
        # core = torch.sum(score_bg/score.detach()) / (score.size(0)*score.size(1))
        # print("core",core)
        loss = 0
        for i in range(batch):
            sc_bg = score_bg[i,target[i]]
            sc = score[i,target[i]]
            loss = loss+sc_bg/(sc.detach()+0.00035)
        loss = loss/batch
        return loss


class DenseCRFLossFunction(Function):
    @staticmethod
    def forward(ctx,
                images,
                segmentations,
                sigma_rgb,
                sigma_xy
                ):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape

        # ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)
        # segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        # ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss -= np.dot(segmentations, AS)
        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = - 2 * grad_output * torch.from_numpy(ctx.AS) / ctx.N
        grad_segmentation = grad_segmentation.cuda()
        # grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None


class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        """
        Init. function.
        :param weight: float. It is Lambda for the crf core.
        :param sigma_rgb: float. sigma for the bilateheral filtering (
        appearance kernel): color similarity.
        :param sigma_xy: float. sigma for the bilateral filtering
        (appearance kernel): proximity.
        :param scale_factor: float. ratio to scale the image and
        segmentation. Helpful to control the computation (speed) / precision.
        """
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations):
        """
        Forward core.
        Image and segmentation are scaled with the same factor.

        :param images: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W. DEVICE: CPU.
        :param segmentations: softmaxed logits.
        :return: core score (scalar).
        """
        # segmentations = F.interpolate(segmentations, size=(224, 224), mode='bilinear', align_corners=False)
        segmentations = segmentations.float()

        scaled_images = F.interpolate(images,
                                      scale_factor=self.scale_factor,
                                      mode='nearest',
                                      recompute_scale_factor=False
                                      )
        # print(scaled_images.shape)
        scaled_segs = F.interpolate(segmentations,
                                    scale_factor=self.scale_factor,
                                    mode='bilinear',
                                    recompute_scale_factor=False,
                                    align_corners=False)

        val = self.weight * DenseCRFLossFunction.apply(
            scaled_images,
            segmentations,
            self.sigma_rgb,
            self.sigma_xy * self.scale_factor
        ).cuda()
        return val

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class WeightedEntropyLoss(nn.Module):
    def __init__(self, miu=0.5, sigma=0.1, beta=0.):
        super(WeightedEntropyLoss, self).__init__()

        self.miu = miu
        self.sigma = sigma
        self.eps = torch.finfo(torch.float32).eps
        self.beta = beta

    def _gaussian(self, p):
        return torch.exp(-(p - self.miu) ** 2 / (2 * self.sigma ** 2)) + self.beta

    def forward(self, p):
        # p = torch.sigmoid(p)
        return - torch.sum(
            (p * torch.log(p + self.eps) + (1 - p) * torch.log(1 - p + self.eps)) * self._gaussian(p)) / \
               (p.shape[0] * p.shape[2] * p.shape[3])


#
class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, feature, sim, target, args):
        B = target.size(0)
        feature_norm = torch.norm(feature, dim=1)
        feature_norm_minmax = normalize_minmax(feature_norm)
        sim_target_flat = sim[torch.arange(B), target].view(B, -1)
        feature_norm_minmax_flat = feature_norm_minmax.view(B, -1)
        # if self.args.dataset_name == 'ILSVRC':
        #     sim_fg = (feature_norm_minmax_flat > self.args.sim_fg_thres).float()
        #     sim_bg = (feature_norm_minmax_flat < self.args.sim_bg_thres).float()
        #
        #     sim_fg_mean = (sim_fg * sim_target_flat).sum(dim=1) / (sim_fg.sum(dim=1) + 1e-15)
        #     sim_bg_mean = (sim_bg * sim_target_flat).sum(dim=1) / (sim_bg.sum(dim=1) + 1e-15)
        #     loss_sim = torch.mean(sim_bg_mean - sim_fg_mean)
        #
        #     norm_fg = (sim_target_flat > 0).float()
        #     norm_bg = (sim_target_flat < 0).float()
        #
        #     norm_fg_mean = (norm_fg * feature_norm_minmax_flat).sum(dim=1) / (norm_fg.sum(dim=1) + 1e-15)
        #     norm_bg_mean = (norm_bg * feature_norm_minmax_flat).sum(dim=1) / (norm_bg.sum(dim=1) + 1e-15)
        #
        #     loss_norm = torch.mean(norm_bg_mean - norm_fg_mean)

        # elif self.args.dataset_name == 'CUB':
        # sim_fg = (feature_norm_minmax_flat > args.sim_fg_thres).float()
        sim_bg = (feature_norm_minmax_flat < args.sim_bg_thres).float()

        # sim_fg_mean = (sim_fg * sim_target_flat).sum(dim=1) / (sim_fg.sum(dim=1) + 1e-15)
        sim_bg_mean = (sim_bg * sim_target_flat).sum(dim=1) / (sim_bg.sum(dim=1) + 1e-15)
        # loss_sim = torch.mean(sim_bg_mean - sim_fg_mean)
        loss_sim = torch.mean(sim_bg_mean)


        sim_max_class, _ = sim.max(dim=1)
        sim_max_class_flat = sim_max_class.view(B, -1)

        # norm_fg = (sim_max_class_flat > 0).float()
        norm_bg = (sim_max_class_flat < 0).float()

        # norm_fg_mean = (norm_fg * feature_norm_minmax_flat).sum(dim=1) / (norm_fg.sum(dim=1) + 1e-15)
        norm_bg_mean = (norm_bg * feature_norm_minmax_flat).sum(dim=1) / (norm_bg.sum(dim=1) + 1e-15)

        # loss_norm = torch.mean(norm_bg_mean - norm_fg_mean)
        loss_norm = torch.mean(norm_bg_mean)
    # else:
    #     raise ValueError("dataset_name should be in ['ILSVRC', 'CUB']")

        return loss_sim, loss_norm

class KLUniformLoss(nn.Module):
    """
    KL loss KL(q, p) where q is a uniform distribution.
    This amounts to = -1/c . sum_i log2 p_i.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(KLUniformLoss, self).__init__()

        self.softmax = nn.Softmax(dim=1)  # The log-softmax.

    def forward(self, scores):
        """
        Forward function
        :param scores: unormalized scores (batch_size, nbr_classes)
        :return: loss. scalar.
        """
        logsoftmax = torch.log2(self.softmax(scores))
        loss = (-logsoftmax).mean(dim=1).mean()
        return loss

def normalize_minmax(cams):
    """
    Args:
        cam: torch.Tensor(size=(B, H, W), dtype=np.float)
    Returns:
        torch.Tensor(size=(B, H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    B, _,_, _ = cams.shape
    min_value, _ = cams.view(B, -1).min(1)
    cams_minmax = cams - min_value.view(B, 1, 1)
    max_value, _ = cams_minmax.view(B, -1).max(1)
    cams_minmax /= max_value.view(B, 1, 1) + 1e-15
    return cams_minmax

class _LossExtendedLB(nn.Module):
    """
    Extended log-barrier loss (ELB).
    Optimize inequality constraint : f(x) <= 0.

    Refs:
    1. Kervadec, H., Dolz, J., Yuan, J., Desrosiers, C., Granger, E., and Ben
     Ayed, I. (2019b). Constrained deep networks:Lagrangian optimization
     via log-barrier extensions.CoRR, abs/1904.04205
    2. S. Belharbi, I. Ben Ayed, L. McCaffrey and E. Granger,
    “Deep Ordinal Classification with Inequality Constraints”, CoRR,
    abs/1911.10720, 2019.
    """
    def __init__(self,
                 init_t=1.,
                 max_t=10.,
                 mulcoef=1.01
                 ):
        """
        Init. function.

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        """
        super(_LossExtendedLB, self).__init__()

        msg = "`mulcoef` must be a float. You provided {} ....[NOT OK]".format(
            type(mulcoef))
        assert isinstance(mulcoef, float), msg
        msg = "`mulcoef` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(mulcoef)
        assert mulcoef > 0., msg

        msg = "`init_t` must be a float. You provided {} ....[NOT OK]".format(
            type(init_t))
        assert isinstance(init_t, float), msg
        msg = "`init_t` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(init_t)
        assert init_t > 0., msg

        msg = "`max_t` must be a float. You provided {} ....[NOT OK]".format(
            type(max_t))
        assert isinstance(max_t, float), msg
        msg = "`max_t` must be > `init_t`. float. You provided {} " \
              "....[NOT OK]".format(max_t)
        assert max_t > init_t, msg

        self.init_t = init_t

        self.register_buffer(
            "mulcoef", torch.tensor([mulcoef], requires_grad=False).float())
        # create `t`.
        self.register_buffer(
            "t_lb", torch.tensor([init_t], requires_grad=False).float())

        self.register_buffer(
            "max_t", torch.tensor([max_t], requires_grad=False).float())

    def set_t(self, val):
        """
        Set the value of `t`, the hyper-parameter of the log-barrier method.
        :param val: float > 0. new value of `t`.
        :return:
        """
        msg = "`t` must be a float. You provided {} ....[NOT OK]".format(
            type(val))
        assert isinstance(val, float) or (isinstance(val, torch.Tensor) and
                                          val.ndim == 1 and
                                          val.dtype == torch.float), msg
        msg = "`t` must be > 0. float. You provided {} ....[NOT OK]".format(val)
        assert val > 0., msg

        if isinstance(val, float):
            self.register_buffer(
                "t_lb", torch.tensor([val], requires_grad=False).float()).to(
                self.t_lb.device
            )
        elif isinstance(val, torch.Tensor):
            self.register_buffer("t_lb", val.float().requires_grad_(False))

    def get_t(self):
        """
        Returns the value of 't_lb'.
        """
        return self.t_lb

    def update_t(self):
        """
        Update the value of `t`.
        :return:
        """
        self.set_t(torch.min(self.t_lb * self.mulcoef, self.max_t))

    def forward(self, fx):
        """
        The forward function.
        :param fx: pytorch tensor. a vector.
        :return: real value extended-log-barrier-based loss.
        """
        assert fx.ndim == 1, "fx.ndim must be 1. found {}.".format(fx.ndim)

        loss_fx = fx * 0.

        # vals <= -1/(t**2).
        ct = - (1. / (self.t_lb**2))

        idx_less = ((fx < ct) | (fx == ct)).nonzero().squeeze()
        if idx_less.numel() > 0:
            val_less = fx[idx_less]
            loss_less = - (1. / self.t_lb) * torch.log(- val_less)
            loss_fx[idx_less] = loss_less

        # vals > -1/(t**2).
        idx_great = (fx > ct).nonzero().squeeze()
        if idx_great.numel() > 0:
            val_great = fx[idx_great]
            loss_great = self.t_lb * val_great - (1. / self.t_lb) * \
                torch.log((1. / (self.t_lb**2))) + (1. / self.t_lb)
            loss_fx[idx_great] = loss_great

        loss = loss_fx.sum()

        return loss

    def __str__(self):
        return "{}(): extended-log-barrier-based method.".format(
            self.__class__.__name__)

class Size_const(nn.Module):
    def __init__(self,normalize_sz,init_t=1.,max_t=10., mulcoef=1.01,):
        super(Size_const, self).__init__()
        self.normalize_sz = normalize_sz
        self.epsilon = 0
        self.elb = _LossExtendedLB(init_t=init_t,
                                   max_t=max_t,
                                   mulcoef=mulcoef
                                   )
    def forward(self, masks_pred):
        """
        Compute the loss over the size of the masks.
        :param masks_pred: foreground predicted mask. shape: (bs, 1, h, w).
        :return: ELB loss. a scalar that is the sum of the losses over bs.
        """
        # assert masks_pred.ndim == 4, "Expected 4 dims, found {}.".format(
        #     masks_pred.ndim)
        #
        # msg = "nbr masks must be 1. found {}.".format(masks_pred.shape[1])
        # assert masks_pred.shape[1] == 1, msg

        # background
        backgmsk = 1. - masks_pred
        bsz = backgmsk.shape[0]
        h = backgmsk.shape[2]
        w = backgmsk.shape[3]
        l1 = torch.abs(backgmsk.contiguous().view(bsz, -1)).sum(dim=1)
        if self.normalize_sz:
            l1 = l1 / float(h * w)
        loss_back = self.elb(-l1)

        # foreground
        l1_fg = torch.abs(masks_pred.contiguous().view(bsz, -1)).sum(dim=1)
        if self.normalize_sz:
            l1_fg = l1_fg / float(h * w)

        l1_fg = l1_fg - self.epsilon
        loss_fg = self.elb(-l1_fg)

        return loss_back + loss_fg

class ImgReconstruction(nn.Module):
    def __init__(self, **kwargs):
        super(ImgReconstruction, self).__init__()

        self.loss = nn.MSELoss(reduction="none")
        self.elb = _LossExtendedLB(init_t=1., max_t=10., mulcoef=1.01)

    def forward(self,x_in=None,im_recon=None):

        n = x_in.shape[0]
        loss = self.elb(self.loss(x_in, im_recon).view(n, -1).mean( dim=1).view(-1, ))
        return loss.mean()

class SelfLearningFcams(nn.Module):
    def __init__(self):
        super(SelfLearningFcams, self).__init__()
        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=-255)
        # self.loss =  F.BCEWithLoss()

    def forward(self,fcams, seeds, weight):

        # fcams = F.softmax(fcams, dim=1)
        fcams = F.interpolate(fcams, size=(224, 224), mode='bilinear', align_corners=False)
        loss= F.binary_cross_entropy_with_logits(fcams.squeeze(1), seeds, weight=weight)
        # loss = self.loss(fcams, seeds) #32 2 224 224 vs 32 224 224
        return loss

class MaxSizePositiveFcams(nn.Module):
    def __init__(self):
        super(MaxSizePositiveFcams, self).__init__()

        self.elb = _LossExtendedLB(init_t=1., max_t=10., mulcoef=1.01)

    def forward(self,fcams=None):

        # if fcams.shape[1] > 1:
        # fcams = F.interpolate(fcams, size=(224, 224), mode='bilinear', align_corners=False)
        # fcams_n = F.softmax(fcams, dim=1)
        # else:
        fcams =  F.interpolate(fcams, size=(224, 224), mode='bilinear', align_corners=False)
        fcams_n = F.sigmoid(fcams)
        fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return loss * (1. / 2.)

if __name__ == '__main__':
    a = torch.rand(32,1,7,7)
    b = torch.rand(32,200)
    c = torch.rand(32,200,7,7)
    loss =  AreaLoss()
    lo = loss(a,b,c)
    print(lo)



