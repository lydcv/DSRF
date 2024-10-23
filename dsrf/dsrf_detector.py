import copy
from typing import Dict, List, Optional

from mmdet.models.builder import DETECTORS
from mmfewshot.detection.models import MetaRCNN

from .utils import TestMixins
import torch
from torch import Tensor
from mmcv.runner import auto_fp16
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.utils import ConfigDict
from mmfewshot.detection.models.builder import build_information_fusion
from torch.autograd import Function

@DETECTORS.register_module()
class DSRF(MetaRCNN, TestMixins):
    def __init__(self, *args, with_refine=False,information_fusion: Optional[ConfigDict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # refine results for COCO. We do not use it for VOC.
        self.with_refine = with_refine
        self.backatt = BackAtt(1024)
        self.gre = GRE(in_channel=1024, in_spatial=14*14)  
   
        self.with_information_fusion = False
        if information_fusion is not None:
            self.with_information_fusion = True
            self.information_fusion = build_information_fusion(information_fusion)
      
    
    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img):
        """Extracting features from support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of input image, each item with shape
                (N, C, H, W).
        """
        feats = self.backbone(img, use_meta_conv=True)
        if self.support_neck is not None:
            feats = self.support_neck(feats) 
            
        ba_list = []
        for f in feats:
            f_ = self.backatt(f)
            ba_list.append(f_)
        res = []
        for i in ba_list:
            out = self.gre(i)
            res.append(out)
        return res
        
    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        global count
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        
        

        
        ##################################加入CIC##################################
        if not self.with_information_fusion:
            support_feats = self.extract_support_feat(support_img)
        else:
            support_feats_origin = self.extract_support_feat(support_img)
            batch_size = len(query_data['img_metas'])
            num_support = support_feats_origin[0].size(0)
            support_feats = []
            for img_id in range(batch_size):
                query_feats_ = (query_feats[0][img_id].unsqueeze(0)).repeat(num_support, 1, 1, 1)
                support_feats_ = self.information_fusion(support_feats_origin[0], query_feats_)
                support_feats.append(support_feats_)
        ###########################################################################

        

        # stop gradient at RPN
        query_feats_rpn = [x.detach() for x in query_feats]
        query_feats_rcnn = query_feats


        
        # support_feats = self.extract_support_feat(support_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_rpn,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_rpn,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats_rcnn,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals = None, rescale = False): 
        bbox_results = super().simple_test(img, img_metas, proposals, rescale)
        if self.with_refine:
            return self.refine_test(bbox_results, img_metas)
        else:
            return bbox_results

class BackAtt(nn.Module):
    def __init__(self, in_dim):
        super(BackAtt, self).__init__()
        
        self.linear = nn.Linear(in_dim, 1)
        init.normal_(self.linear.weight, std=0.01)
        init.constant_(self.linear.bias, 0)
        
    def forward(self, x):
        single_s_mat_flatten = x.view(-1, 196, 1024)  # [B, 196, 1024]        
        support_spatial = self.linear(single_s_mat_flatten)  # [B, 196, 1]
        support_spatial = F.softmax(support_spatial, dim=1)
        support_channel_global = torch.bmm(support_spatial.transpose(1, 2), single_s_mat_flatten)  # [B, 1, 1024]
        single_s_mat_flatten = single_s_mat_flatten + 0.1 * F.leaky_relu(support_channel_global)   #    [B, 196, 1024]        
        out = single_s_mat_flatten.view(-1, 1024, 14, 14)

        return out


class GRE(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=False, use_channel=True,
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(GRE, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        # self.inter_channel = in_channel // cha_ratio
        # self.inter_spatial = in_spatial // spa_ratio
        self.inter_channel = max(in_channel // cha_ratio, 1)
        self.inter_spatial = max(in_spatial // spa_ratio, 1)

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = max(1 + self.inter_channel, 1)  # 确保至少为1
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=max(num_channel_c // down_ratio, 1),  # 防止为0
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(max(num_channel_c // down_ratio, 1)),  # 同样防止为0
                nn.ReLU(),
                nn.Conv2d(in_channels=max(num_channel_c // down_ratio, 1), out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out = Gs.view(b, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                x = F.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            out = F.sigmoid(W_yc) * x

            return out
