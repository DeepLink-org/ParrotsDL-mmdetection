# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmdet.models.utils import build_linear_layer
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils.normed_predictor import NormedLinear
from ..builder import HEADS
from .detr_head import DETRHead

from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple
from torch.nn.init import normal_
from ..utils import DyReLU
import sys
import numpy as np
from inspect import signature
from mmcv.ops import batched_nms
from mmdet.core import bbox_mapping_back, merge_aug_proposals
if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed

class FFN(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 hidden_dims=1024):
        super().__init__()
        self.embed_dims = embed_dims
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, embed_dims)
        )
        self.norm = nn.LayerNorm(embed_dims)
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.norm(x + self.layers(x))
        return x


def load_class_freq(
    path='/mnt/lustre/zongzhuofan/labels/lvis/lvis_v1_train_cat_info.json', freq_weight=1.0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


@HEADS.register_module()
class CompositeDeformDETRHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 num_patterns=1,
                 max_pos_priors=300,
                 lambda_1=1,
                 use_labeled_loss=False,
                 labeled_mask=None,
                 with_box_refine=False,
                 as_two_stage=False,
                 mixed_selection=False,
                 transformer=None,
                 frozen_stages=-1,
                 query_attn=False,
                 cost_aware=False,
                 use_zero_padding=False,
                 norm_on_cls=False,
                 **kwargs):
        self.num_patterns = num_patterns
        self.max_pos_priors = max_pos_priors
        self.lambda_1 = lambda_1
        self.frozen_stages = frozen_stages
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.mixed_selection = mixed_selection
        self.query_attn = query_attn
        self.cost_aware = cost_aware
        self.use_zero_padding = use_zero_padding
        self.norm_on_cls = norm_on_cls
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if self.mixed_selection:
            transformer['mixed_selection'] = self.mixed_selection

        super(CompositeDeformDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.return_enc_intermediate = self.transformer.return_enc_intermediate
        self.use_head_feats = self.transformer.use_head_feats
        self.use_labeled_loss = use_labeled_loss
        if self.use_labeled_loss:
            self.labeled_mask = {}
            f = open(labeled_mask, 'r')
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                mask = np.array(data['labeled_cls'])
                mask = torch.from_numpy(mask).cuda().long()
                self.labeled_mask[data['filename']] = mask
            

    def _freeze_stages(self):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        if self.frozen_stages >= 0:
            self.transformer.encoder.eval()
            for m in [self.transformer.encoder]:
                for param in m.parameters():
                    param.requires_grad = False
        if self.frozen_stages >= 1:
            self.query_embedding.eval()
            for m in [self.query_embedding]:
                for param in m.parameters():
                    param.requires_grad = False
        if self.frozen_stages >= 2:
            self.transformer.decoder.eval()
            for m in [self.transformer.decoder]:
                for param in m.parameters():
                    param.requires_grad = False
        if self.frozen_stages == -10:
            self.query_embedding.eval()
            for m in [self.query_embedding]:
                for param in m.parameters():
                    param.requires_grad = False


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims)
        )
        if self.norm_on_cls:
            fc_cls = NormedLinear(self.embed_dims, self.cls_out_channels, tempearture=20)
        else:
            fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:

            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if self.num_patterns > 1:
            self.patterns_embed = nn.Embedding(self.num_patterns, self.embed_dims)

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        elif self.mixed_selection:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims)
        self._freeze_stages()

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(CompositeDeformDETRHead, self).train(mode)
        self._freeze_stages()

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight
            if self.num_patterns > 1:
                query_embeds = query_embeds.repeat(self.num_patterns, 1) # nq*num_pat, 256
                pat_embed = self.patterns_embed.weight  
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_query, 1).flatten(0, 1) # nq*num_pat, 256
                query_embeds = query_embeds + pat_embed
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, enc_outputs = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_encoder_output=True
            )
        if self.return_enc_intermediate:
            outs = []
            num_level = len(mlvl_feats)
            for enc_feat in enc_outputs:
                start = 0
                layer_outs = []
                for lvl in range(num_level):
                    bs, c, h, w = mlvl_feats[lvl].shape
                    end = start + h*w
                    feat = enc_feat[start:end].permute(1, 2, 0).contiguous()
                    start = end
                    layer_outs.append(feat.reshape(bs, c, h, w))
                layer_outs.append(self.downsample(layer_outs[-1]))
                outs.append(layer_outs)            
        else:
            outs = []
            num_level = len(mlvl_feats)
            start = 0
            for lvl in range(num_level):
                bs, c, h, w = mlvl_feats[lvl].shape
                end = start + h*w
                feat = enc_outputs[start:end].permute(1, 2, 0).contiguous()
                start = end
                outs.append(feat.reshape(bs, c, h, w))
            outs.append(self.downsample(outs[-1]))

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

            #outs = self.query_enhance(outs, hs[-1])
        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid(), mlvl_positional_encodings, outs, hs
        else:
            return outputs_classes, outputs_coords, \
                None, None, mlvl_positional_encodings, outs, hs

    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        anx_anchors, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, init_reference, inter_references = self.transformer.forward_aux(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    anx_anchors,
                    pos_feats=aux_feats,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_encoder_output=True,
                    attn_masks=attn_masks,
                    head_idx=head_idx
            )

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)


            #outs = self.query_enhance(outs, hs[-1])
        return outputs_classes, outputs_coords, \
                None, None


    def loss_single_aux(self,
                    cls_scores,
                    bbox_preds,
                    labels,
                    label_weights,
                    bbox_targets,
                    bbox_weights,
                    img_metas):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)
        labels = labels.reshape(num_imgs * num_q)
        label_weights = label_weights.reshape(num_imgs * num_q)
        bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
        bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)

        bg_class_ind = self.num_classes
        num_total_pos = len(((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs*num_q - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if self.use_labeled_loss:
            labeled_mask = []
            for i in range(num_imgs):
                filename = img_metas[i]['filename']
                labeled_mask.append(self.labeled_mask[img_metas[i]['ori_filename']])
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, labeled_mask, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls*self.lambda_1, loss_bbox*self.lambda_1, loss_iou*self.lambda_1

    def get_aux_targets(self, pos_anchors, img_metas, mlvl_feats, head_idx):
        anchors, labels, targets = pos_anchors[:3]
        bs, c = len(anchors), mlvl_feats[0].shape[1]
        max_num_pos = 0
        all_feats = []
        for i in range(bs):
            label = labels[i]
            feats = [feat[i].reshape(c, -1).transpose(1, 0) for feat in mlvl_feats]
            feats = torch.cat(feats, dim=0)
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)  
            max_num_pos = max(max_num_pos, len(pos_inds))
            all_feats.append(feats)
        max_num_pos = min(self.max_pos_priors, max_num_pos)
        max_num_pos = max(9, max_num_pos)

        if self.use_zero_padding:
            attn_masks = []
            label_weights = anchors[0].new_zeros([bs, max_num_pos])
            bbox_weights = anchors[0].new_zeros([bs, max_num_pos, 4])
        else:
            attn_masks = None
            label_weights = anchors[0].new_ones([bs, max_num_pos])
            bbox_weights = anchors[0].new_zeros([bs, max_num_pos, 4])

        aux_anchors, aux_labels, aux_targets, aux_feats = [], [], [], []

        for i in range(bs):
            anchor, label, target = anchors[i], labels[i], targets[i]
            feats = all_feats[i]
            if not anchor.shape[0]==feats.shape[0]:
                if anchor.shape[0]<=512:
                    feats = pos_anchors[-1][i]
                    num_anchors_per_point = 1
                else:
                    num_anchors_per_point = anchor.shape[0] // feats.shape[0]
            else:
                num_anchors_per_point = anchor.shape[0] // feats.shape[0]
            #feats = all_feats[i]
            feats = feats.unsqueeze(1).repeat(1, num_anchors_per_point, 1)
            feats = feats.reshape(feats.shape[0]*num_anchors_per_point, feats.shape[-1])
            img_meta = img_metas[i]
            img_h, img_w, _ = img_meta['img_shape']
            factor = anchor.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)
            neg_inds = ((label == bg_class_ind)).nonzero().squeeze(1)
            if pos_inds.shape[0] > max_num_pos:
                indices = torch.randperm(pos_inds.shape[0])[:max_num_pos].cuda()
                pos_inds = pos_inds[indices]

            anchor = bbox_xyxy_to_cxcywh(anchor[pos_inds] / factor)
            label = label[pos_inds]
            target = bbox_xyxy_to_cxcywh(target[pos_inds] / factor)

            feat = feats[pos_inds]
            #label_weights[i][:len(label)] = 1 # for empty padding
            if self.use_zero_padding:
                label_weights[i][:len(label)] = 1
                bbox_weights[i][:len(label)] = 1
                attn_mask = torch.zeros([max_num_pos, max_num_pos,]).bool().to(anchor.device)
            else:
                bbox_weights[i][:len(label)] = 1

            if anchor.shape[0] < max_num_pos:
                padding_shape = max_num_pos-anchor.shape[0]
                if self.use_zero_padding:
                    padding_anchor = anchor.new_zeros([padding_shape, 4])
                    padding_label = label.new_ones([padding_shape]) * self.num_classes
                    padding_target = target.new_zeros([padding_shape, 4])
                    padding_feat = feat.new_zeros([padding_shape, c])
                    attn_mask[anchor.shape[0] :, 0 : anchor.shape[0],] = True
                    attn_mask[:, anchor.shape[0] :,] = True
                else:
                    indices = torch.randperm(neg_inds.shape[0])[:padding_shape].cuda()
                    neg_inds = neg_inds[indices]
                    padding_anchor = bbox_xyxy_to_cxcywh(anchors[i][neg_inds] / factor)
                    padding_label = labels[i][neg_inds]
                    padding_target = bbox_xyxy_to_cxcywh(targets[i][neg_inds] / factor)
                    padding_feat = feats[neg_inds]
                anchor = torch.cat((anchor, padding_anchor), dim=0)
                label = torch.cat((label, padding_label), dim=0)
                target = torch.cat((target, padding_target), dim=0)
                feat = torch.cat((feat, padding_feat), dim=0)
            if self.use_zero_padding:
                attn_masks.append(attn_mask.unsqueeze(0))
            aux_anchors.append(anchor.unsqueeze(0))
            aux_labels.append(label.unsqueeze(0))
            aux_targets.append(target.unsqueeze(0))
            aux_feats.append(feat.unsqueeze(0))

        if self.use_zero_padding:
            attn_masks = torch.cat(attn_masks, dim=0).unsqueeze(1).repeat(1, 8, 1, 1)
            attn_masks = attn_masks.reshape(bs*8, max_num_pos, max_num_pos)
        attn_masks = None
        #label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        anx_anchors = torch.cat(aux_anchors, dim=0)
        aux_labels = torch.cat(aux_labels, dim=0)
        aux_targets = torch.cat(aux_targets, dim=0)
        aux_feats = torch.cat(aux_feats, dim=0)
        aux_label_weights = label_weights
        aux_bbox_weights = bbox_weights
        return (anx_anchors, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train_aux(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      pos_anchors=None,
                      head_idx=0,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        aux_targets = self.get_aux_targets(pos_anchors, img_metas, x, head_idx)
        outs = self.forward_aux(x[:4], img_metas, aux_targets, head_idx)
        outs = outs + aux_targets
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss_aux(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss_aux(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds,
             anx_anchors, 
             aux_labels, 
             aux_targets, 
             aux_label_weights, 
             aux_bbox_weights,
             aux_feats,
             attn_masks,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_labels = [aux_labels for _ in range(num_dec_layers)]
        all_label_weights = [aux_label_weights for _ in range(num_dec_layers)]
        all_bbox_targets = [aux_targets for _ in range(num_dec_layers)]
        all_bbox_weights = [aux_bbox_weights for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single_aux, all_cls_scores, all_bbox_preds,
            all_labels, all_label_weights, all_bbox_targets, 
            all_bbox_weights, img_metas_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.

        # loss from the last decoder layer
        loss_dict['loss_cls_aux'] = losses_cls[-1]
        loss_dict['loss_bbox_aux'] = losses_bbox[-1]
        loss_dict['loss_iou_aux'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_aux'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_aux'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou_aux'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses, final_costs = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        mlvl_pos = None
        enc_outputs, hs = outs[-2:]
        mlvl_pos = outs[-3]
        if self.query_attn:
            if self.cost_aware:
                costs = final_costs[-1]
                k_cost, indices = torch.kthvalue(costs, k=int(0.2*self.num_query), dim=-1)
                k_cost = k_cost.unsqueeze(-1).repeat(1, costs.shape[-1])
                attn_mask = torch.where(costs > k_cost, (-1e6)*torch.ones_like(costs), torch.ones_like(costs))
            else:
                attn_mask = None
            enc_outputs = self.query_enhance(enc_outputs, hs.mean(0), attn_mask=attn_mask, mlvl_pos=mlvl_pos)
            #enc_outputs = self.query_enhance(enc_outputs, hs[-1], attn_mask=attn_mask)
            #l, b, n, c = hs.shape
            #enc_outputs = self.query_enhance(enc_outputs, hs.permute(1, 0, 2, 3).reshape(b,l*n,c), attn_mask=attn_mask, mlvl_pos=mlvl_pos)
            #enc_outputs = self.query_enhance(enc_outputs, hs, attn_mask=attn_mask, mlvl_pos=mlvl_pos)
        return losses, enc_outputs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds,
             mlvl_positional_encodings,
             enc_outputs,
             hs,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, costs = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, _ = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict, None

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   mlvl_positional_encodings,
                   enc_outputs,
                   hs,
                   img_metas,
                   rescale=False,
                   with_nms=False,
                   ms_test=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale, with_nms,
                                                ms_test)
            result_list.append(proposals)
        return result_list


    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, cost_result_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if self.use_labeled_loss:
            labeled_mask = []
            for i in range(num_imgs):
                filename = img_metas[i]['filename']
                labeled_mask.append(self.labeled_mask[img_metas[i]['ori_filename']])
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, labeled_mask, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou, cost_result_list

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, cost_result_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, cost_result_list)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                            gt_labels, img_meta,
                                            gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, None)

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False,
                           with_nms=False,
                           ms_test=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        if ms_test:
            max_per_img = self.num_query
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        if with_nms:
            #cfg_nms = dict(type='soft_nms', iou_threshold=0.5)
            #cfg_max_per_img = 100
            det_bboxes, keep_idxs = batched_nms(det_bboxes, scores.unsqueeze(1),
                                                det_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = det_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        if ms_test:
            return det_bboxes, scores.unsqueeze(1), det_labels
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def aug_test_bboxes_simple(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x, img_meta)
            bbox_outputs = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                rescale=False,
                with_nms=False,
                ms_test=True)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            if len(bbox_outputs) >= 3:
                aug_labels.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0) if aug_labels else None

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.test_cfg.nms)
        det_bboxes = det_bboxes[:self.test_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.test_cfg.max_per_img]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

            return [
                (det_bboxes, merged_labels),
            ]


    def merge_aug_vote_results(self, aug_bboxes, aug_labels, img_metas):
        """Merge augmented detection bboxes and labels.
        Args:
            aug_bboxes (list[Tensor]): shape (n, 5)
            aug_labels (list[Tensor] or None): shape (n, )
            img_metas (list[Tensor]): shape (3, ).
        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes[:, :4] = bbox_mapping_back(bboxes[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_labels is None:
            return bboxes
        else:
            labels = torch.cat(aug_labels, dim=0)
            return bboxes, labels

    def remove_boxes(self, boxes, min_scale, max_scale):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        in_range_idxs = torch.nonzero(
            (areas >= min_scale * min_scale) &
            (areas <= max_scale * max_scale),
            as_tuple=False).squeeze(1)
        return in_range_idxs

    def vote_bboxes(self, boxes, scores, vote_thresh=0.76):
        assert self.test_cfg.fusion_cfg.type == 'soft_vote'
        eps = 1e-6
        score_thr = self.test_cfg.score_thr

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy().reshape(-1, 1)
        det = np.concatenate((boxes, scores), axis=1)
        if det.shape[0] <= 1:
            return np.zeros((0, 5)), np.zeros((0, 1))
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = area[0] + area[:] - inter
            union = np.maximum(union, eps)
            o = inter / union
            o[0] = 1

            # get needed merge det and delete these det
            merge_index = np.where(o >= vote_thresh)[0]
            det_accu = det[merge_index, :]
            det_accu_iou = o[merge_index]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except ValueError:
                    dets = det_accu
                continue
            else:
                soft_det_accu = det_accu.copy()
                soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
                soft_index = np.where(soft_det_accu[:, 4] >= score_thr)[0]
                soft_det_accu = soft_det_accu[soft_index, :]

                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                    det_accu[:, -1:], (1, 4))
                max_score = np.max(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(
                    det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = max_score

                if soft_det_accu.shape[0] > 0:
                    det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except ValueError:
                    dets = det_accu_sum

        order = dets[:, 4].ravel().argsort()[::-1]
        dets = dets[order, :]

        boxes = torch.from_numpy(dets[:, :4]).float().cuda()
        scores = torch.from_numpy(dets[:, 4]).float().cuda()

        return boxes, scores

    def aug_test_bboxes_vote(self, feats, img_metas, rescale=False):
        """Test det bboxes with soft-voting test-time augmentation.
        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[ndarray]: bbox results of each class
        """
        scale_ranges = self.test_cfg.fusion_cfg.scale_ranges
        num_same_scale_tta = len(feats) // len(scale_ranges)
        aug_bboxes = []
        aug_labels = []
        for aug_idx, (x, img_meta) in enumerate(zip(feats, img_metas)):
            # only one image in the batch
            outs = self.forward(x, img_meta)
            det_bboxes, det_scores, det_labels = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                rescale=False,
                with_nms=False,
                ms_test=True)[0]
            det_bboxes = torch.cat((det_bboxes, det_scores), dim=-1)
            min_scale, max_scale = scale_ranges[aug_idx // num_same_scale_tta]
            in_range_idxs = self.remove_boxes(det_bboxes, min_scale, max_scale)
            det_bboxes = det_bboxes[in_range_idxs, :]
            det_labels = det_labels[in_range_idxs]
            aug_bboxes.append(det_bboxes)
            aug_labels.append(det_labels)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_labels = self.merge_aug_vote_results(
            aug_bboxes, aug_labels, img_metas)

        det_bboxes = []
        det_labels = []
        for j in range(self.num_classes):
            inds = (merged_labels == j).nonzero(as_tuple=False).squeeze(1)

            scores_j = merged_bboxes[inds, 4]
            bboxes_j = merged_bboxes[inds, :4].view(-1, 4)
            bboxes_j, scores_j = self.vote_bboxes(bboxes_j, scores_j, self.test_cfg.fusion_cfg.vote_thresh)

            if len(bboxes_j) > 0:
                det_bboxes.append(
                    torch.cat([bboxes_j, scores_j[:, None]], dim=1))
                det_labels.append(
                    torch.full((bboxes_j.shape[0], ),
                               j,
                               dtype=torch.int64,
                               device=scores_j.device))

        if len(det_bboxes) > 0:
            det_bboxes = torch.cat(det_bboxes, dim=0)
            det_labels = torch.cat(det_labels)
        else:
            det_bboxes = merged_bboxes.new_zeros((0, 5))
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

        max_per_img = self.test_cfg.max_per_img
        if det_bboxes.shape[0] > max_per_img > 0:
            cls_scores = det_bboxes[:, 4]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), det_bboxes.shape[0] - max_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test-time augmentation.
        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[ndarray]: bbox results of each class
        """
        fusion_cfg = self.test_cfg.get('fusion_cfg', None)
        fusion_method = fusion_cfg.type if fusion_cfg else 'simple'
        if fusion_method == 'simple':
            return self.aug_test_bboxes_simple(feats, img_metas, rescale)
        elif fusion_method == 'soft_vote':
            return self.aug_test_bboxes_vote(feats, img_metas, rescale)
        else:
            raise ValueError('Unknown TTA fusion method')
