import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .single_stage import SingleStageDetector_kd


@DETECTORS.register_module()
class RetinaNet_kd(SingleStageDetector_kd):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_kd,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet_kd, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained,init_cfg)
        '''
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        '''
        # import pdb;pdb.set_trace()
        self.backbone = build_backbone(backbone)

        start_idx = 0

        if neck is not None:
            self.neck = build_neck(neck)

        # if query_head is not None:
        #     query_head.update(train_cfg=train_cfg[start_idx] if (
        #                 train_cfg is not None and train_cfg[start_idx] is not None) else None)
        #     query_head.update(test_cfg=test_cfg[start_idx])
        #     self.query_head = build_head(query_head)
        #     self.query_head.init_weights()
        #     start_idx += 1

        # if rpn_head is not None:
        #     rpn_train_cfg = train_cfg[start_idx].rpn if (
        #                 train_cfg is not None and train_cfg[start_idx] is not None) else None
        #     rpn_head_ = rpn_head.copy()
        #     rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[start_idx].rpn)
        #     self.rpn_head = build_head(rpn_head_)
        #     self.rpn_head.init_weights()
        #
        # self.roi_head = nn.ModuleList()
        # for i in range(len(roi_head)):
        #     # update train and test cfg here for now
        #     # TODO: refactor assigner & sampler
        #     if roi_head[i]:
        #         rcnn_train_cfg = train_cfg[i + start_idx].rcnn if (
        #                     train_cfg and train_cfg[i + start_idx] is not None) else None
        #         roi_head[i].update(train_cfg=rcnn_train_cfg)
        #         roi_head[i].update(test_cfg=test_cfg[i + start_idx].rcnn)
        #         # roi_head[i].pretrained = pretrained[i+start_idx] if pretrained and pretrained[i+start_idx] else None
        #         self.roi_head.append(build_head(roi_head[i]))
        #         self.roi_head[-1].init_weights()
        #         self.roi_head[-1].co_training = True
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = nn.ModuleList()
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.init_weights()

        bbox_head_kd.update(train_cfg=train_cfg)
        bbox_head_kd.update(test_cfg=test_cfg)
        self.bbox_head_kd = nn.ModuleList()
        self.bbox_head_kd = build_head(bbox_head_kd)
        self.bbox_head_kd.init_weights()




        self.start_idx = start_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head) > 0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head) > 0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head) > 0))


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head[0].forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      tea_bboxes,
                      tea_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        # x = self.extract_feat(img)

        losses = dict()

        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        # import pdb;pdb.set_trace()
        # x = [feat.float() for feat in x]
        bbox_losses = super(RetinaNet_kd,self).forward_train(img, img_metas, gt_bboxes, gt_labels, tea_bboxes, tea_labels, gt_bboxes_ignore)
        bbox_losses = upd_loss(bbox_losses, idx=0)
        losses.update(bbox_losses)

        check = False
        for bbox in tea_bboxes:
            if bbox.nelement() == 0:
                check = True
                break
        if check:
            bbox_losses_kd = super(RetinaNet_kd,self).forward_train_kd(img, img_metas, gt_bboxes, gt_labels, gt_bboxes, gt_labels, gt_bboxes_ignore)
        else:
            bbox_losses_kd = super(RetinaNet_kd,self).forward_train_kd(img, img_metas, gt_bboxes, gt_labels, tea_bboxes, tea_labels, gt_bboxes_ignore)
        bbox_losses_kd = upd_loss(bbox_losses_kd, idx= 1)
        losses.update(bbox_losses_kd)

        # import pdb;pdb.set_trace()
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-2]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img, img_metas)
        if self.use_dc5:
            x = x[-1:]
        results_list = self.query_head.simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # return self.simple_test_query_head(img, img_metas, proposals, rescale)
        # return self.simple_test_bbox_head(img, img_metas, proposals, rescale)
        # return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        if self.with_query_head:
            return self.simple_test_query_head(img, img_metas, proposals, rescale)
        if self.with_roi_head:
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_bbox_head(img, img_metas, proposals, rescale)

    def aug_test_query_head(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        '''
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        '''
        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        return self.aug_test_query_head(imgs, img_metas, rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'
                # noqa E501
            )