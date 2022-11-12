# Copyright (c) OpenMMLab. All rights reserved.
import sys

import torch
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        multiclass_nms)

if sys.version_info >= (3, 7):
	from mmdet.utils.contextmanagers import completed


class TDBUTestMixin:
	if sys.version_info >= (3, 7):

		async def async_test_bboxes(self,
		                            x,
		                            img_metas,
		                            proposals,
		                            rcnn_test_cfg,
		                            rescale=False,
		                            **kwargs):
			"""Asynchronized test for box head without augmentation."""
			rois = bbox2roi(proposals)
			roi_feats = self.bbox_roi_extractor(
				x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
			if self.with_shared_head:
				roi_feats = self.shared_head(roi_feats)
			sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

			async with completed(
					__name__, 'bbox_head_forward',
					sleep_interval=sleep_interval):
				cls_score, bbox_pred = self.bbox_head(roi_feats)

			img_shape = img_metas[0]['img_shape']
			scale_factor = img_metas[0]['scale_factor']
			det_bboxes, det_labels = self.bbox_head.get_bboxes(
				rois,
				cls_score,
				bbox_pred,
				img_shape,
				scale_factor,
				rescale=rescale,
				cfg=rcnn_test_cfg)
			return det_bboxes, det_labels

	def simple_test_bboxes(self,
	                       x,
	                       img_metas,
	                       proposals,
	                       rcnn_test_cfg,
	                       rescale=False):
		"""Test only det bboxes without augmentation.

		Args:
			x (tuple[Tensor]): Feature maps of all scale level.
			img_metas (list[dict]): Image meta info.
			proposals (List[Tensor]): Region proposals.
			rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
			rescale (bool): If True, return boxes in original image space.
				Default: False.

		Returns:
			tuple[list[Tensor], list[Tensor]]: The first list contains
				the boxes of the corresponding image in a batch, each
				tensor has the shape (num_boxes, 5) and last dimension
				5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
				in the second list is the labels with shape (num_boxes, ).
				The length of both lists should be equal to batch_size.
		"""

		rois = bbox2roi(proposals)

		if rois.shape[0] == 0:
			batch_size = len(proposals)
			det_bbox = rois.new_zeros(0, 5)
			det_label = rois.new_zeros((0,), dtype=torch.long)
			if rcnn_test_cfg is None:
				det_bbox = det_bbox[:, :4]
				det_label = rois.new_zeros(
					(0, self.bbox_head.fc_cls.out_features))
			# There is no proposal in the whole batch
			return [det_bbox] * batch_size, [det_label] * batch_size

		bbox_results = self._bbox_forward(x, rois)
		bbox_results.pop("bbox_feats")
		return bbox_results  # TODO ?
		img_shapes = tuple(meta['img_shape'] for meta in img_metas)
		scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

		# split batch bbox prediction back to each image
		cls_score = bbox_results['cls_score']
		bbox_pred = bbox_results['bbox_pred']
		sp_att_score = bbox_results['sp_att_score']
		att_score = bbox_results['att_score']
		return cls_score, bbox_pred, sp_att_score, att_score

		num_proposals_per_img = tuple(len(p) for p in proposals)
		rois = rois.split(num_proposals_per_img, 0)
		cls_score = cls_score.split(num_proposals_per_img, 0)

		# some detector with_reg is False, bbox_pred will be None
		if bbox_pred is not None:
			# TODO move this to a sabl_roi_head
			# the bbox prediction of some detectors like SABL is not Tensor
			if isinstance(bbox_pred, torch.Tensor):
				bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
			else:
				bbox_pred = self.bbox_head.bbox_pred_split(
					bbox_pred, num_proposals_per_img)
		else:
			bbox_pred = (None,) * len(proposals)

		# apply bbox post-processing to each image individually
		det_bboxes = []
		det_labels = []
		for i in range(len(proposals)):
			if rois[i].shape[0] == 0:
				# There is no proposal in the single image
				det_bbox = rois[i].new_zeros(0, 5)
				det_label = rois[i].new_zeros((0,), dtype=torch.long)
				if rcnn_test_cfg is None:
					det_bbox = det_bbox[:, :4]
					det_label = rois[i].new_zeros(
						(0, self.bbox_head.fc_cls.out_features))

			else:
				det_bbox, det_label = self.bbox_head.get_bboxes(
					rois[i],
					cls_score[i],
					bbox_pred[i],
					img_shapes[i],
					scale_factors[i],
					rescale=rescale,
					cfg=rcnn_test_cfg)
			det_bboxes.append(det_bbox)
			det_labels.append(det_label)
		return det_bboxes, det_labels

	def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
		"""Test det bboxes with test time augmentation."""
		aug_bboxes = []
		aug_scores = []
		for x, img_meta in zip(feats, img_metas):
			# only one image in the batch
			img_shape = img_meta[0]['img_shape']
			scale_factor = img_meta[0]['scale_factor']
			flip = img_meta[0]['flip']
			flip_direction = img_meta[0]['flip_direction']
			# TODO more flexible
			proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
			                         scale_factor, flip, flip_direction)
			rois = bbox2roi([proposals])
			bbox_results = self._bbox_forward(x, rois)
			bboxes, scores = self.bbox_head.get_bboxes(
				rois,
				bbox_results['cls_score'],
				bbox_results['bbox_pred'],
				img_shape,
				scale_factor,
				rescale=False,
				cfg=None)
			aug_bboxes.append(bboxes)
			aug_scores.append(scores)
		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes, merged_scores = merge_aug_bboxes(
			aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
		if merged_bboxes.shape[0] == 0:
			# There is no proposal in the single image
			det_bboxes = merged_bboxes.new_zeros(0, 5)
			det_labels = merged_bboxes.new_zeros((0,), dtype=torch.long)
		else:
			det_bboxes, det_labels = multiclass_nms(merged_bboxes,
			                                        merged_scores,
			                                        rcnn_test_cfg.score_thr,
			                                        rcnn_test_cfg.nms,
			                                        rcnn_test_cfg.max_per_img)
		return det_bboxes, det_labels
