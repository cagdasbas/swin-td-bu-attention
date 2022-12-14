# model settings
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = "output"
gpu_ids = [0]
seed = 3
device = "cuda"
auto_scale_lr = dict(enable=False, base_batch_size=192)
model = dict(
	type='FasterRCNN',
	backbone=dict(
		type='SwinTransformer',
		embed_dims=96,
		depths=[2, 2, 6, 2],
		num_heads=[3, 6, 12, 24],
		window_size=7,
		mlp_ratio=4,
		qkv_bias=True,
		qk_scale=None,
		drop_rate=0.,
		attn_drop_rate=0.,
		drop_path_rate=0.2,
		patch_norm=True,
		out_indices=(0, 1, 2, 3),
		with_cp=False,
		convert_weights=True,
		init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
	neck=dict(
		type='FPN',  # RCNN_base
		in_channels=[96, 192, 384, 768],
		out_channels=256,
		num_outs=5),
	rpn_head=dict(
		type='RPNHead',  # RCNN_rpn
		in_channels=256,
		feat_channels=256,
		anchor_generator=dict(
			type='AnchorGenerator',
			scales=[8],
			ratios=[0.5, 1.0, 2.0],
			strides=[4, 8, 16, 32, 64]),
		bbox_coder=dict(
			type='DeltaXYWHBBoxCoder',
			target_means=[.0, .0, .0, .0],
			target_stds=[1.0, 1.0, 1.0, 1.0])
	),
	roi_head=dict(
		type='TDBURoIHead',
		bbox_roi_extractor=dict(
			type='SingleRoIExtractor',  # roi_align
			roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[4, 8, 16, 32]),
		bbox_head=dict(
			type='Shared2TopDownBottomUpAttentionHead',  # head_to_tail? RCNN_bbox_pred?
			in_channels=256,
			fc_out_channels=1024,
			roi_feat_size=7,
			num_classes=40,
			bbox_coder=dict(
				type='DeltaXYWHBBoxCoder',  # RCNN_bbox_pred
				target_means=[0., 0., 0., 0.],
				target_stds=[0.1, 0.1, 0.2, 0.2]),
			reg_class_agnostic=False,
			loss_cls=dict(
				type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
		),
	),
	# model training and testing settings
	train_cfg=dict(
		rpn=dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.7,
				neg_iou_thr=0.3,
				min_pos_iou=0.3,
				match_low_quality=True,
				ignore_iof_thr=-1),
			sampler=dict(
				type='RandomSampler',
				num=256,
				pos_fraction=0.5,
				neg_pos_ub=-1,
				add_gt_as_proposals=False),
			allowed_border=-1,
			pos_weight=-1,
			debug=False),
		rpn_proposal=dict(
			nms_pre=2000,
			max_per_img=1000,
			nms=dict(type='nms', iou_threshold=0.7),
			min_bbox_size=0),
		rcnn=dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.5,
				neg_iou_thr=0.5,
				min_pos_iou=0.5,
				match_low_quality=False,
				ignore_iof_thr=-1),
			sampler=dict(
				type='RandomSampler',
				num=512,
				pos_fraction=0.25,
				neg_pos_ub=-1,
				add_gt_as_proposals=True),
			pos_weight=-1,
			debug=False)),
	test_cfg=dict(
		rpn=dict(
			nms_pre=1000,
			max_per_img=1000,
			nms=dict(type='nms', iou_threshold=0.7),
			min_bbox_size=0),
		rcnn=dict(
			score_thr=0.05,
			nms=dict(type='nms', iou_threshold=0.5),
			max_per_img=100)
		# soft-nms is also supported for rcnn testing
		# e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
	))
dataset_type = 'Stanford40Dataset'
data_root = '/var/home/cagdas/storage/dataset/Stanford40/'
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(
		type='Expand',
		mean=[123.675, 116.28, 103.53],
		to_rgb=True,
		ratio_range=(1, 2)),
	dict(
		type='MinIoURandomCrop',
		min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
		min_crop_size=0.3),
	dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='PhotoMetricDistortion'),
	dict(
		type='Normalize',
		mean=[123.675, 116.28, 103.53],
		std=[58.395, 57.12, 57.375],
		to_rgb=True),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(320, 320),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(type='RandomFlip'),
			dict(
				type='Normalize',
				mean=[123.675, 116.28, 103.53],
				std=[58.395, 57.12, 57.375],
				to_rgb=True),
			dict(type='Pad', size_divisor=32),
			dict(type='DefaultFormatBundle'),
			dict(type='Collect', keys=['img'])
		])
]
data = dict(
	samples_per_gpu=24,
	workers_per_gpu=4,
	train=dict(
		type='RepeatDataset',
		times=10,
		dataset=dict(
			type='Stanford40Dataset',
			ann_file='/var/home/cagdas/storage/dataset/Stanford40/COCOAnnotations/instances_train.json',
			img_prefix='/var/home/cagdas/storage/dataset/Stanford40/JPEGImages',
			pipeline=[
				dict(type='LoadImageFromFile'),
				dict(type='LoadAnnotations', with_bbox=True),
				dict(
					type='Expand',
					mean=[123.675, 116.28, 103.53],
					to_rgb=True,
					ratio_range=(1, 2)),
				dict(
					type='MinIoURandomCrop',
					min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
					min_crop_size=0.3),
				dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
				dict(type='RandomFlip', flip_ratio=0.5),
				dict(type='PhotoMetricDistortion'),
				dict(
					type='Normalize',
					mean=[123.675, 116.28, 103.53],
					std=[58.395, 57.12, 57.375],
					to_rgb=True),
				dict(type='Pad', size_divisor=32),
				dict(type='DefaultFormatBundle'),
				dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
			])),
	val=dict(
		type='Stanford40Dataset',
		ann_file='/var/home/cagdas/storage/dataset/Stanford40/COCOAnnotations/instances_test.json',
		img_prefix='/var/home/cagdas/storage/dataset/Stanford40/JPEGImages',
		pipeline=[
			dict(type='LoadImageFromFile'),
			dict(
				type='MultiScaleFlipAug',
				img_scale=(320, 320),
				flip=False,
				transforms=[
					dict(type='Resize', keep_ratio=True),
					dict(type='RandomFlip'),
					dict(
						type='Normalize',
						mean=[123.675, 116.28, 103.53],
						std=[58.395, 57.12, 57.375],
						to_rgb=True),
					dict(type='Pad', size_divisor=32),
					dict(type='DefaultFormatBundle'),
					dict(type='Collect', keys=['img'])
				])
		]),
	test=dict(
		type='Stanford40Dataset',
		ann_file='/var/home/cagdas/storage/dataset/Stanford40/COCOAnnotations/instances_test.json',
		img_prefix='/var/home/cagdas/storage/dataset/Stanford40/JPEGImages',
		pipeline=[
			dict(type='LoadImageFromFile'),
			dict(
				type='MultiScaleFlipAug',
				img_scale=(320, 320),
				flip=False,
				transforms=[
					dict(type='Resize', keep_ratio=True),
					dict(type='RandomFlip'),
					dict(
						type='Normalize',
						mean=[123.675, 116.28, 103.53],
						std=[58.395, 57.12, 57.375],
						to_rgb=True),
					dict(type='Pad', size_divisor=32),
					dict(type='DefaultFormatBundle'),
					dict(type='Collect', keys=['img'])
				])
		]))
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
	policy='step',
	warmup='linear',
	warmup_iters=4000,
	warmup_ratio=0.0001,
	step=[24, 28])
runner = dict(type='EpochBasedRunner', max_epochs=5)
evaluation = dict(interval=1, start=200, metric='accuracy', metric_options={'topk': (1,)})
find_unused_parameters = True
