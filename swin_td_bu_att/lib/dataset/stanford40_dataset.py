import os

import numpy as np
import xmltodict
from mmdet.datasets import DATASETS, CustomDataset

dataset_root = "/var/home/cagdas/storage/dataset/Stanford40/"
split_folder = os.path.join(dataset_root, "ImageSplits")

with open(os.path.join(split_folder, "actions.txt"), "r") as file:
	actions = [line.split("\t")[0] for line in file.readlines()]

actions.pop(0)


@DATASETS.register_module()
class Stanford40Dataset(CustomDataset):
	CLASSES = actions

	def load_annotations(self, ann_file):
		action2label = {k: i for i, k in enumerate(self.CLASSES)}
		# load image list from file

		data_infos = []
		mat_anno_folder = os.path.join(dataset_root, "XMLAnnotations")

		set_name = os.path.split(ann_file)[-1].split("_")[1].split(".")[0]

		for action in self.CLASSES:
			with open(os.path.join(split_folder, action + "_" + set_name + ".txt"), "r") as file:
				images = [line.replace("\n", "").replace("\t", "") for line in file.readlines()]

			for image in images:
				with open(os.path.join(mat_anno_folder, image[:-4] + ".xml"), "r") as file:
					annotation = xmltodict.parse(file.read())

				height = int(annotation['annotation']['size']['height'])
				width = int(annotation['annotation']['size']['width'])
				bndbox = annotation['annotation']['object']['bndbox']
				bndbox = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']) - int(bndbox['xmin']),
				          int(bndbox['ymax']) - int(bndbox['ymin'])]

				data_info = dict(filename=image, width=width, height=height)
				data_anno = dict(
					bboxes=np.array([bndbox], dtype=np.float32).reshape(-1, 4),
					labels=np.array([action2label[action]], dtype=np.long),
					bboxes_ignore=np.array([],
					                       dtype=np.float32).reshape(-1, 4),
					labels_ignore=np.array([], dtype=np.long))

				data_info.update(ann=data_anno)
				data_infos.append(data_info)

		return data_infos
