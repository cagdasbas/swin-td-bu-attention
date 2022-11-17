import os

import numpy as np
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

from swin_td_bu_att.release import __release__

print(f"release ver: {__release__}")

dataset_root = "/var/home/cagdas/storage/dataset/Stanford40/"
image_folder = os.path.join(dataset_root, "JPEGImages")
split_folder = os.path.join(dataset_root, "ImageSplits")
mat_anno_folder = os.path.join(dataset_root, "XMLAnnotations")
out_folder = os.path.join(dataset_root, "COCOAnnotations")
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(split_folder, "actions.txt"), "r") as file:
	actions = [line.split("\t")[0] for line in file.readlines()]

actions.pop(0)

id_to_action = {index + 1: value for index, value in enumerate(actions)}
action_to_id = {value: index for index, value in id_to_action.items()}

set_names = ["train", "test"]

"demo.jpg topdown_bottomup_attentional_swin.py swin_tiny_patch4_window7_224.pth --device cuda --out-file result.jpg"

model = init_detector(
	"/var/home/cagdas/storage/workspace/swin-object-detection/configs/td_bu_attention/topdown_bottomup_attentional_swin.py",
	"/var/home/cagdas/storage/workspace/swin-object-detection/output/epoch_5.pth", device="cuda:0")

compared = []
image_id = 1
anno_id = 1
for action in tqdm(actions):
	actual = action_to_id[action]
	with open(os.path.join(split_folder, action + "_test.txt"), "r") as file:
		images = [line.replace("\n", "").replace("\t", "") for line in file.readlines()]

	for image in images:
		image_path = os.path.join(image_folder, image)
		result = inference_detector(model, image_path)
		# show the results
		predicted = np.argmax(result['cls_score'].cpu().numpy())
		compared.append(predicted == actual)

	print(f"acc: {sum(compared) / len(compared)}")

print(f"acc: {sum(compared) / len(compared)}")
