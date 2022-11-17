## Top-Down Bottom-Up Attention on top of Swin Transformer
 
Implementation of top-down bottom-up attentional modules on top of [Swin Transformer object detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) network.

### Installation:

- Follow official doc from [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).
- Download the model from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

### How to run
#### Test
```shell
python -m swin_td_bu_att demo.jpg configs/td_bu_attention/topdown_bottomup_attentional_swin.py swin_tiny_patch4_window7_224.pth --device cuda --out-file result.jpg
```
#### Train
```shell
python swin_td_bu_att/train.py
```

### Current implementation progress:

- [x] Implement modules
- [x] Implement functioning test code
- [x] Fix training parameters in [config file](configs/td_bu_attention/topdown_bottomup_attentional_swin.py)
- [x] Implement missing training functionality
- [ ] Decrease the training loss