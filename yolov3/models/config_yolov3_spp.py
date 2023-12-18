# coding=gbk
from easydict import EasyDict
import yaml
import torch.nn as nn


# def process_yaml(yaml_data, opt):
#     if isinstance(yaml_data, dict):
#         for key, value in yaml_data.items():
#             if isinstance(value, dict):
#                 # 递归处理嵌套的字典
#                 process_yaml(value, opt)
#             else:
#                 # 将键值对存储到 opt 字典中
#                 opt[key] = value

opt = EasyDict()

opt.cfg = r"E:\Project\atom_sunshine\OpenSSLab\object_detection\yolov3\yolov3\models\yolov3-spp.yaml" # Configuration 配置信息

# with open(opt.cfg,"r") as yaml_file:
#     yaml_data = yaml.safe_load(yaml_file)

# process_yaml(yaml_data, opt)


opt.anchors = [[10,13, 16,30, 33,23],  # P3/8
               [30,61, 62,45, 59,119],  # P4/16
               [116,90, 156,198, 373,326]]  # P5/32


# darknet53 backbone
# [from, number, module, args]
opt.backbone =  [[[-1, 1, "Conv", [32, 3, 1]],  # 0
   [-1, 1, "Conv", [64, 3, 2]],  # 1-P1/2
   [-1, 1, "Bottleneck", [64]],
   [-1, 1, "Conv", [128, 3, 2]],  # 3-P2/4
   [-1, 2, "Bottleneck", [128]],
   [-1, 1, "Conv", [256, 3, 2]],  # 5-P3/8
   [-1, 8, "Bottleneck", [256]],
   [-1, 1, "Conv", [512, 3, 2]],  # 7-P4/16
   [-1, 8, "Bottleneck", [512]],
   [-1, 1, "Conv", [1024, 3, 2]],  # 9-P5/32
   [-1, 4, "Bottleneck", [1024]]  # 10
  ]]


# YOLOv3-SPP head
opt.head = [[[-1, 1, "Bottleneck", [1024, False]],
   [-1, 1, "SPP", [512, [5, 9, 13]]],
   [-1, 1, "Conv", [1024, 3, 1]],
   [-1, 1, "Conv", [512, 1, 1]],
   [-1, 1, "Conv", [1024, 3, 1]],  # 15 (P5/32-large)

   [-2, 1, "Conv", [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 8], 1, 'Concat', [1]],  # cat backbone P4
   [-1, 1, 'Bottleneck', [512, False]],
   [-1, 1, 'Bottleneck', [512, False]],
   [-1, 1, 'Conv', [256, 1, 1]],
   [-1, 1, 'Conv', [512, 3, 1]],  # 22 (P4/16-medium)

   [-2, 1, 'Conv', [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, 'Concat', [1]],  # cat backbone P3
   [-1, 1, 'Bottleneck', [256, False]],
   [-1, 2, 'Bottleneck', [256, False]],  # 27 (P3/8-small)

   [[27, 22, 15], 1, 'Detect', ['nc', 'anchors']],   # Detect(P3, P4, P5)
  ]]
opt.batch_size = 1
opt.device = "cpu"
opt.nc = 80
opt.depth_multiple = 1.0  # model depth multiple
opt.width_multiple = 1.0  # layer channel multiple

# print("opt:",opt)
