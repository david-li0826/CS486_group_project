import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def create_model(num_classes, min_size=300, max_size=500):
  # use resnet 101 as the backbone
  resnet_net = torchvision.models.resnet101(pretrained=True)
  modules = list(resnet_net.children())[:-1]
  backbone = nn.Sequential(*modules)
  backbone.out_channels = 2048

  anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                     output_size=7,
                                                     sampling_ratio=2)

  ft_model = FasterRCNN(backbone=backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler)