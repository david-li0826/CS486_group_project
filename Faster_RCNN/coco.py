import os
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO


class CocoSet(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        # number of objects in the image
        num_objs = len(coco_annotation)

        if num_objs == 0:
            print("background images")
            width, height = img.size
            boxes = torch.as_tensor([[0, 0, width, height]])
            labels = torch.ones((1,), dtype=torch.int64)
            # Tensorise img_id
            img_id = torch.tensor([img_id])
            areas =(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            boxes = []
            for i in range(num_objs):
                xmin = coco_annotation[i]["bbox"][0]
                ymin = coco_annotation[i]["bbox"][1]
                xmax = xmin + coco_annotation[i]["bbox"][2]
                ymax = ymin + coco_annotation[i]["bbox"][3]
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # Tensorise img_id
            img_id = torch.tensor([img_id])
            # Size of bbox (Rectangular)
            areas = []
            for i in range(num_objs):
                areas.append(coco_annotation[i]["area"])
            areas = torch.as_tensor(areas, dtype=torch.float32)
            # Iscrowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
