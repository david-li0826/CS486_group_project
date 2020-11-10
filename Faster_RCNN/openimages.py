import os
import torch
import torch.utils.data
import pandas as pd
import collections
import time
from tqdm import tqdm
from PIL import Image


class OpenSet(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.open = collections.defaultdict(dict)
        self.ids = []
        if annotation is not None:
            print('loading annotations into memory...')
            tic = time.time()
            self.dataset = pd.read_csv(os.path.join(annotation))
            self.dataset = self.dataset.drop_duplicates('ImageID', keep='last').reset_index(drop=True)
            for index, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
                image_id = row['ImageID']
                if image_id in self.open:
                    ann = {
                        'bbox': [row['XMin'], row['XMax'], row['YMin'], row['YMax']],
                        'label': row['LabelName']
                    }
                    self.open[image_id]['annotations'].append(ann)
                else:
                    image_path = os.path.join(self.root, image_id)
                    if os.path.exists(image_path + '.jpg'):
                        ann = {
                            'bbox': [row['XMin'], row['XMax'], row['YMin'], row['YMax']],
                            'label': row['LabelName']
                        }
                        self.open[image_id]['image_path'] = image_path + '.jpg'
                        self.open[image_id]['annotations'] = [ann]
                        self.ids.append(image_id)
            self.ids = sorted(self.ids)
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def __getitem__(self, index):
        open = self.open
        # Image ID
        img_id = self.ids[index]
        img_path = self.open[img_id]['image_path']
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        # number of objects
        num_object = len(open[img_id]['annotations'])

        # bounding boxes
        boxes = []
        for i in range(num_object):
            xmin = open[img_id]['annotations']['bbox'][0] * width
            xmax = open[img_id]['annotations']['bbox'][1] * width
            ymin = open[img_id]['annotations']['bbox'][2] * height
            ymax = open[img_id]['annotations']['bbox'][3] * height
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels
        labels = []
        for i in range(num_object):
            labels.append(open[img_id]['annotations']['label'])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # tensor img_id
        img_id = torch.tensor([img_id])
        # size of bbox
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # iscrowd
        iscrowd = torch.zeros((num_object,), dtype=torch.int64)

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
