# path to your own data and coco file
data_dir_coco = 'cocodata'
coco_train_data_dir = 'cocodata/images/train2017'
coco_train_ann_file = 'cocodata/annotations/instances_train2017.json'
coco_validation_data_dir = 'cocodata/images/train2017'

# path to open images file
data_dir_open = 'openimagesdata'
open_validation_data_dir = 'openimagesdata/validation'
open_validation_ann_file = 'openimagesdata/annotations/validation-annotations-bbox.csv'

# Batch size
train_batch_size = 16

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training
es_patience = 3
detection_threshold = 0.5

# coco parameters
coco_num_classes = 101
coco_model_path = 'models\FasterRCNN_coco'

# open images parameters
open_num_classes = 601
open_model_path = 'models\FasterRCNN_openimages'
open_label_col = 'LabelName'

num_epochs = 10
lr = 0.005
momentum = 0.9
weight_decay = 0.005
