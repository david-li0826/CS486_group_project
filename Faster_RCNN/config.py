# path to your own data and coco file
data_dir = 'cocodata'
train_data_dir = 'cocodata/images/train2017'
train_ann_file = "{}/annotations/instances_train2017.json".format(data_dir)
validation_data_dir = 'cocodata/images/train2017'

# Batch size
train_batch_size = 1

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

#
num_classes = 81
num_epochs = 5

lr = 0.005
momentum = 0.9
weight_decay = 0.005
