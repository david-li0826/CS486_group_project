import torch.utils.data
import torchvision

import config
import utils
import transforms as T
from faster_rcnn import create_model
from engine import train_one_epoch, evaluate
from coco import CocoSet
from openimages import OpenSet


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_transforms():
    transform = [T.ToTensor(), T.RandomHorizontalFlip(0.5)]
    return T.Compose(transform)


def train_coco():
    dataset = CocoSet(
        root=config.coco_train_data_dir,
        annotation=config.coco_train_ann_file,
        transforms=get_transforms()
    )

    print(dataset[0])

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_test = torch.utils.data.Subset(dataset, indices[-60:])
    dataset = torch.utils.data.Subset(dataset, indices[:200])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=utils.collate_fn
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers_dl,
        collate_fn=utils.collate_fn
    )

    print('Number of samples: ', len(dataset))

    # get model
    model = create_model(config.coco_num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # training
    for epoch in range(config.num_epochs):
        print(f'Epoch: {epoch}/{config.num_epochs}')
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=dataloader,
            device=device,
            epoch=epoch,
            print_freq=50
        )
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(
            model=model,
            data_loader=dataloader_test,
            device=device
        )


def train_open():
    dataset = OpenSet(
        root=config.open_validation_data_dir,
        annotation=config.open_validation_ann_file,
        transforms=get_transforms()
    )

    print(dataset[0])

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_test = torch.utils.data.Subset(dataset, indices[-60:])
    dataset = torch.utils.data.Subset(dataset, indices[:200])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=utils.collate_fn
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers_dl,
        collate_fn=utils.collate_fn
    )

    print('Number of samples: ', len(dataset))

    # get model
    model = create_model(config.open_num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # training
    for epoch in range(config.num_epochs):
        print(f'Epoch: {epoch}/{config.num_epochs}')
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=dataloader,
            device=device,
            epoch=epoch,
            print_freq=50
        )
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(
            model=model,
            data_loader=dataloader_test,
            device=device
        )


if __name__ == '__main__':
    train_coco()
