import torch.utils.data
import torchvision

import config
import utils
import transforms as T
from faster_rcnn import create_model
from engine import train_one_epoch, evaluate
from coco import CocoSet


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_transforms():
    transform = [T.ToTensor(), T.RandomHorizontalFlip(0.5)]
    return T.Compose(transform)


def train():
    dataset = CocoSet(
        root=config.train_data_dir,
        annotation=config.train_ann_file,
        transforms=get_transforms()
    )

    print(dataset[0])

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:500])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=utils.collate_fn
    )

    print('Number of samples: ', len(dataset))

    # get model
    model = create_model(config.num_classes)

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
        # model.train()
        # i = 0
        # for imgs, annotations in dataloader:
        #     i += 1
        #     imgs = list(img.to(device) for img in imgs)
        #     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        #     loss_dict = model(imgs, annotations)
        #     losses = sum(loss for loss in loss_dict.values())
        #
        #     loss_value = losses.item()
        #
        #     if not math.isfinite(loss_value):
        #         print("Loss is {}, stopping training".format(loss_value))
        #         print(loss_dict)
        #         sys.exit(1)
        #
        #     optimizer.zero_grad()
        #     losses.backward()
        #     optimizer.step()
        #
        #     print(f'Iteration: {i}/{len(dataloader)}, Loss: {losses}')
        #
        lr_scheduler.step()


if __name__ == '__main__':
    train()
