import os
import sys
import yaml
import torch
import random
import logging
import numpy as np
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from eval import eval_net
from utils.dataset import BasicDataset, collate_fn
from utils.utils import split_train_val_test, batch, K_FOLD
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, CLAHE, RandomRotate90, ElasticTransform, RandomGamma,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomSizedCrop, PadIfNeeded,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ToGray, Resize, Normalize
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_net(net,
              train_loader,
              val_loader,
              device,
              dir_checkpoint,
              epochs=5,
              batch_size=1,
              save_cp=True,
              pos_w=0.5,
              lr=0.0001
              ):
    global_step = 0
    if args.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        pos_weight = torch.tensor([pos_w]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = {
        "adam": lambda: optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-8, eps=1e-08),
        "rmsprop": lambda: optim.RMSprop(net.parameters(), lr=lr, momentum=0.9, eps=0.001, weight_decay=1e-8)
    }[args.optim]()

    scheduler = lr_scheduler.LambdaLR(optimizer,
                                      lambda x: (((1 + np.cos(x * np.pi / args.epochs)) / 2) ** 1.0) * 0.9 + 0.1)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch * len(train_loader), eta_min=0, last_epoch=-1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        n_train = len(train_ids)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == args.n_channels, \
                    f'Network has been defined with {args.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if args.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (4 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)
                    # scheduler.step()
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if args.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

        writer.add_images('images', imgs[0:2], epoch)
        if args.n_classes == 1:
            writer.add_images('masks/true', true_masks[0:2], epoch)
            writer.add_images('masks/pred', (torch.sigmoid(masks_pred) > 0.5)[0:2], epoch)
        scheduler.step()

        if save_cp:
            try:
                os.makedirs(dir_checkpoint + f"_fold{fold}/", exist_ok=True)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if epoch % args.checkpoint_epoch_interval == 0:
                torch.save(net.module.state_dict(), dir_checkpoint + f"_fold{fold}/" + f'CP_epoch{epoch}.pth')
                logging.info(f'Checkpoint {epoch} saved !')

    writer.close()


if __name__ == '__main__':
    with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename=f"{args.dir_dataset}/log.txt")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device} {args.gpus}')

    # gray
    norm_mean = [0.46152964]
    norm_std = [0.10963361]

    # RGB
    # norm_mean = [0.4976264, 0.45133978, 0.3993562]
    # norm_std = [0.11552592, 0.10886826, 0.10727626]

    # Albumentations
    train_Transform = Compose([
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
        RandomRotate90(0.5),
        Flip(p=0.5),
        ShiftScaleRotate(p=0.2, interpolation=Image.NEAREST),  # , border_mode=cv2.BORDER_CONSTANT, value=0
        # Normalize(mean=norm_mean, std=norm_std),
    ], p=1.0)

    valid_Transform = Compose([
        Resize(height=args.inputsize, width=args.inputsize, interpolation=Image.NEAREST, p=1),
        # Normalize(mean=norm_mean, std=norm_std),
    ], p=1)

    for fold in range(args.K_FOLD):
        train_ids, val_ids = K_FOLD(args.dir_dataset + 'CV/fold_' + str(fold))
        print(f"\n{len(train_ids)} images for training")
        print(f"\n{len(val_ids)} images for validation")

        train_data = BasicDataset(args.dir_img, args.dir_mask, ids=train_ids, transform=train_Transform,
                                  dir_augs=args.dir_augs)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  collate_fn=collate_fn)

        valid_data = BasicDataset(args.dir_img, args.dir_mask, ids=val_ids, transform=valid_Transform)
        val_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                collate_fn=collate_fn, drop_last=True)

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N
        net = UNet(n_channels=args.n_channels, n_classes=args.n_classes, bilinear=args.bilinear)
        logging.info(f'Network:\n'
                     f'\t{args.n_channels} input channels\n'
                     f'\t{args.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if args.bilinear else "Transposed conv"} upscaling')
        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        net = nn.DataParallel(net, device_ids=[0, 1]).to(device)

        writer = SummaryWriter(
            comment=f'_{args.dir_dataset}/FOLD_{fold}/pos_weight_{args.pos_w}_{args.lr}_BS_{args.batch_size}_EPOCHS_{args.epochs}_inputsize_{args.inputsize}')

        logging.info(f'''Starting training:
                Epochs:          {args.epochs}
                Batch size:      {args.batch_size}
                Learning rate:   {args.lr}
                Training size:   {len(train_ids)}
                Validation size: {len(val_ids)}
                Checkpoints:     {args.save_cp}
                Device:          {device.type}
                Images scaling:  {args.img_scale}
                pos_weight:      {args.pos_w}
                input_size       {args.inputsize}
                optimizer:       {args.optim}
            ''')

        try:
            train_net(net,
                      train_loader,
                      val_loader,
                      device,
                      dir_checkpoint=args.dir_checkpoint,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      save_cp=args.save_cp,
                      pos_w=args.pos_w,
                      lr=args.lr
                      )
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
