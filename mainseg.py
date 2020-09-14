# -*- coding:utf-8 -*-


import argparse
import os
import time
import os.path as op
import cv2

# third party import
from tqdm import tqdm
import logging
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.autograd import Variable

# project import
from data_loader.segment_dataloader import SegDataset
import models
from losses import Dice_loss
from utils import dice


parser = argparse.ArgumentParser(description='pytorch AASCE')
parser.add_argument('--train_batch', default=20, type=int, help="train batch size")
parser.add_argument('--root_dir', default='data/', type=str, help='data path')
parser.add_argument('--save_dir', default='weights', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--log_dir', default='logs', type=str, metavar='SAVE',
                    help='directory to save logs (default: none)')
parser.add_argument('--gpu_devices', default='1,2', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--stage', default='train', type=str, help='segtrain, segtest')
parser.add_argument('--predict_model_path',
                    default='',
                    type=str, help='the path of models for predicting (default: none)')

args = parser.parse_args()


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_weights(model, weight_path):

    pretrained_weights = torch.load(weight_path)
    model_weights = model.state_dict()
    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)

    return model


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    print("================\nArgs:{}\n==============".format(args))
    print('Currently using GPU {}'.format(args.gpu_devices))
    np.random.seed(6)
    torch.manual_seed(6)
    torch.cuda.manual_seed_all(6)

    model = models.init_model('unet', num_classes=1)
    if args.predict_model_path:
        model = load_weights(model, args.predict_model_path)

    if args.stage == 'segtrain':
        print('==>Prepare train data')
        t = time.clock()
        dataset = SegDataset(root_dir=os.path.join(args.root_dir, 'train'))
        dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=True, num_workers=1)
        print("Training sample nums: %d" % len(dataloader))
        print("==> Using time:%ds" % int(time.clock() - t))

        print("==> Prepare validation data")
        t = time.clock()
        dataset_val = SegDataset(root_dir=os.path.join(args.root_dir, 'val'))
        dataloader_val = DataLoader(dataset_val, batch_size=2)
        print("Validation sample nums: %d" % len(dataloader_val))
        print("==> Using time %ds" % int(time.clock() - t))

        print('==> Start training')
        num_epochs = 1000
        dataset_size = len(dataset)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        dir_name = time.strftime("%m%d_", time.localtime()) + args.stage
        model_save_path = os.path.join(args.save_dir, dir_name)
        create_path(model_save_path)
        train(args, model, optimizer, dataloader, dataloader_val, dataset_size, num_epochs, model_save_path)

    elif args.stage == 'segtest':
        result_save_path = os.path.join(args.save_dir, 'result', args.predict_model_path.split('/')[-2])
        create_path(result_save_path)
        predict(model, args, result_save_path)


def train(args, model, optimizer, dataloader, dataloader_val, dataset_size, num_epochs, save_path):
    best_acc = 0.0
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename='log/segAASCE.log', filemode='a')
    logging.info(f'segAASCE Using Unet\n')

    gpu_nums = len(args.gpu_devices.split(','))
    print('Using %d gpus' % gpu_nums)
    if gpu_nums > 1:
        model = DataParallel(model)
    model.cuda()

    for epoch in range(num_epochs):
        s1 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            runing_loss = 0.0
            train_num = 0
            if phase == 'train':
                with tqdm(total=dataset_size, desc=f'Epoch{epoch + 1}/{num_epochs}', unit='img') as pbar:
                    for i_batch, sample_batched in enumerate(dataloader):
                        inputs = Variable(sample_batched['image'].float()).cuda()
                        labels = Variable(sample_batched['mask']).cuda()
                        img_name = sample_batched['name']

                        optimizer.zero_grad()
                        # forward
                        outputs = model(inputs)
                        train_num += len(inputs)
                        criterion = Dice_loss()
                        loss = criterion(outputs.float(), labels.float())
                        # backward
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix(**{'loss (batch)': loss.item()})
                        pbar.update(sample_batched['image'].shape[0])

                        runing_loss += loss * inputs.size(0)

                    epoch_loss = runing_loss / dataset_size
                    print('epoch: %d, loss %.5f' % (epoch, epoch_loss))
                    s2 = time.time()
                    print('Train complete in %.0f m %.0f s' % ((s2 - s1) // 60, (s2 - s1) % 60))
                    logging.info(f'{epoch + 1}/{num_epochs} loss: {epoch_loss}')
            
            else:
                dice_mean = val(model, dataloader_val, args)
                printline = 'Mean Dice: %.4f' % dice_mean
                print(printline)
                logging.info(printline)

                if dice_mean > best_acc:
                    best_acc = dice_mean
                    if gpu_nums > 1:
                        torch.save(model.module.state_dict(), os.path.join(save_path, '3DUnet_%d_%.4f.pth' % (epoch+1, best_acc)))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_path, '3DUnet_%d_%.4f.pth' % (epoch+1, best_acc)))

             
def val(model, dataloader_val, args):

    dices = []
    for i_batch, sample_batched in enumerate(dataloader_val):
        inputs = Variable(sample_batched['image'].float()).cuda()
        labels = sample_batched['mask']
        period_names = sample_batched['period']

        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.data.cpu().numpy()
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        result = np.squeeze(outputs)

        labels = labels.data.numpy()
        labels = np.squeeze(labels)
        dice = dice(labels, result, labels=[1.])[0]
        dices.append(dice)
       
    return np.asarray(dices).mean()


def predict(model, args, result_save_path):
    test_path = os.path.join(args.root_dir, 'val')
    test_files = os.listdir(os.path.join(test_path, 'data'))

    for ifile in test_files:
        print(ifile)
        img_path = os.path.join(args.root_dir, 'data', ifile)
        img = cv2.imread(img_path).swapaxes(0, 2).swapaxes(1, 2)

        _, x, y = img.shape
        x_crop, y_crop = (x // 2) * 2, (y // 2) * 2
        img = img[:, :x_crop, :y_crop]
        img = img[np.newaxis, ...]

        with torch.no_grad():
            inputs = torch.from_numpy(img.float()).cuda()
            output = model(inputs)
        output = output.data.cpu().numpy()
        output[output > 0.5] = 1.
        output[output <= 0.5] = 0.
        output = np.squeeze(output)

        result_final = np.zeros((x, y))
        result_final[:x_crop, :y_crop] = output


        cv2.imwrite(os.path.join(result_save_path, ifile), result_final)

if __name__ == '__main__':
    main()
