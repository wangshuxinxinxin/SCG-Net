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
from data_loader.reg_dataloader import RegDataset
import models
from losses import Dice_loss
from utils import dice


parser = argparse.ArgumentParser(description='pytorch AASCE Reg')
parser.add_argument('--train_batch', default=20, type=int, help="train batch size")
parser.add_argument('--root_dir', default='data/', type=str, help='data path')
parser.add_argument('--save_dir', default='weights', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--log_dir', default='logs', type=str, metavar='SAVE',
                    help='directory to save logs (default: none)')
parser.add_argument('--gpu_devices', default='1,2', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--stage', default='train', type=str, help='regtrain, regtest')
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

    model = models.init_model('densenet', num_classes=3)
    if args.predict_model_path:
        model = load_weights(model, args.predict_model_path)

    if args.stage == 'regtrain':
        print('==>Prepare train data')
        t = time.clock()
        dataset = RegDataset(root_dir=os.path.join(args.root_dir, 'train'))
        dataloader = DataLoader(dataset, batch_size=args.train_batch, shuffle=True, num_workers=1)
        print("Training sample nums: %d" % len(dataloader))
        print("==> Using time:%ds" % int(time.clock() - t))

        print("==> Prepare validation data")
        t = time.clock()
        dataset_val = RegDataset(root_dir=os.path.join(args.root_dir, 'val'))
        dataloader_val = DataLoader(dataset_val, batch_size=2)
        print("Validation sample nums: %d" % len(dataloader_val))
        print("==> Using time %ds" % int(time.clock() - t))

        print('==> Start training')
        num_epochs = 1000
        dataset_size = len(dataset)
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
        dir_name = time.strftime("%m%d_", time.localtime()) + args.stage
        model_save_path = os.path.join(args.save_dir, dir_name)
        create_path(model_save_path)
        train(args, model, optimizer, dataloader, dataloader_val, dataset_size, num_epochs, model_save_path)

    elif args.stage == 'regtest':
        result_save_path = os.path.join(args.save_dir, 'result', args.predict_model_path.split('/')[-2])
        create_path(result_save_path)
        predict(model, args, result_save_path)


def train(args, model, optimizer, dataloader, dataloader_val, dataset_size, num_epochs, save_path):
    best_acc = float('inf') 
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename='log/regAASCE.log', filemode='a')
    logging.info(f'regAASCE Using Densent\n')

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
                        label = Variable(sample_batched['angle']).cuda()
                        img_name = sample_batched['name']

                        optimizer.zero_grad()
                        # forward
                        outputs = model(inputs)
                        train_num += len(inputs)
                        loss = torch.mean(torch.abs(label - outputs))
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
                error = val(model, dataloader_val, args)
                printline = 'Mean error: %.4f' % error
                print(printline)
                logging.info(printline)

                if error < best_acc:
                    best_acc = error
                    if gpu_nums > 1:
                        torch.save(model.module.state_dict(), os.path.join(save_path, '3DUnet_%d_%.4f.pth' % (epoch+1, best_acc)))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_path, '3DUnet_%d_%.4f.pth' % (epoch+1, best_acc)))

             
def val(model, dataloader_val, args):

    error = []
    for i_batch, sample_batched in enumerate(dataloader_val):
        inputs = Variable(sample_batched['image'].float()).cuda()
        labels = sample_batched['angle']
        period_names = sample_batched['period']

        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.data.cpu().numpy()
        mean_error = np.mean(np.abs(np.array(outputs) - np.array(labels)))
        error.append(mean_error)
       
    return np.asarray(error).mean()


def predict(model, args, result_save_path):
    test_path = os.path.join(args.root_dir, 'val')
    test_files = os.listdir(os.path.join(test_path, 'data'))
    segpath = 'segresult/'

    for ifile in test_files:
        print(ifile)
        img_path = os.path.join(args.root_dir, 'data', ifile)
        img = cv2.imread(img_path).swapaxes(0, 2).swapaxes(1, 2)
        label_path = os.path.join(segpath, ifile)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        _, x, y = img.shape
        x_crop, y_crop = (x // 2) * 2, (y // 2) * 2
        img = img[:, :x_crop, :y_crop]
        label = label[:x_crop, :y_crop]
        label = label[np.newaxis, ...]
        img = np.concatenate((img, label), axis = 0)
        img = img[np.newaxis, ...]

        with torch.no_grad():
            inputs = torch.from_numpy(img.float()).cuda()
            output = model(inputs)
        output = output.data.cpu().numpy()
        print(ouput)

if __name__ == '__main__':
    main()
