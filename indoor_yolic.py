#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn as nn
import copy
import os.path
import pandas as pd
import os
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import cv2
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

NumCell = 30  # number of cells
NumClass = 6  # number of classes
save_name = 'mobilenet_indoor'  # name of the model
polygonList = [[0, 0, 151, 0, 52, 54, 0, 54], [151, 0, 254, 0, 191, 54, 52, 54], [254, 0, 359, 0, 334, 54, 191, 54],
               [359, 0, 424, 0, 424, 54, 334, 54], [489, 0, 424, 0, 424, 54, 514, 54],
               [594, 0, 489, 0, 514, 54, 657, 54], [697, 0, 594, 0, 657, 54, 796, 54],
               [848, 0, 697, 0, 796, 54, 848, 54], [0, 54, 52, 54, 0, 82], [52, 54, 191, 54, 133, 102, 0, 102, 0, 82],
               [191, 54, 334, 54, 312, 102, 133, 102], [334, 54, 424, 54, 424, 102, 312, 102],
               [514, 54, 424, 54, 424, 102, 536, 102], [657, 54, 514, 54, 536, 102, 715, 102],
               [796, 54, 657, 54, 715, 102, 848, 102, 848, 82], [848, 54, 796, 54, 848, 82], [0, 102, 133, 102, 0, 212],
               [133, 102, 312, 102, 277, 178, 266, 179, 255, 180, 244, 181, 234, 182, 223, 183, 213, 184, 203, 185, 193,
                186, 183, 187, 173, 188, 163, 189, 153, 190, 145, 191, 137, 192, 129, 193, 121, 194, 113, 195, 106, 196,
                99, 197, 93, 198, 86, 199, 80, 200, 74, 201, 68, 202, 62, 203, 57, 204, 51, 205, 46, 206, 41, 207, 36,
                208, 31, 209, 27, 210, 22, 211, 17, 212, 13, 213, 8, 214, 3, 215, 0, 216, 0, 212],
               [312, 102, 424, 102, 424, 178, 277, 178], [536, 102, 424, 102, 424, 178, 571, 178],
               [715, 102, 536, 102, 571, 178, 582, 179, 593, 180, 604, 181, 614, 182, 625, 183, 635, 184, 645, 185, 655,
                186, 665, 187, 675, 188, 685, 189, 695, 190, 703, 191, 711, 192, 719, 193, 727, 194, 735, 195, 742, 196,
                749, 197, 755, 198, 762, 199, 768, 200, 774, 201, 780, 202, 786, 203, 791, 204, 797, 205, 802, 206, 807,
                207, 812, 208, 817, 209, 821, 210, 826, 211, 831, 212, 835, 213, 840, 214, 845, 215, 848, 216, 848,
                212], [848, 102, 715, 102, 848, 212],
               [277, 178, 266, 179, 255, 180, 244, 181, 234, 182, 223, 183, 213, 184, 203, 185, 193, 186, 183, 187, 173,
                188, 163, 189, 153, 190, 145, 191, 137, 192, 129, 193, 121, 194, 113, 195, 106, 196, 99, 197, 93, 198,
                86, 199, 80, 200, 74, 201, 68, 202, 62, 203, 57, 204, 51, 205, 46, 206, 41, 207, 36, 208, 31, 209, 27,
                210, 22, 211, 17, 212, 13, 213, 8, 214, 3, 215, 0, 216, 0, 338, 203, 338],
               [277, 178, 424, 178, 424, 338, 203, 338], [571, 178, 424, 178, 424, 338, 645, 338],
               [571, 178, 582, 179, 593, 180, 604, 181, 614, 182, 625, 183, 635, 184, 645, 185, 655, 186, 665, 187, 675,
                188, 685, 189, 695, 190, 703, 191, 711, 192, 719, 193, 727, 194, 735, 195, 742, 196, 749, 197, 755, 198,
                762, 199, 768, 200, 774, 201, 780, 202, 786, 203, 791, 204, 797, 205, 802, 206, 807, 207, 812, 208, 817,
                209, 821, 210, 826, 211, 831, 212, 835, 213, 840, 214, 845, 215, 848, 216, 848, 338, 645, 338],
               [0, 338, 203, 338, 137, 480, 0, 480], [203, 338, 424, 338, 424, 480, 137, 480],
               [645, 338, 424, 338, 424, 480, 711, 480], [848, 338, 645, 338, 711, 480, 848, 480]]

def generate_fractional_masks(polygon_list, original_size=(848, 480), input_size=(224, 224), feature_size=(7, 7)):
    masks = []
    scale_x = input_size[0] / original_size[0]
    scale_y = input_size[1] / original_size[1]
    
    for poly in polygon_list:
        pts = np.array(poly).reshape(-1, 2)
        scaled_pts = np.zeros_like(pts, dtype=np.float32)
        scaled_pts[:, 0] = pts[:, 0] * scale_x
        scaled_pts[:, 1] = pts[:, 1] * scale_y
        scaled_pts = np.int32(scaled_pts)
        
        mask = np.zeros((input_size[1], input_size[0]), dtype=np.float32)
        cv2.fillPoly(mask, [scaled_pts], 1.0)
        masks.append(mask)
        
    mask_tensor = torch.tensor(np.stack(masks))
    mask_tensor = mask_tensor.unsqueeze(0)
    mask_features = F.adaptive_avg_pool2d(mask_tensor, feature_size)
    mask_features = mask_features.squeeze(0)
    return mask_features

class YolicAlignModel(nn.Module):
    def __init__(self, pretrained=True):
        super(YolicAlignModel, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)
        self.features = base_model.features
        
        self.classifier = nn.Linear(1280, NumClass + 1)
        self.register_buffer('fractional_masks', generate_fractional_masks(polygonList))
        
    def forward(self, x):
        feat = self.features(x)
        B = feat.shape[0]
        
        feat_expanded = feat.unsqueeze(2) 
        masks_expanded = self.fractional_masks.unsqueeze(0).unsqueeze(0)
        
        masked_feat = feat_expanded * masks_expanded
        pooled = masked_feat.sum(dim=(3, 4))
        
        mask_area = masks_expanded.sum(dim=(3, 4))
        pooled = pooled / (mask_area + 1e-6)
        
        pooled = pooled.transpose(1, 2)
        logits = self.classifier(pooled)
        logits = logits.reshape(B, -1)
        return logits

model = YolicAlignModel(pretrained=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # optimizer and learning rate
torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def random_augmentation(image, label_list, seq_list):
    # flip image horizontally
    image = image.flip(1)
    n_groups = len(seq_list)
    n_labels = len(label_list)
    assert n_labels % n_groups == 0  # make sure it's evenly divisible

    group_size = n_labels // n_groups

    # divide the label_list into groups based on seq_list
    label_groups = []
    start_idx = 0
    for group_idx in seq_list:
        end_idx = start_idx + group_size
        label_groups.append(label_list[start_idx:end_idx])
        start_idx = end_idx

    # create a new label_list based on seq_list
    new_label_list = []
    for group_idx in seq_list:
        group = label_groups[group_idx]
        new_label_list.extend(group)

    return image, new_label_list


class MultiLabelRGBataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None, train=1):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        self.train = train
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = Image.open(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        if self.train == 1:
            if random.random() > 0.5:
                img, label = random_augmentation(img, label,
                                                 [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 21, 20,
                                                  19, 18, 17, 16, 25, 24, 23, 22, 29, 28, 27, 26])
        label = torch.tensor(label, dtype=torch.float32)
        return img, label, filename


train_trans = transforms.Compose(([

    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()  # divides by 255
]))
val_test_trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # divides by 255
]))

img_dir = 'images'
label_dir = 'labels'
img_list = os.listdir(img_dir)
train_img, Val_Test = train_test_split(img_list, test_size=0.3, random_state=2)
val_img, test_img = train_test_split(Val_Test, test_size=0.6666, random_state=2)

train = MultiLabelRGBataSet(img_dir, train_img, label_dir, train_trans, train=1)
valid = MultiLabelRGBataSet(img_dir, val_img, label_dir, val_test_trans, train=0)
test = MultiLabelRGBataSet(img_dir, test_img, label_dir, val_test_trans, train=0)

train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid,
                                           batch_size=args.batch_size,
                                           shuffle=False, num_workers=8)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.batch_size,
                                          shuffle=False, num_workers=8)

if args.cuda:
    model.cuda()

criterion = nn.BCEWithLogitsLoss()
scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)


def pred_acc(original, predicted):
    pred = torch.round(predicted).detach().numpy().astype(np.int64)
    orig = original.detach().numpy()
    pred = np.reshape(pred, (NumCell * (NumClass + 1), 1)).flatten()
    orig = np.reshape(orig, (NumCell * (NumClass + 1), 1)).flatten()
    num = 0
    enum = 0
    normal = np.asarray([0] * NumClass + [1])
    for cell in range(0, (NumCell * (NumClass + 1)), NumClass + 1):
        if (orig[cell:cell + NumClass + 1] == pred[cell:cell + NumClass + 1]).all():
            num = num + 1
        else:
            if not (orig[cell:cell + NumClass + 1] == normal).all() and not (
                    pred[cell:cell + NumClass + 1] == normal).all():
                enum = enum + 1
    return num / NumCell, (num + enum) / NumCell


def train(epoch, model):
    model.train()
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        target = target.type_as(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


best_correct = -999


def evaluate(model, data_loader, save_mode=False):
    model.eval()
    running_loss = []
    running_acc = []
    running_binary = []
    global best_correct
    with torch.no_grad():
        for batch_idx, (data, target, filenames) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            target = target.type_as(output)
            loss = criterion(output, target)
            output = torch.sigmoid(output)
            acc_all = []
            acc_binary = []
            for each_image, d in enumerate(output):
                all_acc, b_acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d))
                acc_all.append(all_acc)
                acc_binary.append(b_acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_all).mean())
            running_binary.append(np.asarray(acc_binary).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_binary = np.asarray(running_binary).mean()
    print('\n loader set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Binary ACC: ({:.4f}%)\n'.format(
        total_batch_loss, len(data_loader.dataset), total_batch_acc, total_batch_binary))
    if save_mode:
        now_correct = total_batch_acc
        if best_correct < now_correct:
            best_correct = now_correct
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       os.path.join(os.getcwd(), save_name + ".pth.tar"))
            print("New weight!")
    return total_batch_loss, total_batch_acc


if __name__ == '__main__':

    import datetime

    start_time = datetime.datetime.now()
    print(save_name)
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    all_test_loss = []
    all_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, model)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, valid_loader, save_mode=True)
        test_loss, test_acc = evaluate(model, test_loader)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        all_val_acc.append(val_acc)
        all_val_loss.append(val_loss)
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
        scheduler.step()
    list_res = []
    for i in range(len(all_train_loss)):
        list_res.append([all_train_loss[i], all_train_acc[i], all_val_loss[i], all_val_acc[i],
                         all_test_loss[i], all_test_acc[i]])

    column_name = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
    csv_name = save_name + '.csv'
    xml_df = pd.DataFrame(list_res, columns=column_name)
    xml_df.to_csv(csv_name, index=None)
    end_time = datetime.datetime.now()
    print('\nTime taken: {}\n'.format(end_time - start_time))
