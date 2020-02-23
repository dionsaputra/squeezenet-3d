import os
import sys
import shutil
import json
import glob
import signal
import pickle
import time
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from video_folder import VideoFolder
from torchvision.transforms import *
from model.squeezenet3d import SqueezeNet3D


def str2bool(x): return (str(x).lower() == 'true')


eval_only = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_prec1 = 0

config_file = "./train/mini_config.json"
with open(config_file) as data_file:
    config = json.load(data_file)


class ModelName:
    squeezet_3d = "squeezenet_3d"


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def load_model(model, pretrain_path):
    checkpoint = torch.load(pretrain_path)
    old_keys = [key for key in checkpoint["state_dict"].keys()]
    for key in old_keys:
        new_key = key[7:]
        checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)

    model.load_state_dict(checkpoint["state_dict"])
    return model


def initialize_model(model_name, num_classes, pretrained_path):
    model_ft = None
    input_size = 0

    if model_name == ModelName.squeezet_3d:
        model_ft = SqueezeNet3D()
        model_ft = load_model(model_ft, pretrained_path)
        set_parameter_requires_grad(model_ft)
        model_ft.classifier[1] = torch.nn.Conv3d(
            512, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    since = time.time()
    train_acc_history = []
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            )/len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == "train" and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc

            if phase == "val":
                val_acc_history.append(epoch_acc.detach())
            elif phase == "train":
                train_acc_history.append(epoch_acc.detach())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60}m {time_elapsed%60}s")
    print(f"Best val Acc: {best_acc}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history


def perform_train(config):
    model_ft, input_size = initialize_model(
        config["model_name"],
        config["num_classes"],
        config["pretrained_path"]
    )
    print(f"[INFO] success initialize model {config['model_name']}")

    # define data transform
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]
    train_transform = Compose([
        RandomResizedCrop(input_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(transform_mean, transform_std)
    ])

    val_transform = Compose([
        Resize(input_size),
        CenterCrop(input_size),
        ToTensor(),
        Normalize(transform_mean, transform_std)
    ])

    data_transforms = {
        "train": train_transform,
        "val": val_transform
    }
    print(f"[INFO] success create data transforms")

    video_datasets = {
        x: VideoFolder(
            root=config['train_data_folder'],
            csv_file_input=config['train_data_csv'],
            csv_file_labels=config['labels_csv'],
            clip_size=config['clip_size'],
            nclips=1,
            step_size=config['step_size'],
            is_val=False,
            transform=data_transforms[x],
        )
        for x in ["train", "val"]
    }

    # create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            video_datasets[x],
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"]
        )
        for x in ["train", "val"]
    }

    print(f"[INFO] success create data loaders")

    # send model to GPU
    model_ft = model_ft.to(device)
    print(f"[INFO] success send model to device {device}")

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    optimizer_ft = optim.SGD(
        params_to_update,
        lr=config["learning_rate"],
        momentum=config["momentum"]
    )
    criterion = nn.CrossEntropyLoss()

    print("[INFO] starting train")
    # train and evaluate
    model_ft, val_hist, train_hist = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        num_epochs=config["num_epochs"]
    )

    # saving best model
    timestamp = time.time()
    model_result_path = f"{config['output_dir']}/{timestamp}_{config['model_name']}.pth"
    torch.save(model_ft.state_dict(), model_result_path)
    acc_history = {
        'train': [x.item() for x in train_hist],
        'val': [x.item() for x in val_hist]
    }

    history_result_path = f"{config['output_dir']}/{timestamp}_{config['model_name']}.json"
    with open(history_result_path, "w") as file:
        json.dump(acc_history, file)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input, target = input.to(device), target.to(device)

        model.zero_grad()

        # compute output and loss
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(
            output.detach(), target.detach().cpu(), topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config["print_freq"] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, class_to_idx=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    targets_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input, target = input.to(device), target.to(device)

            # compute output and loss
            output = model(input)
            loss = criterion(output, target)

            if args.eval_only:
                logits_matrix.append(output.detach().cpu().numpy())
                targets_list.append(target.detach().cpu().numpy())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                output.detach(), target.detach().cpu(), topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % config["print_freq"] == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if args.eval_only:
            logits_matrix = np.concatenate(logits_matrix)
            targets_list = np.concatenate(targets_list)
            print(logits_matrix.shape, targets_list.shape)
            save_results(logits_matrix, targets_list, class_to_idx, config)
        return losses.avg, top1.avg, top5.avg


def save_results(logits_matrix, targets_list, class_to_idx, config):
    print("Saving inference results ...")
    path_to_save = os.path.join(
        config['output_dir'], config['model_name'], "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, targets_list, class_to_idx], f)


def save_checkpoint(state, is_best, config, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(
        config['output_dir'], config['model_name'], filename)
    model_path = os.path.join(
        config['output_dir'], config['model_name'], 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, model_path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.cpu().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    perform_train(config)
