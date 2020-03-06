# coding = utf-8
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.utils.data as data
import argparse
import os
import datetime
import random
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import utils.Dataset as dataset
from utils.focalLoss import *


def print_eval(name, scores, labels, loss, step=0, interval=50, file=None, model_type=None, end="\n"):
    batch_size = labels.size(0)
    # print(batch_size, scores.shape)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    if model_type == "eval":
        print("the predicted value:", torch.max(scores, 1)[1].numpy() - 2 )
        print("the  labels   value:", labels.numpy() - 2)
    if step % interval == 0:
        print_result = "{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss)
        if file:
            file.write(print_result+end)
        print(print_result)
    return accuracy.item()


def set_seed(config):
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def evaluate(config, model=None, test_loader=None):
    # config["feature_type"] = "log_mel"
    if not test_loader:
        _, _, test_set = dataset.SpeechDataset.splits(config)
        test_loader = data.DataLoader(
            test_set,
            batch_size=len(test_set),
            collate_fn=test_set.collate_fn)

    if model == None:
        model = config["model"]
        if config["input_file"]:
            if not config["no_cuda"]:
                parameters = torch.load(config["input_file"])
            else:
                parameters = torch.load(config["input_file"], map_location='cpu')
            model.load_state_dict(parameters)
        if not config["no_cuda"]:
            torch.cuda.set_device(config["gpu_no"])
            model.cuda()
    model.eval()
    results = []
    total = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        model_in = torch.unsqueeze(model_in, 1)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        # loss = criterion(scores, labels)
        total += model_in.size(0)
        results.append(print_eval("test", scores, labels, 0, total, model_type=config["type"]) * model_in.size(0))
    print("final test accuracy: {}".format(sum(results) / total))
    return sum(results) / total


def train(config):
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    train_path = "{}-{}-train.txt".format(config["output_file"], datetime.datetime.now().strftime("%m-%d %H.%M.%S"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = config["model"]
    if config["input_file"]:
        if not config["no_cuda"]:
            parameters = torch.load(config["input_file"])
        else:
            parameters = torch.load(config["input_file"], map_location='cpu')
        model.load_state_dict(parameters)
    if not config["no_cuda"]:
        print(config["gpu_no"])
        # torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"][0])
    if config["loss"] == "CE":
        criterion = nn.CrossEntropyLoss()
    if config["loss"] == "focal":
        criterion = FocalLoss()
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    max_acc = 0
    # model = model.float()

    # preprocess_data 1 表示离线， 2 表示在线
    if config["preprocess_data"] == 1:
        train_set = dataset.WavDataset(r"{}/Train_1.h5".format(config["data_folder"]))
        dev_set = dataset.WavDataset(r"{}/Eval_1.h5".format(config["data_folder"]))
        test_set = dataset.WavDataset(r"{}/Test_1.h5".format(config["data_folder"]))

        train_loader = data.DataLoader(
            train_set,
            batch_size=config["batch_size"],  # 64
            shuffle=True, drop_last=True,
            # num_workers=4
        )
        dev_loader = data.DataLoader(
            dev_set,
            batch_size=min(len(dev_set), 16),
            shuffle=True,
            # num_workers=4
        )
        test_loader = data.DataLoader(
            test_set,
            batch_size=min(len(test_set), 16),
            shuffle=True,
            # num_workers=4
        )

    if config["preprocess_data"] == 2:
        train_set, dev_set, test_set = dataset.SpeechDataset.splits(config)
        train_loader = data.DataLoader(
            train_set,
            batch_size=config["batch_size"],  # 64
            shuffle=True, drop_last=True,
            collate_fn=train_set.collate_fn,
            num_workers=4
        )
        dev_loader = data.DataLoader(
            dev_set,
            batch_size=min(len(dev_set), 16),
            shuffle=True,
            collate_fn=dev_set.collate_fn,
            num_workers=4
        )
        test_loader = data.DataLoader(
            test_set,
            batch_size=min(len(test_set), 16),
            shuffle=True,
            collate_fn=test_set.collate_fn,
            num_workers=4
        )
    step_no = 0

    train_file = open(train_path, "a")
    train_file.write(config["output_file"])
    for epoch_idx in range(config["n_epochs"]):
        train_accs = []
        print("epoch {} start time：{}".format(epoch_idx, datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")))
        for batch_idx, (model_in, labels) in enumerate(train_loader):  # 花费5秒-64, 7秒-128
            # print("for star:", time.time())
            model.train()  # switch model to train model
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model = model.cuda()
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=True)
            model_in = torch.unsqueeze(model_in, 1)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False).long()
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            # print("for end:", time.time())
            train_accs.append(print_eval("[{}] train Epoch:{} step #{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"), epoch_idx, step_no),
                                         scores, labels, loss, step_no, file=train_file))

        # LR setting, 50 epoch
        if (epoch_idx + 1) % 50 == 0:
            lr_intel = config["lr"][0] - (epoch_idx + 1) / config["n_epochs"] * (config["lr"][0] - config["lr"][1])
            print("changing learning rate to {}".format(lr_intel))
            train_file.write("changing learning rate to {}".format(lr_intel))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_intel)

        print_log = "[{}] train Epoch:{} Accuracy：{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"), epoch_idx, np.mean(train_accs))
        print(print_log)
        train_file.write(print_log)
        print("epoch {} end  time：{}".format(epoch_idx, datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")))
        # 测试阶段
        with torch.no_grad():
            model.eval()
            accs = []
            index = 0
            total_score = torch.Tensor()
            total_label = torch.Tensor()
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                model_in = torch.unsqueeze(model_in, 1)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False).long()
                if len(total_label):
                    total_label = torch.cat((total_label, labels))
                    total_score = torch.cat((total_score, scores))
                else:
                    total_label = labels
                    total_score = scores

                loss = criterion(scores, labels)
                index = index + 1
                accs.append(print_eval("[{}] dev Epoch:{} ".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"), epoch_idx),
                                       scores, labels, loss, index, file=train_file, interval=30))

            if not config["no_cuda"]:
                total_score = total_score.cpu()
                total_label = total_label.cpu()
            avg_acc = np.mean(accs)
            print("final dev accuracy: {}".format(avg_acc))
            train_file.write("[{}] Epochs {} final dev accuracy: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"), epoch_idx, avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                train_file.write("the best accuracy:{}, saving best model...\n".format(avg_acc))
                max_acc = avg_acc
                if max_acc > 0.90:
                    torch.save(model.state_dict(), config["output_file"]+"-{:.5f}.pt".format(avg_acc))
                    # 计算ROC曲线
                    y_one_hot = label_binarize(total_label, np.arange(config["n_labels"]))
                    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), total_score.detach().numpy().ravel())
                    np.savetxt(config["output_file"] + "-{:.4f}-{}.csv".format(avg_acc, epoch_idx), [fpr, tpr],
                               delimiter=',', header="FPR,TPR")
                    config["input_file"] = config["output_file"]+"-{:.5f}.pt".format(avg_acc)  # input model

        evaluate(config, model, test_loader)

    train_file.close()

    # evaluate(config, model, test_loader)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


