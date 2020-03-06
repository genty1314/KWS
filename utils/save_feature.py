# coding = utf-8
import h5py
import numpy as np
import torch.utils.data as data
import utils.Dataset as mod
import datetime
from utils.train import ConfigBuilder, set_seed


def train(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)

    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)

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
        num_workers=4)

    h5_train = h5py.File("Train.h5", 'w')
    h5_eval = h5py.File("Eval.h5", 'w')
    h5_test = h5py.File("Test.h5", 'w')
    print("==> Preparing train data.. ")
    h5_data = h5_train.create_dataset("data", data=[[[]]], maxshape=(None, 101, 40), chunks=(1, 101, 40), compression="gzip", compression_opts=9)
    h5_labels = h5_train.create_dataset("labels", data=[], maxshape=((None,)), chunks=(1,), dtype=int, compression="gzip", compression_opts=9)
    train_total = 0
    for batch_idx, (model_in, labels) in enumerate(train_loader):  # 花费5秒-64, 7秒-128 model_in shape()?
        print("[{}] Batch:".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")), batch_idx)
        train_total = len(labels) + train_total  # 32  0:32; 64, 32:64;... 32n, 32n:32n+32,
        h5_data.resize((train_total, 101, 40))
        h5_labels.resize((train_total, ))
        h5_data[train_total-len(labels):train_total, :, :] = model_in
        h5_labels[train_total-len(labels):train_total] = labels

    h5_train.close()
    print("==> train data finish!")

    print("==> Preparing dev data.. ")
    eval_h5_data = h5_eval.create_dataset("data", data=[[[]]], maxshape=(None, 101, 40), chunks=(1, 101, 40), compression="gzip", compression_opts=9)
    eval_h5_labels = h5_eval.create_dataset("labels", data=[], maxshape=((None,)), chunks=(1,), dtype=int, compression="gzip", compression_opts=9)
    train_total = 0
    for batch_idx, (model_in, labels) in enumerate(dev_loader):
        print("[{}] Batch:".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")), batch_idx)
        train_total = len(labels) + train_total
        eval_h5_data.resize((train_total,101, 40))
        eval_h5_labels.resize((train_total, ))
        eval_h5_data[train_total - len(labels):train_total, :, :] = model_in
        eval_h5_labels[train_total - len(labels):train_total] = labels

    h5_eval.close()
    print("==> dev data finish!")

    print("==> Preparing test data.. ")
    test_h5_data = h5_test.create_dataset("data", data=[[[]]], maxshape=(None,  101, 40), chunks=(1, 101, 40), compression="gzip", compression_opts=9)
    test_h5_labels = h5_test.create_dataset("labels", data=[], maxshape=((None, )), chunks=(1,), dtype=int, compression="gzip", compression_opts=9)
    train_total = 0
    for batch_idx, (model_in, labels) in enumerate(test_loader):
        print("[{}] Batch:".format(datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")), batch_idx)
        train_total = len(labels) + train_total
        test_h5_data.resize((train_total, 101, 40))
        test_h5_labels.resize((train_total, ))
        test_h5_data[train_total - len(labels):train_total, :, :] = model_in
        test_h5_labels[train_total - len(labels):train_total] = labels
    h5_test.close()
    print("==> test data finish!")


def read_h5():
    h5_train = h5py.File(r"G:\Train.h5", 'r')
    # h5_eval = h5py.File(r"G:\Project-Code\honk-ShuffleNet\utils\Eval.h5", 'r')
    # h5_test = h5py.File(r"G:\Project-Code\honk-ShuffleNet\utils\Test.h5", 'r')

    print(h5_train["data"][:].shape, type(h5_train["data"][:]))  # shape(10, 2, 3)
    print(h5_train["labels"][:].shape, type(h5_train["labels"][:]))  # shape(10, 2, 3)
    print(len(h5_train["labels"][:]))
    # print(h5_eval["data"][:].shape, type(h5_eval["data"][:]))  # shape(10, 2, 3)
    # print(h5_eval["labels"][:].shape, type(h5_eval["labels"][:]))  # shape(10, 2, 3)
    # print(h5_test["data"][:].shape, type(h5_test["data"][:]))  # shape(10, 2, 3)
    # print(h5_test["labels"][:].shape, type(h5_test["labels"][:]))  # shape(10, 2, 3)


def default_config():
    config = {}
    config["group_speakers_by_id"] = True
    config["silence_prob"] = 0.1
    config["noise_prob"] = 0.8
    config["n_dct_filters"] = 40
    config["input_length"] = 16000
    config["n_mels"] = 40
    config["timeshift_ms"] = 100
    config["unknown_prob"] = 0.1
    config["train_pct"] = 80
    config["dev_pct"] = 10
    config["test_pct"] = 10
    config["wanted_words"] = ["command", "random"]
    config["data_folder"] = r"E:\Project-Data\honk\speech_commands_v0.02"
    config["audio_preprocess_type"] = "MFCCs"
    config["n_labels"] = 10
    config["input_file"] = "model/model.pt"
    return config


def main():
    global_config = dict(n_epochs=1, lr=[0.0001], schedule=[np.inf], batch_size=64, dev_every=1, seed=0,
                         model=None, use_nesterov=False,
                         gpu_no="1", cache_size=32768, momentum=0.9, weight_decay=0.00001)
    builder = ConfigBuilder(default_config(), global_config)
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    config["no_cuda"] = False
    config["type"] = "train"
    set_seed(config)
    train(config)
    # read_h5()


if __name__ == "__main__":
    main()
