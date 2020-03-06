# coding = utf-8
from collections import ChainMap
import argparse
import torch
import numpy as np

from utils.train import set_seed, train, evaluate
from utils.config import default_config
from nets.EdgeCRNN import EdgeCRNN

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return args

def main():

    global_config = dict(lr=[0.001, 0.0001], schedule=[np.inf], batch_size=64, dev_every=1, seed=0,
                         model=None, use_nesterov=False, gpu_no=0, cache_size=32768, momentum=0.9, weight_decay=0.00001)
    builder = ConfigBuilder(default_config(), global_config)
    parser = builder.build_argparse()
    # parser.add_argument("--no_cuda", type=str2bool, nargs='?', const=True)

    config = builder.config_from_argparse(parser)
    if config["model_type"] == "EdgeCRNN":
        model = EdgeCRNN(width_mult=config["width_mult"])
        model = torch.nn.DataParallel(model)
    if config["model_type"] == "shuffleNet":
        from nets.ShuffleNetV2 import shufflenetv2
        model_shuffle = shufflenetv2(width_mult=config["width_mult"])
        model = torch.nn.DataParallel(model_shuffle)
    elif config["model_type"] == "mobileNet":
        from nets.MobileNetV2 import MobileNetV2
        model = MobileNetV2(width_mult=config["width_mult"])
    elif config["model_type"] == "mobileNetV3-Small":
        from nets.MobileNetV3 import MobileNetV3_Small
        model = MobileNetV3_Small()
    elif config["model_type"] == "mobileNetV3-Large":
        from utils.MobileNetV3 import MobileNetV3_Large
        model = MobileNetV3_Large()
    elif config["model_type"] == "Tpool2":
        from nets.Tpool2 import CNN
        model = CNN()
    else:
        pass

    config["model"] = model
    set_seed(config)
    if config["type"] == "train":
        train(config)
    elif config["type"] == "eval":
        evaluate(config)

if __name__ == "__main__":
    main()