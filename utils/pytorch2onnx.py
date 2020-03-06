# coding=utf-8
import torch
import torch.nn, torch.onnx
from torch.autograd import Variable
from utils.ShuffleNetV2 import shufflenetv2

model_path = r"G:\VM-Debian-shared\shuffleNet_V2_epoch1-28.pt"
# model_path = r"G:\Project-Code\honk\model\speech_new-4-4.pt"
model_shuffle = shufflenetv2(width_mult=0.5)
model = torch.nn.DataParallel(model_shuffle)
parameters = torch.load(model_path, map_location='cpu')
model.load_state_dict(parameters)
model.eval()
x = Variable(torch.randn(1, 1, 101, 40))
y = model(x)
torch_out = torch.onnx._export(model, x, "shuffleNet_V2_28.onnx",  verbose=True)
print("Export of torch_model.onnx complete!")