from thop import profile
import torch
from utils.EdgeCRNN import EdgeCRNN


model = EdgeCRNN(width_mult=0.5)
# model = torch.nn.DataParallel(model_shuffle)  # 调用shufflenet2 模型，该模型为自己定义的

# model = MobileNetV2(width_mult=1.5)

# model = MobileNetV3_Small()
flop, para = profile(model, input_size=(1,1, 39, 101))
print("FLOPs:%.2fM" % (flop / 1e6), "Parameters:%.2fM" % (para / 1e6))
# print(stat(model, (1, 39, 101)))

# total = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))
