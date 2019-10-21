import torch
import torch.nn as nn
import numpy as np
import torchvision.models.inception as inceptionModel
import torch.nn.functional as F

class InceptionV3_6e(nn.Module):
    def __init__(self):
        super(InceptionV3_6e, self).__init__()
        self.inceptionv3_model = inceptionModel.inception_v3(pretrained=True)
        self.inceptionv3_model.eval()
        self.Conv2d_1a_3x3 = self.inceptionv3_model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.inceptionv3_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.inceptionv3_model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = self.inceptionv3_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.inceptionv3_model.Conv2d_4a_3x3
        self.Mixed_5b = self.inceptionv3_model.Mixed_5b
        self.Mixed_5c = self.inceptionv3_model.Mixed_5c
        self.Mixed_5d = self.inceptionv3_model.Mixed_5d
        self.Mixed_6a = self.inceptionv3_model.Mixed_6a
        self.Mixed_6b = self.inceptionv3_model.Mixed_6b
        self.Mixed_6c = self.inceptionv3_model.Mixed_6c
        self.Mixed_6d = self.inceptionv3_model.Mixed_6d
        self.Mixed_6e = self.inceptionv3_model.Mixed_6e
    def forward(self, input):
        x = self.Conv2d_1a_3x3(input)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        return self.Mixed_6e(x)
        
        
class InceptionFeatures(nn.Module):
    def __init__(self):
        super(InceptionFeatures, self).__init__()
        self.inception = InceptionV3_6e()

    def forward(self, x):
        x = x.unsqueeze(0)
        out = self.inception(x)
        return out

#Creating this on top of InceptionFeatures since operations after concat appear to fail when going from onnx to coreml
class CompressFeatures(nn.Module):
    def __init__(self):
        super(CompressFeatures, self).__init__()
        self.fc_condense = nn.Linear(768, 50, bias=True)

    def forward(self, y):
        b,c,h,w = y.size()
        channelView = y.view(b, 768, -1)
        y = torch.mean(channelView, 2)
        condensed_features = self.fc_condense(y)
        return condensed_features
