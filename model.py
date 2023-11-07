import torch.nn as nn
from torchvision.models import resnet50

class Resnet(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self._initModel(num_classes)

    def _initModel(self, num_classes):
        pre_model = resnet50(weights=None)
        num_fc_in_features = pre_model.fc.in_features
        pre_model.fc = nn.Linear(num_fc_in_features, num_classes)
        pre_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model = pre_model

    def forward(self, images):
        outputs = self.model(images)

        return outputs