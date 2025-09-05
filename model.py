import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=2, pretrained=True):
    if pretrained:
        weights = models.DenseNet121_Weights.DEFAULT
    else:
        weights = None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
