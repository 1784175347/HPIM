import torchvision.models as models
alexnet = models.alexnet(pretrained=True)
print(alexnet)