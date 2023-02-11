import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.dropout = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        x = self.dropout(self.bn2(self.pool(x)))

        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64*4*4)

        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.softmax(self.fc3(x),dim = 1)
        return x




# unit test
if __name__ == "__main__":
    import torch
    from torchsummary import summary

    example = torch.rand((5, 3, 224, 224)).to("cuda:0")
    model = CustomCNN(num_classes=3).to("cuda:0")
    summary(model, (3, 224, 224))
    logits = model(example)
    print("\n\n", logits)
    print(logits.sum(dim=1))
      
        