import torch.nn as nn

class CustomCNN1(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # stateful layers 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(1024, 512) # assuming input image shape is 224,224
        self.fc2 = nn.Linear(512, num_classes)

        # stateless layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(0.5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)  

        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.dropout(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x


class CustomCNN2(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # stateful layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(1024, 512) # assuming input image shape is 224,224
        self.fc2 = nn.Linear(512, num_classes)

        # stateless layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(0.2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.flatten(start_dim=1) # flatten all dimensions except batch

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

# unit test
if __name__ == "__main__":
    import torch
    from torchsummary import summary

    model = CustomCNN2(num_classes=3).to("cuda:0")
    summary(model, (3, 224, 224))

    example = torch.rand((5, 3, 224, 224)).to("cuda:0")
    logits = model(example)
    print("\n\n", logits)
    print(logits.sum(dim=1))