import torch
import torch.nn as nn


torch.manual_seed(42)

class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  
            nn.SiLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  
            nn.SiLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.conv_layers(x)  # 
        x = torch.flatten(x, 1)  # Flatten before FC layer
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
model = CNNModel(num_classes=4).to(device)