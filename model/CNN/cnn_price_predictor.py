import torch.nn as nn

class CNNPricePredictor(nn.Module):
    def __init__(self):
        super(CNNPricePredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
        )

        # ВАЖНО: вычисли правильный размер после Flatten
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 64),  # Например, если вход был (3, 64, 64)
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)  # Теперь через fc_layers
        return x
