""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # m = Flatten(name='flatten')(m)
        # m = Dense(1024, activation='relu', name='fc1')(m);m = Dropout(0.7)(m)
        # m = Dense(1024, activation='relu', name='fc2')(m);m = Dropout(0.7)(m)
        # m = Dense(num_labels, activation='softmax')(m)

        self.classifier = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),

            nn.Linear(1024, 10)
            # nn.CrossEntropyLoss() has include Softmax
        )


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #segmentation
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_out = self.outc(x)

        # classifiy
        flat = torch.flatten(x5, start_dim=1)#[n, 8192]
        cla_out = self.classifier(flat)#([n, 10])
        return seg_out, cla_out
