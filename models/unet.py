import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, downsample=False):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=2, padding=1,
                                    bias=False)
            self.skip_conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=2,
                                        padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=outplanes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=outplanes, out_channels=outplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=outplanes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.skip_conv(residual)

        out += residual
        out = self.relu2(out)
        return out


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class UpBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=2,
                                        padding=1, output_padding=1)

        self.bn = nn.BatchNorm2d(num_features=outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class Residual_Unet(nn.Module):
    def __init__(self, num_classes):
        super(Residual_Unet, self).__init__()

        self.encode_conv1 = ConvBlock(inplanes=3, outplanes=30)

        self.res1_1 = BasicBlock(inplanes=64, outplanes=64)

        self.down1 = BasicBlock(inplanes=64, outplanes=128, downsample=True)
        self.res2_1 = BasicBlock(inplanes=128, outplanes=128)

        self.down2 = BasicBlock(inplanes=128, outplanes=256, downsample=True)
        self.res3_1 = BasicBlock(inplanes=256, outplanes=256)
        self.res3_2 = BasicBlock(inplanes=256, outplanes=256)

        self.down3 = BasicBlock(inplanes=256, outplanes=512, downsample=True)
        self.res4_1 = BasicBlock(inplanes=512, outplanes=512)
        self.res4_2 = BasicBlock(inplanes=512, outplanes=512)
        self.res4_3 = BasicBlock(inplanes=512, outplanes=512)

        self.down4 = BasicBlock(inplanes=512, outplanes=1024, downsample=True)
        self.res5_1 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res5_2 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res5_3 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res5_4 = BasicBlock(inplanes=1024, outplanes=1024)

        self.down5 = BasicBlock(inplanes=1024, outplanes=1024, downsample=True)
        self.res6_1 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res6_2 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res6_3 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res6_4 = BasicBlock(inplanes=1024, outplanes=1024)
        self.res6_5 = BasicBlock(inplanes=1024, outplanes=1024)

        self.up5 = UpBlock(inplanes=1024, outplanes=1024)
        self.decode_conv5 = ConvBlock(inplanes=1024, outplanes=1024)

        self.up4 = UpBlock(inplanes=1024, outplanes=512)
        self.decode_conv4 = ConvBlock(inplanes=512, outplanes=512)

        self.up3 = UpBlock(inplanes=512, outplanes=256)
        self.decode_conv3 = ConvBlock(inplanes=256, outplanes=256)

        self.up2 = UpBlock(inplanes=256, outplanes=128)
        self.decode_conv2 = ConvBlock(inplanes=128, outplanes=128)

        self.up1 = UpBlock(inplanes=128, outplanes=64)
        self.decode_conv1 = ConvBlock(inplanes=64, outplanes=64)

        self.classifier = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encode_conv1(x)
        skip1 = x = self.res1_1(x)

        x = self.down1(x)
        skip2 = x = self.res2_1(x)

        x = self.down2(x)
        x = self.res3_1(x)
        skip3 = x = self.res3_2(x)

        x = self.down3(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        skip4 = x = self.res4_3(x)

        x = self.down4(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        skip5 = x = self.res5_4(x)

        x = self.down5(x)
        x = self.res6_1(x)
        x = self.res6_2(x)
        x = self.res6_3(x)
        x = self.res6_4(x)
        x = self.res6_5(x)

        x = self.up5(x)
        x = x + skip5
        x = self.decode_conv5(x)

        x = self.up4(x)
        x = x + skip4
        x = self.decode_conv4(x)

        x = self.up3(x)
        x = x + skip3
        x = self.decode_conv3(x)

        x = self.up2(x)
        x = x + skip2
        x = self.decode_conv2(x)

        x = self.up1(x)
        x = x + skip1
        x = self.decode_conv1(x)

        out = self.classifier(x)
        return out

