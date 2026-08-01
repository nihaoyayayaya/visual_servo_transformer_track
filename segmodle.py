import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    """Basic convolution block: two convolutional layers, each followed by BatchNorm and LeakyReLU activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels=None, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y) * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.conv1(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SimplifiedSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=32, depth=4, use_cbam=True, use_aspp=True,
                 use_aux_head=True, use_boundary_loss=True):
        super().__init__()
        self.depth = depth
        self.use_aux_head = use_aux_head  # Controls whether to enable the auxiliary head
        self.use_boundary_loss = use_boundary_loss  # Controls whether to enable boundary loss

        # ResNet18 backbone - modified to accept 1-channel input
        self.backbone = timm.create_model('resnet18', features_only=True, pretrained=False, in_chans=in_channels)

        backbone_channels = [c['num_chs'] for c in self.backbone.feature_info]

        # CBAM modules
        if use_cbam:
            self.cbams = nn.ModuleList([
                CBAM(channels) for channels in backbone_channels
            ])
        else:
            self.cbams = nn.ModuleList([nn.Identity() for _ in backbone_channels])

        # ASPP module (controlled by the use_aspp parameter)
        if use_aspp:
            self.aspp = ASPP(backbone_channels[-1], 512)
        else:
            self.aspp = nn.Identity()  # Skip directly if not using ASPP

        # Decoder
        self.decoder_stages = nn.ModuleList()

        decoder_channels = [512, 256, 128, 64, 32]

        for i in range(depth):
            skip_channels = 0 if i == 0 else backbone_channels[-(i + 1)]
            self.decoder_stages.append(
                DecoderBlock(
                    decoder_channels[i],
                    skip_channels,
                    decoder_channels[i + 1]
                )
            )

        # Spatial attention
        self.spatial_attention = SpatialAttention(in_channels=decoder_channels[-1])

        # Output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        # Boundary enhancement output
        self.boundary_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=3, padding=1)

        # Auxiliary head (for the second stage of the decoder)
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]

        # Feature extraction
        features = self.backbone(x)

        # Apply CBAM
        for i in range(len(features)):
            features[i] = self.cbams[i](features[i])

        # ASPP
        x = self.aspp(features[-1])  # Selectively executed based on whether ASPP is used

        # Decoder
        for i in range(self.depth):
            skip = None if i == 0 else features[-(i + 1)]
            x = self.decoder_stages[i](x, skip)

        # Apply the auxiliary head after the second stage of the decoder
        if self.use_aux_head and self.depth >= 2:
            aux_out = self.aux_head(features[
                                        2])  # Assuming the output size of the second decoder stage matches the input requirements of the auxiliary head

        # Spatial attention
        x = self.spatial_attention(x)

        # Main output
        main_out = self.final_conv(x)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=False)

        # Boundary output
        boundary_out = self.boundary_conv(x)

        outputs = [main_out]
        if self.use_boundary_loss:
            outputs.append(boundary_out)
        if self.use_aux_head:
            outputs.append(aux_out)

        return outputs