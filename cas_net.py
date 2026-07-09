import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.shared_mlp(self.avg_pool(x))
        max_attention = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_attention + max_attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True).values
        attention = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return attention * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        return self.spatial_attention(x)


class ASPPBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        padding = 0 if kernel_size == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ASPPBranch(in_channels, out_channels, kernel_size=1),
                ASPPBranch(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=6,
                ),
                ASPPBranch(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=12,
                ),
                ASPPBranch(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=18,
                ),
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        pooled = self.global_pool(x)
        pooled = F.interpolate(
            pooled,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class CASNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        use_cbam=True,
        use_aspp=True,
        use_aux_head=True,
        use_boundary_head=True,
        pretrained=False,
    ):
        super().__init__()
        self.use_aux_head = use_aux_head
        self.use_boundary_head = use_boundary_head
        self.backbone = timm.create_model(
            "resnet18",
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        backbone_channels = [
            feature["num_chs"] for feature in self.backbone.feature_info
        ]
        if len(backbone_channels) < 5:
            raise ValueError("CASNet requires five ResNet feature levels.")

        self.cbam_blocks = nn.ModuleList(
            [
                CBAM(channels) if use_cbam else nn.Identity()
                for channels in backbone_channels
            ]
        )
        self.aspp = (
            ASPP(backbone_channels[-1], 512)
            if use_aspp
            else nn.Identity()
        )
        decoder_channels = [512, 256, 128, 64, 32]
        skip_channels = [
            0,
            backbone_channels[-2],
            backbone_channels[-3],
            backbone_channels[-4],
        ]
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    decoder_channels[index],
                    skip_channels[index],
                    decoder_channels[index + 1],
                )
                for index in range(4)
            ]
        )
        self.spatial_attention = SpatialAttention()
        self.main_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(
            32,
            num_classes,
            kernel_size=3,
            padding=1,
        )
        self.auxiliary_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)
        features = [
            attention(feature)
            for attention, feature in zip(self.cbam_blocks, features)
        ]

        decoded = self.aspp(features[-1])
        decoder_outputs = []

        for index, decoder in enumerate(self.decoder_blocks):
            skip = None if index == 0 else features[-(index + 1)]
            decoded = decoder(decoded, skip)
            decoder_outputs.append(decoded)

        decoded = self.spatial_attention(decoded)

        main_output = F.interpolate(
            self.main_head(decoded),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        outputs = [main_output]

        if self.use_boundary_head:
            boundary_output = F.interpolate(
                self.boundary_head(decoded),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(boundary_output)

        if self.use_aux_head:
            auxiliary_output = F.interpolate(
                self.auxiliary_head(decoder_outputs[1]),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            outputs.append(auxiliary_output)

        return outputs
