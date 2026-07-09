import math

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        feature_dim,
        image_feature_dim,
        hidden_dim,
        num_heads,
    ):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, features, image_features):
        token_count = features.shape[1]
        projected_features = self.feature_projection(features)
        projected_image = self.image_projection(image_features).unsqueeze(1)
        attended_image, _ = self.cross_attention(
            projected_image,
            projected_features,
            projected_features,
            need_weights=False,
        )
        attended_image = self.norm(projected_image + attended_image)
        attended_image = attended_image.expand(-1, token_count, -1)
        fused = torch.cat(
            [projected_features, attended_image],
            dim=-1,
        )
        return self.feed_forward(fused)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=100):
        super().__init__()
        encoding = torch.zeros(max_length, hidden_dim)
        positions = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.shape[1]]


class PoseEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.embedding(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attended, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )
        return self.norm(x + self.dropout(attended))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attended, _ = self.attention(
            query,
            key,
            value,
            need_weights=False,
        )
        return self.norm(query + self.dropout(attended))


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, feed_forward_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x + self.network(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feed_forward_dim,
        num_heads,
        dropout,
    ):
        super().__init__()
        self.feature_self_attention = SelfAttention(
            hidden_dim,
            num_heads,
            dropout,
        )
        self.pose_self_attention = SelfAttention(
            hidden_dim,
            num_heads,
            dropout,
        )
        self.feature_cross_attention = CrossAttention(
            hidden_dim,
            num_heads,
            dropout,
        )
        self.pose_cross_attention = CrossAttention(
            hidden_dim,
            num_heads,
            dropout,
        )
        self.feature_feed_forward = FeedForward(
            hidden_dim,
            feed_forward_dim,
            dropout,
        )
        self.pose_feed_forward = FeedForward(
            hidden_dim,
            feed_forward_dim,
            dropout,
        )

    def forward(self, features, pose):
        features = self.feature_self_attention(features)
        pose = self.pose_self_attention(pose)
        attended_features = self.feature_cross_attention(
            features,
            pose,
            pose,
        )
        attended_pose = self.pose_cross_attention(
            pose,
            features,
            features,
        )
        features = self.feature_feed_forward(attended_features)
        pose = self.pose_feed_forward(attended_pose)
        return features, pose


class FusionSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, features, pose):
        batch_size = features.shape[0]
        tokens = torch.stack([features, pose], dim=1)
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        return self.norm(tokens.reshape(batch_size, -1))


class FeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class PredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.output(x)


class TVSController(nn.Module):
    def __init__(
        self,
        feature_dim=6,
        pose_dim=6,
        output_dim=6,
        hidden_dim=256,
        feed_forward_dim=1024,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        image_channels=1,
        image_feature_dim=128,
    ):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, image_feature_dim),
        )
        self.multimodal_fusion = MultimodalFusion(
            feature_dim,
            image_feature_dim,
            hidden_dim,
            num_heads,
        )
        self.pose_embedding = PoseEmbedding(pose_dim, hidden_dim)
        self.pose_positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fusion_self_attention = FusionSelfAttention(
            hidden_dim,
            num_heads,
            dropout,
        )
        self.feature_fusion = FeatureFusion(
            hidden_dim * 2,
            hidden_dim,
            dropout,
        )
        self.prediction_head = PredictionHead(
            hidden_dim,
            output_dim,
        )

    def forward(
        self,
        multimodal_error,
        probe_pose,
        ultrasound_image,
    ):
        image_features = self.image_encoder(ultrasound_image)
        features = self.multimodal_fusion(
            multimodal_error,
            image_features,
        )
        pose = self.pose_embedding(probe_pose).unsqueeze(1)
        pose = self.pose_positional_encoding(pose)

        for block in self.transformer_blocks:
            features, pose = block(features, pose)

        features = features.mean(dim=1)
        pose = pose.mean(dim=1)
        fused = self.fusion_self_attention(features, pose)
        fused = self.feature_fusion(fused)
        return self.prediction_head(fused)
