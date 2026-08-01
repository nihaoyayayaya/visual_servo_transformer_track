import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultimodalFusion(nn.Module):
    def __init__(self, feature_dim, img_feat_dim, hidden_dim):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)
        # Cross attention mechanism
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, features, img_feat):
        """
        features: [B, N, 7] Feature points + force information
        img_feat: [B, D_img] Image features
        """
        # Project to the same space
        B, N, _ = features.shape
        feature_proj = self.feature_proj(features)  # [B, N, D]
        img_proj = self.img_proj(img_feat).unsqueeze(1)  # [B, 1, D]
        # Cross attention: image features as query, feature points as key/value
        attn_output, _ = self.cross_attn(
            query=img_proj,
            key=feature_proj,
            value=feature_proj
        )
        # Residual connection + layer normalization
        attended = self.layer_norm(img_proj + attn_output)
        # Feature point enhancement: broadcast attention results to all feature points
        attended_expanded = attended.repeat(1, N, 1)  # [B, N, D]
        # Fusion of original features and image attention features
        combined = torch.cat([feature_proj, attended_expanded], dim=-1)  # [B, N, 2D]
        # Feed-forward network fusion
        fused = self.ffn(combined)  # [B, N, D]
        return fused


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256):
        super(FeatureEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class EndPoseEmbedding(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256):
        super(EndPoseEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        query = query + self.dropout(attn_output)
        query = self.norm(query)
        return query


class FeedForward(nn.Module):
    def __init__(self, hidden_dim=256, ff_dim=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=256, ff_dim=1024, num_heads=8, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Feature stream
        self.feature_self_attn = SelfAttention(hidden_dim, num_heads, dropout)
        self.feature_cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.feature_ff = FeedForward(hidden_dim, ff_dim, dropout)

        # Joint stream
        self.end_pose_self_attn = SelfAttention(hidden_dim, num_heads, dropout)
        self.end_pose_cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.end_pose_ff = FeedForward(hidden_dim, ff_dim, dropout)

    def forward(self, feature, end_pose):
        # Feature stream self-attention
        feature = self.feature_self_attn(feature)
        # Joint stream self-attention
        end_pose = self.end_pose_self_attn(end_pose)

        # Cross attention: feature attends to joint
        feature_temp = self.feature_cross_attn(feature, end_pose, end_pose)
        # Cross attention: joint attends to feature
        end_pose_temp = self.end_pose_cross_attn(end_pose, feature, feature)

        # Feed-forward network
        feature = self.feature_ff(feature_temp)
        end_pose = self.end_pose_ff(end_pose_temp)

        return feature, end_pose


class FusionSelfAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super(FusionSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature, end_pose):
        batch_size = feature.shape[0]

        # Ensure both inputs are in [B,1,256] format
        # feature is already [B,256], needs to be [B,1,256]
        feature = feature.unsqueeze(1)  # [B,1,256]
        # end_pose should also be [B,256], needs to be [B,1,256]
        end_pose = end_pose.unsqueeze(1)  # [B,1,256]

        # Reshape to [B,2,256]
        combined = torch.cat([feature, end_pose], dim=1)  # [B,2,256]

        # Self-attention
        attn_output, _ = self.multihead_attn(combined, combined, combined)
        combined = combined + self.dropout(attn_output)

        # Reshape back to [B,512]
        combined = combined.reshape(batch_size, -1)  # [B,512]
        combined = self.norm(combined)

        return combined


class FeatureFusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.2):
        super(FeatureFusion, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class PredictionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=6):
        super(PredictionHead, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation2 = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden_dim // 2)

        self.linear3 = nn.Linear(hidden_dim // 2, output_dim)
        self.tanh = nn.Tanh()  # Add Tanh activation function
        # self.hard_tanh = nn.Hardtanh()  # Add Hardtanh activation function
        # self.hard_tanh = nn.Hardtanh(min_val=-3.0, max_val=3.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.norm1(x)

        x = self.linear2(x)
        x = self.activation2(x)
        x = self.norm2(x)

        x = self.linear3(x)
        # x = self.tanh(x)  # Apply Tanh activation to limit output range to [-1,1]
        return x


class VisualServoTransformer(nn.Module):
    def __init__(self,
                 feature_dim=7,
                 end_pose_dim=6,
                 hidden_dim=256,
                 ff_dim=1024,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 img_channels=1,
                 img_feat_dim=128):
        super(VisualServoTransformer, self).__init__()

        # Ultrasound image feature extractor (more powerful network)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, img_feat_dim)
        )

        # Multimodal fusion module
        self.multimodal_fusion = MultimodalFusion(
            feature_dim=feature_dim,
            img_feat_dim=img_feat_dim,
            hidden_dim=hidden_dim
        )

        # Feature point positional encoding remains unchanged
        self.feature_pos_encoding = PositionalEncoding(hidden_dim)

        # endpose embedding
        self.end_pose_embedding = EndPoseEmbedding(end_pose_dim, hidden_dim)
        self.end_pose_pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer layers (unchanged)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, ff_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Self-attention layer before fusion (newly added)
        self.fusion_self_attention = FusionSelfAttention(hidden_dim, num_heads, dropout)

        # Feature fusion layer
        self.feature_fusion = FeatureFusion(hidden_dim * 2, hidden_dim)

        # Prediction layer
        self.prediction_head = PredictionHead(hidden_dim, hidden_dim // 2, end_pose_dim)

    def forward(self, feature_error, end_pose_angles, ultrasound_img):
        # Extract ultrasound image features [B, img_feat_dim]
        img_feat = self.img_encoder(ultrasound_img)
        fused_features = self.multimodal_fusion(feature_error, img_feat)

        # End pose embedding [B,6] -> [B,1,256]
        end_pose = self.end_pose_embedding(end_pose_angles).unsqueeze(1)
        end_pose = self.end_pose_pos_encoding(end_pose)

        # Transformer processing
        for transformer_block in self.transformer_blocks:
            feature, end_pose = transformer_block(fused_features, end_pose)

        # Global pooling and dimension squeezing
        # feature = torch.mean(feature, dim=1)  # [B,256]
        feature = feature.squeeze(1)
        end_pose = end_pose.squeeze(1)  # [B,256]

        # Self-attention before fusion (newly added)
        combined = self.fusion_self_attention(feature, end_pose)  # [B,512]

        # Feature fusion
        fused = self.feature_fusion(combined)  # [B,256]

        # End effector velocity change prediction
        end_theta = self.prediction_head(fused)  # [B,6]

        return end_theta
