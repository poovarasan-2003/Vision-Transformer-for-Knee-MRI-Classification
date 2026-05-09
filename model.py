import timm
import torch
import torch.nn as nn

class MRNetViT(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(MRNetViT, self).__init__()
        # Feature extractor
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        self.in_features = 384
        
        # Attention-based Aggregator for slice importance
        self.attention_pool = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.in_features, 3)
        )

    def forward(self, x):
        # x: (batch, num_slices, 3, 224, 224)
        batch_size, num_slices, c, h, w = x.shape
        x = x.view(-1, c, h, w) 
        
        features = self.vit(x) # (batch * num_slices, 384)
        features = features.view(batch_size, num_slices, -1) # (batch, num_slices, 384)
        
        # Compute slice importance weights
        attn_weights = self.attention_pool(features) # (batch, num_slices, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum of features
        aggregated = torch.sum(attn_weights * features, dim=1) # (batch, 384)
        
        logits = self.classifier(aggregated)
        return logits

    def get_attention_weights(self, x):
        """Returns the slice-wise importance weights."""
        batch_size, num_slices, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        with torch.no_grad():
            features = self.vit(x)
            features = features.view(batch_size, num_slices, -1)
            attn_weights = torch.softmax(self.attention_pool(features), dim=1)
        return attn_weights

    def get_spatial_attention(self, x_slice):
        """Extracts the self-attention map from the last layer of ViT for a single slice."""
        # x_slice: (1, 3, 224, 224)
        # We use timm's internal attention storage if available, or just hook it.
        # For simplicity with timm's ViT, we can access the last block's attention.
        
        # Ensure we have the attention weights
        # vit_small_patch16_224 has blocks[0...11]
        last_block = self.vit.blocks[-1]
        
        # Temporary hook to catch attention
        attentions = []
        def hook_fn(module, input, output):
            # Attention is usually in the second element of the block output if returning attn
            # But standard timm ViT doesn't return it by default. 
            # We hook the 'attn_drop' or the Attention module itself.
            pass
            
        # Refined approach: Use the attention scores from the Attention module
        # The 'Attention' module in timm ViT computes: attn = (q @ k.transpose(-2, -1)) * scale
        # We will manually perform one pass through the last block to get it.
        
        x = self.vit.patch_embed(x_slice)
        x = self.vit._pos_embed(x)
        for i, block in enumerate(self.vit.blocks):
            if i == len(self.vit.blocks) - 1:
                # Last block - extract attention
                # attn = q @ k^T
                B, N, C = x.shape
                qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1) # (B, heads, N, N)
                # Average across heads and take the CLS token's attention to other tokens
                cls_attn = attn[:, :, 0, 1:].mean(dim=1) # (B, N-1)
                return cls_attn
            x = block(x)
        return None

def get_model(pretrained=True):
    return MRNetViT(pretrained=pretrained)
