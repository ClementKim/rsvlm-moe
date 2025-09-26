import torch
import torch.nn as nn
from torchvision.models import vit_b_32
from transformers import RobertaModel

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim = 512):
        super().__init__()
        self.model = vit_b_32(pretrained = True)
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, images):
        x = self.model(images)
        x = self.fc(x)

        return x
    
class TransformerTextEncoder(nn.Module):
    def __init__(self, embed_dim = 512):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids = input_ids, attention_mask = attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.fc(x)

        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, embed_dim = 512, projection_dim = 512):
        super().__init__()
        self.projection = nn.Linear(embed_dim, projection_dim) # Linear layer
        self.scale = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x):
        x = self.projection(x)
        x = x / x.norm(dim = -1, keepdim = True) # L2 normalization

        return x * self.scale
    
class CLIPEncoder(nn.Module):
    def __init__(self, embed_dim = 512):
        super().__init__()
        self.vision_encoder = VisionTransformer(embed_dim)
        self.text_encoder = TransformerTextEncoder(embed_dim)
        self.vision_projection = ProjectionHead(embed_dim)
        self.text_projection = ProjectionHead(embed_dim)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.vision_projection(self.vision_encoder(images))
        text_features = self.text_projection(self.text_encoder(input_ids, attention_mask))
        
        return torch.cat([image_features, text_features], dim = 1)