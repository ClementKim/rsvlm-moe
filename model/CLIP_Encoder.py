import torch
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from transformers import RobertaModel

class VisionTransformer(nn.Module):
    """
    사전 학습된 Vision Transformer (ViT-B/32)를 사용하여 이미지를 인코딩하는 클래스.
    최종 분류 헤드를 Identity로 교체하고, 지정된 임베딩 차원으로 특징을 사영하는 선형 계층을 추가합니다.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # 사전 학습된 ViT-B/32 모델을 로드합니다.
        self.model = vit_b_32(weights = ViT_B_32_Weights.DEFAULT)
        # 원래의 분류 헤드를 Identity 함수로 교체하여 특징 추출기로 사용합니다.
        self.model.heads = nn.Identity()
        # ViT의 출력 차원(768)을 목표 임베딩 차원(embed_dim)으로 변환하는 선형 계층입니다.
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, images):
        """
        이미지를 입력받아 임베딩 벡터를 반환합니다.
        
        Args:
            images (torch.Tensor): (B, C, H, W) 형태의 이미지 텐서.
        
        Returns:
            torch.Tensor: (B, embed_dim) 형태의 이미지 임베딩.
        """
        # ViT 모델을 통해 이미지 특징을 추출합니다.
        x = self.model(images)
        # 선형 계층을 통해 최종 임베딩을 생성합니다.
        x = self.fc(x)
        return x

class TransformerTextEncoder(nn.Module):
    """
    사전 학습된 RoBERTa 모델을 사용하여 텍스트를 인코딩하는 클래스.
    [CLS] 토큰의 출력을 사용하여 문장 전체의 표현을 얻고, 이를 지정된 임베딩 차원으로 사영합니다.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # 사전 학습된 "roberta-base" 모델을 로드합니다.
        self.model = RobertaModel.from_pretrained("roberta-base")
        # RoBERTa의 출력 차원(768)을 목표 임베딩 차원(embed_dim)으로 변환하는 선형 계층입니다.
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        """
        텍스트 토큰을 입력받아 임베딩 벡터를 반환합니다.
        
        Args:
            input_ids (torch.Tensor): (B, T) 형태의 입력 토큰 ID.
            attention_mask (torch.Tensor): (B, T) 형태의 어텐션 마스크.
        
        Returns:
            torch.Tensor: (B, embed_dim) 형태의 텍스트 임베딩.
        """
        # RoBERTa 모델을 통해 텍스트 특징을 추출합니다.
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] 토큰에 해당하는 마지막 은닉 상태를 사용합니다 (시퀀스의 첫 번째 토큰).
        x = x.last_hidden_state[:, 0, :]
        # 선형 계층을 통해 최종 임베딩을 생성합니다.
        x = self.fc(x)
        return x

class ProjectionHead(nn.Module):
    """
    CLIP 모델의 표준 구성 요소인 프로젝션 헤드.
    입력 임베딩을 선형 변환하고, L2 정규화를 수행한 후, 학습 가능한 온도 파라미터(scale)를 곱합니다.
    """
    def __init__(self, embed_dim=512, projection_dim=512):
        super().__init__()
        # 임베딩 차원을 프로젝션 차원으로 변환하는 선형 계층입니다.
        self.projection = nn.Linear(embed_dim, projection_dim)
        # 대조 손실(contrastive loss)의 온도를 조절하는 학습 가능한 파라미터입니다.
        self.scale = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x):
        """
        입력 임베딩에 프로젝션을 적용합니다.
        
        Args:
            x (torch.Tensor): (B, embed_dim) 형태의 임베딩.
        
        Returns:
            torch.Tensor: (B, projection_dim) 형태의 정규화 및 스케일링된 임베딩.
        """
        # 선형 프로젝션을 적용합니다.
        x = self.projection(x)
        # L2 정규화를 통해 특징 벡터의 크기를 1로 만듭니다.
        x = x / x.norm(dim=-1, keepdim=True)
        # 학습 가능한 스케일 파라미터를 곱합니다.
        return x * self.scale

class CLIPEncoder(nn.Module):
    """
    이미지 인코더와 텍스트 인코더를 결합하여 멀티모달 임베딩을 생성하는 메인 CLIP 인코더 클래스.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # 비전 인코더를 초기화합니다.
        self.vision_encoder = VisionTransformer(embed_dim)
        # 텍스트 인코더를 초기화합니다.
        self.text_encoder = TransformerTextEncoder(embed_dim)
        # 비전 프로젝션 헤드를 초기화합니다.
        self.vision_projection = ProjectionHead(embed_dim)
        # 텍스트 프로젝션 헤드를 초기화합니다.
        self.text_projection = ProjectionHead(embed_dim)

    def forward(self, images, input_ids, attention_mask):
        """
        이미지와 텍스트를 입력받아 각각의 임베딩을 계산하고, 이를 하나의 텐서로 묶어 반환합니다.
        
        Args:
            images (torch.Tensor): (B, C, H, W) 형태의 이미지 텐서.
            input_ids (torch.Tensor): (B, T) 형태의 입력 토큰 ID.
            attention_mask (torch.Tensor): (B, T) 형태의 어텐션 마스크.
        
        Returns:
            torch.Tensor: (B, 2, embed_dim) 형태의 텐서. dim=1은 [이미지, 텍스트] 임베딩을 나타냅니다.
        """
        # 이미지를 인코딩하고 프로젝션을 적용합니다.
        image_features = self.vision_projection(self.vision_encoder(images))
        # 텍스트를 인코딩하고 프로젝션을 적용합니다.
        text_features = self.text_projection(self.text_encoder(input_ids, attention_mask))
        
        # 이미지와 텍스트 특징을 새로운 차원(dim=1)을 따라 쌓습니다.
        return torch.stack([image_features, text_features], dim=1)
