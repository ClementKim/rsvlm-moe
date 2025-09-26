import torch
import model.CLIP_Encoder as CLIP_encoder
import model.Transformer_Block as Transformer_Block

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = CLIP_encoder.CLIPEncoder(embed_dim=512).to(device)


if __name__ == "__main__":
    main()