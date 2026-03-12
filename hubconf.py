import torch
import torch.nn as nn

from src.models.roformer import RoFormerClassifier

dependencies = ["torch"]

def accelerest_sleepstage(pretrained = True):
    model = RoFormerClassifier(
        mode = "token_wise",
        num_classes = 4,
        patch_size = 900,
        in_channels = 3,
        embed_dim = 256,
        num_heads = 8,
        mlp_ratio = 2,
        num_layers = 12,
        num_lstm_layers = 0,
        max_seq_len = 256,
        lstm_dim = 0,
        head = 'linear', 
        head_dropout = 0,
    )

    if pretrained:
        # print(f'Loading {config.pretrained_model}')
        repo = "https://github.com/NielsRLorenzen/AcceleRest/" 
        release = "releases/download/v0.1.0/"
        weights = "accelerest_sleepstage_linear_weights.pt"
        url = repo + release + weights
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True) 
    
    return model

def accelerest_respevent(pretrained = True):
    model = RoFormerClassifier(
        mode = "token_wise",
        num_classes = 2,
        patch_size = 900,
        in_channels = 3,
        embed_dim = 256,
        num_heads = 8,
        mlp_ratio = 2,
        num_layers = 12,
        num_lstm_layers = 0,
        max_seq_len = 256,
        lstm_dim = 0,
        head = 'linear', 
        head_dropout = 0,
    )

    if pretrained:
        # print(f'Loading {config.pretrained_model}')
        repo = "https://github.com/NielsRLorenzen/AcceleRest/" 
        release = "releases/download/v0.1.0/"
        weights = "accelerest_respevent_linear_weights.pt"
        url = repo + release + weights
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True) 
    
    return model

class DualHeadAcceleRest(nn.Module):
    def __init__(self, sleepstage_model, respevent_model):
        super().__init__()
        #Ensure identical encoder backbones
        if not identical_backbone(sleepstage_model.state_dict(), respevent_model.state_dict()):
            raise RuntimeError("The two AcceleRest models do not share identical encoders.")
        self.patch_size = sleepstage_model.patch_size
        self.max_seq_len = sleepstage_model.max_seq_len

        self.patch_embedding = sleepstage_model.patch_embedding
        self.encoder = sleepstage_model.encoder

        self.sleepstage_norm = sleepstage_model.norm
        self.sleepstage_head = sleepstage_model.classification_head

        self.respevent_norm = respevent_model.norm
        self.respevent_head = respevent_model.classification_head

    def forward(self, x):
        features = self.encoder(
            self.patch_embedding(x),
            use_sdpa=True,
        )

        sleepstages = self.sleepstage_head(
            self.sleepstage_norm(features)
        )

        respevents = self.respevent_head(
            self.respevent_norm(features)
        )

        return sleepstages, respevents

def identical_backbone(sd1, sd2):
    '''Compare two AcceleRest encoder statedicts to ensure identical parameters'''
    backbone_prefixes = ("patch_embedding", "encoder")

    for prefix in backbone_prefixes:
        keys1 = sorted(k for k in sd1 if k.startswith(prefix))
        keys2 = sorted(k for k in sd2 if k.startswith(prefix))

        if keys1 != keys2:
            print(f"Key mismatch under prefix '{prefix}'")
            return False

        for k in keys1:
            if not torch.equal(sd1[k], sd2[k]):
                max_diff = (sd1[k] - sd2[k]).abs().max().item()
                print(f"Mismatch in {k}, max abs diff = {max_diff}")
                return False
    return True

def accelerest_dualhead(pretrained = True):
    # Load accelerest w. sleep stage head
    sleepstage = accelerest_sleepstage(pretrained = pretrained)

    # Load accelerest w. resp event head
    respevent = accelerest_respevent(pretrained = pretrained)

    model = DualHeadAcceleRest(sleepstage, respevent)

    return model
