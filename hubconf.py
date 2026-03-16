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
        release = "releases/download/v0.1.1/"
        weights = "accelerest_sleepstage_linear_weights.pt"
        url = repo + release + weights
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True) 
    
    return model

def accelerest_sleepstage_lstm(pretrained = True):
    model = RoFormerClassifier(
        mode = "token_wise",
        num_classes = 4,
        patch_size = 900,
        in_channels = 3,
        embed_dim = 256,
        num_heads = 8,
        mlp_ratio = 2,
        num_layers = 12,
        num_lstm_layers = 2,
        max_seq_len = 256,
        lstm_dim = 4,
        head = 'linear', 
        head_dropout = 0,
    )

    if pretrained:
        # print(f'Loading {config.pretrained_model}')
        repo = "https://github.com/NielsRLorenzen/AcceleRest/" 
        release = "releases/download/v0.1.1/"
        weights = "accelerest_sleepstage_lstmc_weights.pt"
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
        release = "releases/download/v0.1.1/"
        weights = "accelerest_respevent_linear_weights.pt"
        url = repo + release + weights
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True) 
    
    return model

def accelerest_multihead(
    linear_sleepstage:bool = False,
    lstm_sleepstage:bool = False,
    linear_respevent:bool = False,
    pretrained:bool = True,
):
    heads = []
    names = []
    if linear_sleepstage:
        heads.append(accelerest_sleepstage(pretrained = pretrained))
        names.append('linear_sleepstages')
    if lstm_sleepstage:
        heads.append(accelerest_sleepstage_lstmc(pretrained = pretrained))
        names.append('lstm_sleepstages')
    if linear_respevent:
        heads.append(accelerest_respevent(pretrained = pretrained))
        names.append('linear_respevents')
    if len(heads) == 0:
        raise RuntimeError('Specify at least one prediction head.')
    model = MultiHeadAcceleRest(heads, names)

    return model

class MultiHeadAcceleRest(nn.Module):
    def __init__(self, models = list, names = list):
        super().__init__()
        #Ensure identical encoder backbones
        for i in range(len(models)-1):
            if not identical_backbone(models[i].state_dict(), models[i+1].state_dict()):
                raise RuntimeError(f"The two AcceleRest models at {i} and {i+1} do not share identical encoders.")
        
        self.patch_size = models[0].patch_size
        self.max_seq_len = models[0].max_seq_len

        self.patch_embedding = models[0].patch_embedding
        self.encoder = models[0].encoder
        self.names = names
        self.heads = [self.get_head(model) for model in models]
    
    def get_head(model):
        if hasattr(model, 'lstm'):
            head = nn.Sequential(
                sleepstage_model_lstmc.norm,
                sleepstage_model_lstmc.pre_lstm,
                sleepstage_model_lstmc.lstm,
                sleepstage_model_lstmc.classification_head,
            )
        else:
            head = nn.Sequential(
            sleepstage_model.norm,
            sleepstage_model.classification_head,
        )
        return head

    def forward(self, x):
        features = self.encoder(self.patch_embedding(x), use_sdpa=True)

        outputs = dict()
        for i, head in enumerate(heads):
            outputs[self.names[i]] = head(features)

        return outputs

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