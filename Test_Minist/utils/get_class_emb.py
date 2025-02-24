import torch
import numpy as np
import os
import os.path as osp
from utils.labels_dict import UNI_UID2UNAME, ALL_LABEL2ID, UNAME2EM_NAME
UNI_UNAME2ID = {v: i for i, v in UNI_UID2UNAME.items()}

def create_embs_from_names(labels, other_descriptions=None):
    import clip
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_TEXT_MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE, jit=False)
    u_descrip_dir = 'data/clip_descriptions'
    embs = []
    for name in labels:
        if name in UNAME2EM_NAME.keys():
            with open(os.path.join(u_descrip_dir, UNAME2EM_NAME[name] + '.txt'), 'r') as f:
                description = f.readlines()[0]
        elif name in other_descriptions:
            description = other_descriptions[name]

        text = clip.tokenize([description, ]).to(DEVICE)
        with torch.no_grad():
            text_features = CLIP_TEXT_MODEL.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        embs.append(text_features)
    embs = torch.stack(embs, dim=0).squeeze()
    return embs

