import PIL.Image as Image
from itertools import product
import clip
import torch.nn as nn
import torch
import random
import numpy as np
from collections import defaultdict
import os
import torchvision.models as models
from transformers import AutoModel, AutoProcessor
import pyiqa
from torchvision.transforms import RandomHorizontalFlip
os.environ["TOKENIZERS_PARALLELISM"] = "false"
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
distortions = ['noise', 'blur', 'color-imbalance', 'over-exposure', 'under-exposure', 'over-saturated', 'under-saturated',
               'white-balance', 'over-sharp', 'other color-related', 'none']

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ViTSigLIP(torch.nn.Module):
    def __init__(self):
        super(ViTSigLIP, self).__init__()

        # self.siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        # self.siglip_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        self.siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex")
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

        for param_name, param in self.siglip_model.named_parameters():
            if param_name in ['logit_scale', 'logit_bias']:
                param.requires_grad = False

        self.all_texts = [f"a photo with {d} artifacts, which is of {q} quality" for q, d in
                          product(qualitys, distortions)]

    def forward(self, x):
        device = x.device

        inputs = self.siglip_processor(
            text=self.all_texts,
            images=None,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,  # Ensures text is truncated if it exceeds max length
        ).to(device)
        bsz = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        inputs['pixel_values'] = x
        outputs = self.siglip_model(**inputs)
        logits_per_image = outputs.logits_per_image

        logits_per_image = logits_per_image.view(bsz, -1, logits_per_image.size(1))
        logits_per_image = logits_per_image.mean(1)

        logits_per_image = torch.softmax(logits_per_image, dim=1)
        logits_per_image = logits_per_image.view(-1, len(qualitys), len(distortions))
        y_pred = logits_per_image.sum(2)
        logits_distortion = logits_per_image.sum(1)
        y_pred = 1 * y_pred[:, 0] + 2 * y_pred[:, 1] + 3 * y_pred[:, 2] + 4 * y_pred[:, 3] + 5 * y_pred[:, 4]
        y_pred = y_pred

        distortion_preds = logits_distortion

        return y_pred, distortion_preds

flip = RandomHorizontalFlip(p=0.5)

class ViTSigLIP_naflex(torch.nn.Module):
    def __init__(self):
        super(ViTSigLIP_naflex, self).__init__()

        self.siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex")
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

        for param_name, param in self.siglip_model.named_parameters():
            if param_name in ['logit_scale', 'logit_bias']:
                param.requires_grad = False

        self.all_texts = [f"a photo with {d} artifacts, which is of {q} quality" for q, d in
                          product(qualitys, distortions)]

    def forward(self, x, flip=False):
        #device = x.device
        bsz = len(x)
        
        if flip:
            x = [flip(img) for img in x]

        inputs1 = self.siglip_processor(
            text=self.all_texts,
            images=x,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            max_num_patches=196,
            truncation=True,  # Ensures text is truncated if it exceeds max length
        )

        inputs2 = self.siglip_processor(
            text=self.all_texts,
            images=x,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            max_num_patches=529,
            truncation=True,  # Ensures text is truncated if it exceeds max length
        )

        inputs3 = self.siglip_processor(
            text=self.all_texts,
            images=x,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            max_num_patches=1024,
            truncation=True,  # Ensures text is truncated if it exceeds max length
        )


        inputs1 = inputs1.to(self.siglip_model.device)
        inputs2 = inputs2.to(self.siglip_model.device)
        inputs3 = inputs3.to(self.siglip_model.device)
        outputs1 = self.siglip_model(**inputs1)
        outputs2 = self.siglip_model(**inputs2)
        outputs3 = self.siglip_model(**inputs3)

        logits_per_image1 = outputs1.logits_per_image
        logits_per_image2 = outputs2.logits_per_image
        logits_per_image3 = outputs3.logits_per_image

        logits_per_image = torch.cat([logits_per_image1.unsqueeze(1), logits_per_image2.unsqueeze(1), logits_per_image3.unsqueeze(1)], 1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = torch.softmax(logits_per_image, dim=1)
        logits_per_image = logits_per_image.view(-1, len(qualitys), len(distortions))
        y_pred = logits_per_image.sum(2)
        logits_distortion = logits_per_image.sum(1)
        y_pred = 1 * y_pred[:, 0] + 2 * y_pred[:, 1] + 3 * y_pred[:, 2] + 4 * y_pred[:, 3] + 5 * y_pred[:, 4]
        distortion_preds = logits_distortion

        return y_pred, distortion_preds


class ViTSigLIP2(torch.nn.Module):
    def __init__(self, sr=True, dropout_sp=0.2):
        super(ViTSigLIP2, self).__init__()

        self.siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        # self.siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
        # self.siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
        # self.siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-naflex")  #TODO: adapt dataloader
        # self.siglip_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex") #TODO: adapt dataloader

        self.load_state_dict(torch.load('checkpoints/1/liqe_llie3.pt')['model_state_dict'])

        for param_name, param in self.siglip_model.named_parameters():
            param.requires_grad = False

        self.dropout_sp = dropout_sp

        self.spatialRec1 = self.spatial_rectifier(16384, self.dropout_sp)

        self.sr = sr
        self.all_texts = [f"a photo with {d} artifacts, which is of {q} quality" for q, d in
                          product(qualitys, distortions)]

    def spatial_rectifier(self, in_channels, dropout_sp):
        '''
            return batch_size * 2
        '''
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp),
        )
        return regression_block

    def forward(self, x, spa_feat):
        device = x.device

        inputs = self.siglip_processor(
            text=self.all_texts,
            images=None,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,  # Ensures text is truncated if it exceeds max length
        ).to(device)

        inputs['pixel_values'] = x
        outputs = self.siglip_model(**inputs)
        logits_per_image = outputs.logits_per_image

        # image_embeds = outputs.image_embeds
        # text_embeds = outputs.text_embeds
        # logits_per_image = torch.matmul(image_embeds, text_embeds.t())
        #
        # logit_scale, logit_bias = self.siglip_model.logit_scale.to(text_embeds.device), self.siglip_model.logit_bias.to(text_embeds.device)
        # logits_per_image = logits_per_image * logit_scale.exp()


        logits_per_image = torch.softmax(logits_per_image, dim=1)
        logits_per_image = logits_per_image.view(-1, len(qualitys), len(distortions))
        y_pred = logits_per_image.sum(2)
        logits_distortion = logits_per_image.sum(1)
        y_pred = 1 * y_pred[:, 0] + 2 * y_pred[:, 1] + 3 * y_pred[:, 2] + 4 * y_pred[:, 3] + 5 * y_pred[:, 4]
        y_pred = y_pred.unsqueeze(1)

        ones = torch.ones_like(y_pred)

        if self.sr:
            lp_size = spa_feat.shape
            spa_feat = spa_feat.view(lp_size[0], -1)
            spatial_s = self.spatialRec1(spa_feat)
            # ax+b
            alphaS = torch.chunk(spatial_s, 2, dim=1)[0]
            alphaS = torch.add(alphaS, ones)
            betaS = torch.chunk(spatial_s, 2, dim=1)[1]
        else:
            alphaS = torch.ones_like(y_pred)
            betaS = torch.zeros_like(y_pred)

        y_pred = torch.add(torch.mul(torch.abs(alphaS), y_pred), betaS).squeeze(1)

        distortion_preds = logits_distortion

        # ones = torch.ones_like(logits_per_image)
        #
        # if self.sr:
        #     lp_size = spa_feat.shape
        #     spa_feat = spa_feat.view(lp_size[0], -1)
        #     spatial_s = self.spatialRec1(spa_feat)
        #     # ax+b
        #     alphaS = torch.chunk(spatial_s, 2, dim=1)[0]
        #     alphaS = torch.add(alphaS, ones)
        #     betaS = torch.chunk(spatial_s, 2, dim=1)[1]
        # else:
        #     alphaS = torch.ones_like(logits_per_image)
        #     betaS = torch.zeros_like(logits_per_image)
        #
        # logits_per_image = torch.add(torch.mul(torch.abs(alphaS), logits_per_image), betaS).squeeze(1)
        # logits_per_image = torch.softmax(logits_per_image, dim=1)
        # logits_per_image = logits_per_image.view(-1, len(qualitys), len(distortions))
        #
        # y_pred = logits_per_image.sum(2)
        # logits_distortion = logits_per_image.sum(1)
        #
        # y_pred = 1 * y_pred[:, 0] + 2 * y_pred[:, 1] + 3 * y_pred[:, 2] + 4 * y_pred[:, 3] + 5 * y_pred[:, 4]
        # distortion_preds = logits_distortion

        return y_pred, distortion_preds

class CrossAttentionFuse(nn.Module):
    def __init__(self, dim_q=768, dim_kv=128, dim_common=128, heads=8):
        super().__init__()
        self.query_proj = nn.Linear(dim_q, dim_common)   # 映射 Query（512 -> 256）
        self.key_proj   = nn.Linear(dim_kv, dim_common)  # 映射 Key（128 -> 256）
        self.value_proj = nn.Linear(dim_kv, dim_common)  # 映射 Value（128 -> 256）

        self.attn = nn.MultiheadAttention(embed_dim=dim_common, num_heads=heads, batch_first=True)

        self.output_proj = nn.Linear(dim_common, dim_q)  # 融合后再映射回原始 Query 维度（可选）

    def forward(self, feat_kv, feat_q):
        """
        feat_kv: [B, D_kv]   (比如 128)
        feat_q:  [B, D_q]    (比如 768)
        """
        # [B, D] -> [B, 1, D]
        q = self.query_proj(feat_q).unsqueeze(1)      # [B, 1, 256]
        k = self.key_proj(feat_kv).unsqueeze(1)       # [B, 1, 256]
        v = self.value_proj(feat_kv).unsqueeze(1)     # [B, 1, 256]

        out, _ = self.attn(q, k, v)                   # [B, 1, 256]
        out = self.output_proj(out).squeeze(1)        # [B, 768]
        return out + feat_q


class ViTSigLIP_cross(torch.nn.Module):
    def __init__(self):
        super(ViTSigLIP_cross, self).__init__()

        self.siglip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        self.siglip_processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

        model = pyiqa.create_metric('dbcnn', as_loss=False)
        self.features = model.net.features2
        for p in self.features.parameters():
            p.requires_grad = False

        for param_name, param in self.siglip_model.named_parameters():
            if param_name in ['logit_scale', 'logit_bias']:
                param.requires_grad = False
        self.all_texts = [f"a photo with {d} artifacts, which is of {q} quality" for q, d in
                          product(qualitys, distortions)]

        self.fusion = CrossAttentionFuse()

        imagenet_mean=[0.485, 0.456, 0.406]
        imagenet_std=[0.229, 0.224, 0.225]

        hf_mean=[0.5, 0.5, 0.5]
        hf_std=[0.5, 0.5, 0.5]

        self.imagenet_mean = torch.Tensor(imagenet_mean).view(1, 3, 1, 1)
        self.imagenet_std = torch.Tensor(imagenet_std).view(1, 3, 1, 1)

        self.hf_mean = torch.Tensor(hf_mean).view(1, 3, 1, 1)
        self.hf_std = torch.Tensor(hf_std).view(1, 3, 1, 1)

    def forward(self, x):
        device = x.device

        inputs = self.siglip_processor(
            text=self.all_texts,
            images=None,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,  # Ensures text is truncated if it exceeds max length
        ).to(device)
        bsz = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        x1 = (x - self.hf_mean.to(x)) / self.hf_std.to(x)
        x2 = (x - self.imagenet_mean.to(x)) / self.imagenet_std.to(x)

        scnn_feat = self.features(x2).mean(3).mean(2)

        inputs['pixel_values'] = x1
        outputs = self.siglip_model(**inputs)
        #logits_per_image = outputs.logits_per_image

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = self.fusion(scnn_feat, image_embeds)
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

        logit_scale, logit_bias = self.siglip_model.logit_scale.to(text_embeds.device), self.siglip_model.logit_bias.to(text_embeds.device)
        logits_per_image = logits_per_image * logit_scale.exp()

        logits_per_image = logits_per_image.view(bsz, -1, logits_per_image.size(1))
        logits_per_image = logits_per_image.mean(1)

        logits_per_image = torch.softmax(logits_per_image, dim=1)
        logits_per_image = logits_per_image.view(-1, len(qualitys), len(distortions))
        y_pred = logits_per_image.sum(2)
        logits_distortion = logits_per_image.sum(1)
        y_pred = 1 * y_pred[:, 0] + 2 * y_pred[:, 1] + 3 * y_pred[:, 2] + 4 * y_pred[:, 3] + 5 * y_pred[:, 4]
        y_pred = y_pred

        distortion_preds = logits_distortion

        return y_pred, distortion_preds
