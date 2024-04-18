# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .vit_comer.mmdet_custom.models.backbones.vit_comer import ViTCoMer
from detectron2.config import configurable

_logger = logging.getLogger(__name__)

def load_pretrained_weights(model, pretrained_weights):
    weight = torch.load(pretrained_weights)
    interpolated_weight = F.interpolate(weight['patch_embed.proj.weight'],
                                        size=(16, 16), mode='bilinear',
                                        align_corners=True)
    weight['patch_embed.proj.weight'] = interpolated_weight
    pre_keys = weight.keys()
    cur_keys = model.state_dict().keys()
    for key in pre_keys:
        assert key in cur_keys, key
    model.load_state_dict(weight, strict=False)
    print("Successfully loaded the DINO V2 pre-trained weights !!!")
    return

@BACKBONE_REGISTRY.register()
class D2VitComerAdapterDinoV2(ViTCoMer, Backbone):
    @configurable
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_CTI=True,
                 use_CTI_toV=True,
                 use_CTI_toC=True,
                 cnn_feature_interaction=False,
                 extra_num=4,
                 out_embeds=[1024, 1024, 1024, 1024, ],
                 pretrain=None,
                 *args, **kwargs):

        super().__init__(
            pretrain_size=pretrain_size,
            num_heads=num_heads,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            init_values=init_values,
            interaction_indexes=interaction_indexes,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_CTI=use_extra_CTI,
            use_CTI_toC=use_CTI_toC,
            use_CTI_toV=use_CTI_toV,
            cnn_feature_interaction=cnn_feature_interaction,
            extra_num=extra_num,
            *args, **kwargs)

        self._out_features = ['res2', 'res3', 'res4', 'res5']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

        self._out_feature_channels = {
            "res2": out_embeds[0],
            "res3": out_embeds[1],
            "res4": out_embeds[2],
            "res5": out_embeds[3],
        }
        self.out_embeds = out_embeds

        if pretrain is not None:
            load_pretrained_weights(self, pretrain)

    @classmethod
    def from_config(cls, cfg):

        img_size = 592,
        patch_size = 16,
        embed_dim = 768,
        depth = 12,
        mlp_ratio = 4,
        drop_path_rate = 0.3,

        window_attn = [True, True, False, True, True, False,
                       True, True, False, True, True, False],
        window_size = [14, 14, None, 14, 14, None,
                       14, 14, None, 14, 14, None],

        img_size = cfg.MODEL.BACKBONE.IMAGE_SIZE
        patch_size = cfg.MODEL.BACKBONE.PATCH_SIZE
        embed_dim = cfg.MODEL.BACKBONE.EMBED_DIM
        depth = cfg.MODEL.BACKBONE.DEPTH
        mlp_ratio = cfg.MODEL.MLP_RATIO
        drop_path_rate = cfg.MODEL.DROP_PATH_RATE
        window_attn = cfg.MODEL.WINDOW_ATTN
        window_size = cfg.MODEL.WINDOW_SIZE

        pretrain_size = cfg.MODEL.BACKBONE.PRETRAIN_SIZE
        num_heads = cfg.MODEL.BACKBONE.NUM_HEADS
        conv_inplane = cfg.MODEL.BACKBONE.CONV_INPLANE
        n_points = cfg.MODEL.BACKBONE.N_POINTS
        deform_num_heads = cfg.MODEL.BACKBONE.DEFORM_NUM_HEADS
        init_values = cfg.MODEL.BACKBONE.INIT_VALUES,
        interaction_indexes = cfg.MODEL.BACKBONE.INTERACTION_INDEXES
        with_cffn = cfg.MODEL.BACKBONE.WITH_CFFN
        cffn_ratio = cfg.MODEL.BACKBONE.CFFN_RATIO
        deform_ratio = cfg.MODEL.BACKBONE.DEFORM_RATIO
        add_vit_feature = cfg.MODEL.BACKBONE.ADD_VIT_FEATURE
        use_extra_CTI = cfg.MODEL.BACKBONE.USE_EXTRA_CTI
        use_CTI_toV = cfg.MODEL.BACKBONE.USE_CTI_TOV
        use_CTI_toC = cfg.MODEL.BACKBONE.USE_CTI_TOC
        cnn_feature_interaction = cfg.MODEL.BACKBONE.CNN_FEATURE_INTERACTION
        extra_num = cfg.MODEL.BACKBONE.EXTRA_NUM
        out_embeds = cfg.MODEL.BACKBONE.OUT_EMBEDS
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN

        return {
            "pretrain_size": pretrain_size,
            "img_size": img_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "mlp_ratio": mlp_ratio,
            "drop_path_rate": drop_path_rate,
            "window_attn": window_attn,
            "window_size": window_size,
            "num_heads": num_heads,
            "conv_inplane": conv_inplane,
            "n_points": n_points,
            "deform_num_heads": deform_num_heads,
            "init_values": init_values,
            "interaction_indexes": interaction_indexes,
            "with_cffn": with_cffn,
            "cffn_ratio": cffn_ratio,
            "deform_ratio": deform_ratio,
            "add_vit_feature": add_vit_feature,
            "use_extra_CTI": use_extra_CTI,
            "use_CTI_toV": use_CTI_toV,
            "use_CTI_toC": use_CTI_toC,
            "cnn_feature_interaction": cnn_feature_interaction,
            "extra_num": extra_num,
            "out_embeds": out_embeds,
            "pretrain": pretrain,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"

        outputs = {}
        y = super().forward(x)
        for i, k in enumerate(self._out_feature_strides.keys()):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.out_embeds, stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32