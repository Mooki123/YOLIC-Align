# -*- coding: utf-8 -*-
"""YOLIC-Align: a cell mask-pooling head for YOLIC, plus helpers to build and load every head."""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import yolic_cells

INPUT_SIZE = 224
FEATURE_SIZE = 14  # stride-16 grid of MobileNetV2 at 224x224 input


def _splat_matrix(num_pixels, num_bins):
    """RoIAlign-style bilinear weights of shape (num_bins, num_pixels); each pixel's weights sum to 1."""
    centers = (torch.arange(num_pixels, dtype=torch.float64) + 0.5) * num_bins / num_pixels - 0.5
    centers = centers.clamp(0, num_bins - 1)
    bins = torch.arange(num_bins, dtype=torch.float64)
    return (1 - (bins[:, None] - centers[None, :]).abs()).clamp(min=0)


def _coverage(start, end, num_pixels):
    """Fraction of each pixel covered by the continuous interval [start, end)."""
    pixels = torch.arange(num_pixels, dtype=torch.float64)
    return ((pixels + 1).clamp(max=end) - pixels.clamp(min=start)).clamp(min=0)


def _normalize(weights):
    area = weights.sum(dim=(1, 2), keepdim=True)
    if (area == 0).any():
        raise ValueError('a cell does not cover any pixel of the input image')
    return (weights / area).float()


def rect_cell_weights(cells, original_size, feature_size=FEATURE_SIZE, input_size=INPUT_SIZE):
    """Pooling weights (N, feature_size, feature_size) for rectangles [(x1, y1), (x2, y2)].

    Rectangles are separable, so pixel coverage is computed exactly (no integer truncation).
    """
    splat = _splat_matrix(input_size, feature_size)
    scale_x = input_size / original_size[0]
    scale_y = input_size / original_size[1]
    weights = []
    for (x1, y1), (x2, y2) in cells:
        col = splat @ _coverage(x1 * scale_x, x2 * scale_x, input_size)
        row = splat @ _coverage(y1 * scale_y, y2 * scale_y, input_size)
        weights.append(torch.outer(row, col))
    return _normalize(torch.stack(weights))


def polygon_cell_weights(polygons, original_size, feature_size=FEATURE_SIZE, input_size=INPUT_SIZE, supersample=4):
    """Pooling weights (N, feature_size, feature_size) for polygons [x1, y1, x2, y2, ...]."""
    size = input_size * supersample
    splat = _splat_matrix(size, feature_size)
    scale = np.array([size / original_size[0], size / original_size[1]])
    weights = []
    for polygon in polygons:
        points = np.round(np.array(polygon, dtype=np.float64).reshape(-1, 2) * scale).astype(np.int32)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        weights.append(splat @ torch.from_numpy(mask).double() @ splat.T)
    return _normalize(torch.stack(weights))


CELL_WEIGHTS = {
    'indoor': lambda: polygon_cell_weights(yolic_cells.INDOOR_POLYGONS, yolic_cells.INDOOR_SIZE),
    'outdoor': lambda: rect_cell_weights(yolic_cells.OUTDOOR_CELLS, yolic_cells.OUTDOOR_SIZE),
    'cityscapes': lambda: rect_cell_weights(yolic_cells.CITYSCAPES_CELLS, yolic_cells.CITYSCAPES_SIZE),
}


class YolicAlignModel(nn.Module):
    """MobileNetV2 backbone with a cell mask-pooling head shared across all cells.

    The stride-16 features (96 ch) and the upsampled stride-32 features (1280 ch) are fused into a
    14x14 map, and a 1x1 conv turns it into per-location logits. Each cell reads those logits out
    through its bilinear mask weights. Per class, the readout mixes a weighted mean with a weighted
    log-sum-exp (a soft max), because a cell label is positive when any pixel of the class lies in
    the cell. A learnable per-cell bias keeps the location prior of the original per-cell FC head.
    """

    def __init__(self, cell_weights, num_outputs, pretrained=True, lateral_channels=128, dropout=0.2,
                 lse_temperature=4.0):
        super(YolicAlignModel, self).__init__()
        features = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None).features
        self.stride16 = features[:14]
        self.stride32 = features[14:]
        self.lateral = nn.Sequential(nn.Conv2d(96, lateral_channels, 1, bias=False),
                                     nn.BatchNorm2d(lateral_channels),
                                     nn.ReLU6(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(1280 + lateral_channels, num_outputs, 1)
        self.cell_bias = nn.Parameter(torch.zeros(cell_weights.shape[0], num_outputs))
        self.mix = nn.Parameter(torch.zeros(num_outputs))
        self.lse_temperature = lse_temperature
        # Derived from the cell configuration, so they are rebuilt rather than stored in checkpoints.
        flat = cell_weights.flatten(1)
        self.register_buffer('cell_weights', flat, persistent=False)
        self.register_buffer('log_cell_weights', flat.log(), persistent=False)  # -inf outside the cell

    def forward(self, x):
        x16 = self.stride16(x)
        x32 = self.stride32(x16)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x32, self.lateral(x16)], dim=1)
        logit_map = self.classifier(self.dropout(fused)).flatten(2)  # B x K x HW
        if logit_map.shape[-1] != self.cell_weights.shape[-1]:
            raise ValueError('expected a {}x{} input image'.format(INPUT_SIZE, INPUT_SIZE))

        mean = torch.einsum('bkp,np->bnk', logit_map, self.cell_weights)
        t = self.lse_temperature
        soft_max = torch.logsumexp(t * logit_map.unsqueeze(2) + self.log_cell_weights, dim=-1) / t
        soft_max = soft_max.transpose(1, 2)
        logits = mean + torch.sigmoid(self.mix) * (soft_max - mean) + self.cell_bias
        return logits.flatten(1)  # B x (N * K), cell-major like the label files


LEGACY_FEATURE_SIZE = 7  # stride-32 grid the first YOLIC-Align checkpoints pooled over

LEGACY_CELLS = {
    'indoor': (yolic_cells.INDOOR_POLYGONS, yolic_cells.INDOOR_SIZE, True),
    'outdoor': (yolic_cells.OUTDOOR_CELLS, yolic_cells.OUTDOOR_SIZE, False),
    'cityscapes': (yolic_cells.CITYSCAPES_CELLS, yolic_cells.CITYSCAPES_SIZE, False),
}


def legacy_cell_masks(dataset, feature_size=LEGACY_FEATURE_SIZE, input_size=INPUT_SIZE):
    """Cell masks exactly as the first YOLIC-Align training scripts built them.

    Cell corners are truncated to whole input pixels, rasterised, then average-pooled down to the
    stride-32 grid, so the masks are coarser than rect_cell_weights/polygon_cell_weights. Kept so the
    checkpoints trained with that head still load and predict the way they did.
    """
    cells, original_size, polygon = LEGACY_CELLS[dataset]
    scale = np.array([input_size / original_size[0], input_size / original_size[1]])
    masks = np.zeros((len(cells), input_size, input_size), dtype=np.float32)
    for mask, cell in zip(masks, cells):
        points = np.int32(np.array(cell, dtype=np.float32).reshape(-1, 2) * scale)
        if polygon:
            cv2.fillPoly(mask, [points], 1.0)
        else:
            cv2.rectangle(mask, tuple(int(v) for v in points[0]), tuple(int(v) for v in points[1]), 1.0, -1)
    return F.adaptive_avg_pool2d(torch.from_numpy(masks), feature_size)


class YolicMaskPoolModel(nn.Module):
    """The first YOLIC-Align head: mask-average-pool the stride-32 features per cell, then a shared FC.

    Superseded by YolicAlignModel, but every checkpoint trained before the stride-16 fusion head
    existed carries these weights (and its masks, hence the persistent buffer).
    """

    def __init__(self, masks, num_outputs, pretrained=True):
        super(YolicMaskPoolModel, self).__init__()
        self.features = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None).features
        self.classifier = nn.Linear(1280, num_outputs)
        self.register_buffer('fractional_masks', masks)

    def forward(self, x):
        feat = self.features(x).flatten(2)  # B x C x HW
        masks = self.fractional_masks.flatten(1)  # N x HW
        if feat.shape[-1] != masks.shape[-1]:
            raise ValueError('expected a {}x{} input image'.format(INPUT_SIZE, INPUT_SIZE))
        pooled = torch.einsum('bcp,np->bnc', feat, masks) / (masks.sum(-1) + 1e-6).unsqueeze(-1)
        return self.classifier(pooled).flatten(1)  # B x (N * K), cell-major like the label files


ARCHS = ('align', 'align-v1', 'baseline')
ARCH_HELP = ('align: cell mask-pooling head; align-v1: the first mask-pooling head (stride-32 pooling '
             'and a shared FC); baseline: original GAP + per-cell FC head')
ARCH_SUFFIX = {'align': '_align', 'align-v1': '_alignv1', 'baseline': ''}


def add_arch_argument(parser, inference=False):
    """--arch for the training scripts; the prediction and evaluation scripts also get auto and --weights."""
    if inference:
        parser.add_argument('--arch', choices=('auto',) + ARCHS, default='auto',
                            help='auto: the head the checkpoint was trained with; ' + ARCH_HELP)
        parser.add_argument('--weights', default=None,
                            help='checkpoint to load (default: the one named for --arch)')
    else:
        parser.add_argument('--arch', choices=ARCHS, default='align', help=ARCH_HELP)


def build_model(arch, dataset, num_cells, num_outputs, pretrained=False):
    if arch == 'baseline':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(1280, num_cells * num_outputs)
        return model
    if arch == 'align-v1':
        masks = legacy_cell_masks(dataset)
        _check_cells(dataset, masks.shape[0], num_cells)
        return YolicMaskPoolModel(masks, num_outputs, pretrained=pretrained)
    cell_weights = CELL_WEIGHTS[dataset]()
    _check_cells(dataset, cell_weights.shape[0], num_cells)
    return YolicAlignModel(cell_weights, num_outputs, pretrained=pretrained)


def _check_cells(dataset, found, expected):
    if found != expected:
        raise ValueError('{} has {} cells, expected {}'.format(dataset, found, expected))


def checkpoint_path(path, arch):
    """Every head keeps its weights next to the baseline ones under its own suffix."""
    suffix = ARCH_SUFFIX[arch]
    if not suffix:
        return path
    for ext in ('.pth.tar', '.pth'):
        if path.endswith(ext):
            return path[:-len(ext)] + suffix + ext
    return path + suffix


def detect_arch(state_dict):
    """Which head a checkpoint was trained with, read off the keys it carries."""
    if 'fractional_masks' in state_dict:
        return 'align-v1'
    if 'cell_bias' in state_dict:
        return 'align'
    if 'classifier.1.weight' in state_dict:
        return 'baseline'
    raise ValueError('cannot tell which head this checkpoint holds, pass --arch explicitly')


def resolve_checkpoint(path, arch):
    """The file to load: the name arch saves under, else the plain name, which predates the suffixes."""
    candidates = [checkpoint_path(path, a) for a in (ARCHS if arch == 'auto' else (arch,))] + [path]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def load_model(args, dataset, num_cells, num_outputs, default_path):
    """Build the head the checkpoint was trained with and load it, so old and new weights both run."""
    path = getattr(args, 'weights', None) or resolve_checkpoint(default_path, args.arch)
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    arch = detect_arch(state) if args.arch == 'auto' else args.arch
    model = build_model(arch, dataset, num_cells, num_outputs)
    model.load_state_dict(state)
    print('loaded {} into the {} head'.format(path, arch))
    return model
