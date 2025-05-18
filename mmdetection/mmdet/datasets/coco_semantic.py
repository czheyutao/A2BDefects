# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .ade20k import ADE20KSegDataset


@DATASETS.register_module()
class CocoSegDataset(ADE20KSegDataset):
    """COCO dataset.

    In segmentation map annotation for COCO. The ``img_suffix`` is fixed to
    '.jpg',  and ``seg_map_suffix`` is fixed to '.png'.
    """

    METAINFO = dict(
        classes=(("ML", "DD", "CR", "SS", "EF", "IR", "BI", "MJ")),
        palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)]
    )
