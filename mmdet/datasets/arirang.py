from .coco import CocoDataset


class ArirangDataset(CocoDataset):
    CLASSES = ('small ship', 'large ship', 'civilian aircraft', 'military aircraft', 'small car',
               'bus', 'truck', 'train', 'crane', 'bridge', 'oil tank', 'dam', 'athletic field',
               'helipad', 'roundabout', 'etc')
