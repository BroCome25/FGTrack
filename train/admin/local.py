class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/zx/projects/FGTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/zx/projects/FGTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/zx/projects/FGTrack/pretrained_networks'
        self.lasot_dir = '/zx/datasets/lasot'
        self.got10k_dir = '/zx/datasets/got10k/train'
        self.got10k_val_dir = '/zx/datasets/got10k/val'
        self.lasot_lmdb_dir = '/zx/datasets/lasot_lmdb'
        self.got10k_lmdb_dir = '/zx/datasets/got10k_lmdb'
        self.trackingnet_dir = '/zx/datasets/trackingnet'
        self.trackingnet_lmdb_dir = '/zx/datasets/trackingnet_lmdb'
        self.coco_dir = '/zx/datasets/coco'
        self.coco_lmdb_dir = '/zx/datasets/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/zx/datasets/vid'
        self.imagenet_lmdb_dir = '/zx/datasets/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
