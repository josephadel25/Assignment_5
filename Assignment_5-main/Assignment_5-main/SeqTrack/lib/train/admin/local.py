# class EnvironmentSettings:
#     def __init__(self):
#         self.workspace_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack'    # Base directory for saving network checkpoints.
#         self.tensorboard_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\tensorboard'    # Directory for tensorboard files.
#         self.pretrained_networks = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\pretrained_networks'
#         self.lasot_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot'
#         self.got10k_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k'
#         self.lasot_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot_lmdb'
#         self.got10k_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k_lmdb'
#         self.trackingnet_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\trackingnet'
#         self.trackingnet_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\trackingnet_lmdb'
#         self.coco_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\coco'
#         self.coco_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\coco_lmdb'
#         self.imagenet1k_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\imagenet1k'
#         self.imagenet22k_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\imagenet22k'
#         self.lvis_dir = ''
#         self.sbd_dir = ''
#         self.imagenet_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\vid'
#         self.imagenet_lmdb_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\vid_lmdb'
#         self.imagenetdet_dir = ''
#         self.ecssd_dir = ''
#         self.hkuis_dir = ''
#         self.msra10k_dir = ''
#         self.davis_dir = ''
#         self.youtubevos_dir = ''

class EnvironmentSettings:
    def __init__(self):
        # Base path where your repo is cloned
        base = "/kaggle/working/SeqTrack"

        # Writable directories for training artifacts
        self.workspace_dir = base  # checkpoints, logs, etc.
        self.tensorboard_dir = f"{base}/tensorboard"
        self.pretrained_networks = f"{base}/pretrained_networks"

        # Data directories
        data = "/kaggle/temp"  # where LaSOT and other datasets are downloaded
        self.lasot_dir = f"{data}/LaSOT_partial"
        self.got10k_dir = f"{data}/got10k"
        self.lasot_lmdb_dir = f"{data}/lasot_lmdb"
        self.got10k_lmdb_dir = f"{data}/got10k_lmdb"
        self.trackingnet_dir = f"{data}/trackingnet"
        self.trackingnet_lmdb_dir = f"{data}/trackingnet_lmdb"
        self.coco_dir = f"{data}/coco"
        self.coco_lmdb_dir = f"{data}/coco_lmdb"
        self.imagenet1k_dir = f"{data}/imagenet1k"
        self.imagenet22k_dir = f"{data}/imagenet22k"
        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenet_dir = f"{data}/vid"
        self.imagenet_lmdb_dir = f"{data}/vid_lmdb"
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""

