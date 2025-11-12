from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .imagenetvid import ImagenetVID
from .coco import MSCOCO
from .coco_seq import MSCOCOSeq
from .imagenet1k import Imagenet1k
from .imagenet22k import Imagenet22k

# Import LMDB-backed datasets only if lmdb is available.
try:
    import lmdb as _lmdb  # noqa: F401
    from .got10k_lmdb import Got10k_lmdb
    from .lasot_lmdb import Lasot_lmdb
    from .imagenetvid_lmdb import ImagenetVID_lmdb
    from .coco_seq_lmdb import MSCOCOSeq_lmdb
    from .tracking_net_lmdb import TrackingNet_lmdb
except Exception:
    class _LMDBNotAvailable:
        def __init__(self, *args, **kwargs):
            raise ImportError("lmdb is not installed. Install `lmdb` or run with --use_lmdb 0.")
    Got10k_lmdb = _LMDBNotAvailable  # type: ignore
    Lasot_lmdb = _LMDBNotAvailable  # type: ignore
    ImagenetVID_lmdb = _LMDBNotAvailable  # type: ignore
    MSCOCOSeq_lmdb = _LMDBNotAvailable  # type: ignore
    TrackingNet_lmdb = _LMDBNotAvailable  # type: ignore
