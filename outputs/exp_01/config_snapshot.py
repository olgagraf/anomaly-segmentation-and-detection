import os

# ---------------------------------------------------------------------------
# Experiment info
# ---------------------------------------------------------------------------

EXP_NAME = 'exp'
EXP_NUMBER = '01'

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATASET_ROOT_DIR = '/mnt/ograf/data'  # directory that contains folders 'img', 'mask', and 'split'
PROJECT_ROOT_DIR = '/mnt/ograf/anomaly-segmentation-and-detection'  # root directory of this project, used for saving outputs and model weights

MASK_DIR = os.path.join(DATASET_ROOT_DIR, 'mask')
IMG_DIR = os.path.join(DATASET_ROOT_DIR, 'img')
SPLIT_DIR = os.path.join(DATASET_ROOT_DIR, 'split')
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'outputs', f'{EXP_NAME}_{EXP_NUMBER}')
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
SAVED_WEIGHTS = os.path.join(MODEL_SAVE_DIR, 'dinov2_lora.pt')

# ---------------------------------------------------------------------------
# GPU settings
# ---------------------------------------------------------------------------

DEVICE = 'cuda:1'
PARALLEL = True  # whether to use DataParallel for multi-GPU training, set to False if using a single GPU

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

IGNORE_INDEX = -1  # marks labels that should not contribute to training (OOD classes)
CLASS_SPECS = {  # tissue classes, healthy/normal class should have key 'normal'
    'apoptosis': {'index': IGNORE_INDEX, 'color': (235, 55, 52)},           # bright red (OOD class)
    'artifact': {'index': IGNORE_INDEX, 'color': (235, 189, 52)},           # yellow-orange (OOD class)
    'ballooning': {'index': 0, 'color': (158, 235, 52)},                    # lime green
    'cyt_vacuolation': {'index': 1, 'color': (235, 89, 160)},               # pink
    'fibrosis': {'index': 2, 'color': (255, 157, 225)},                     # light pink
    'inflammation': {'index': 3, 'color': (52, 235, 235)},                  # turquoise
    'macrosteatosis': {'index': 4, 'color': (52, 100, 235)},                # royal blue
    'microsteatosis': {'index': 5, 'color': (255, 230, 153)},               # pale yellow
    'mitosis': {'index': 6, 'color': (171, 121, 66)},                       # brown
    'necrosis': {'index': 7, 'color': (19, 89, 63)},                        # dark green
    'no_tissue': {'index': 8, 'color': (255, 255, 255)},                    # white
    'normal': {'index': 9, 'color': (248, 203, 173)},                       # peach
}
BACKGROUND_COLOR = (0, 0, 0)  # black, used for unannotated areas in masks
COLOR_MAP = {BACKGROUND_COLOR: IGNORE_INDEX}
COLOR_MAP.update({
    spec['color']: spec['index']
    for spec in CLASS_SPECS.values()
})

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------

# Image normalization (channel-wise) before model training/inference
NORMALIZE_MEAN = (0.5788, 0.3551, 0.5655)
NORMALIZE_STD = (1, 1, 1)

# Our base tile size used for training the segmentation model is 252x252 (which is multiple of transformer patch size 14x14).
# For spatial shift averaging, we define minimal overlap with the central tile to be 42x42 => extended tile dimension is 252x3 - 42x2 = 672
IMG_DIM = (252, 252)  # tile size
EXT_IMG_DIM = (672, 672)  # extended tile size
SHIFT_STEP = 84  # stride (in pixels) used in spatial shift averaging, produced 36 shifts per tile

# ---------------------------------------------------------------------------
# Image crop filter settings
# ---------------------------------------------------------------------------

COLOR_THRESHOLD = 0.01  # accept annotated areas that are >= 1% of tile area
STRICT_THRESHOLD = 0.005  # accept annotated areas of listed classes that are >= 0.5% of tile area
STRICT_THRESHOLD_CLASSES = (  # classes with strict threshold
    "apoptosis",
    "cyt_vacuolation",
    "mitosis",
)
NO_THRESHOLD_CLASSES = (  # accept any annotated areas of listed classes
    "microsteatosis",
)
STRICT_COLOR_LIST = [CLASS_SPECS[name]["color"] for name in STRICT_THRESHOLD_CLASSES]
NO_THRESHOLD_LIST = [CLASS_SPECS[name]["color"] for name in NO_THRESHOLD_CLASSES]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

N_CLASSES = sum(1 for v in COLOR_MAP.values() if v != IGNORE_INDEX)

# DINOv2
BACKBONES = {
    'small': 'vits14_reg',
    'base': 'vitb14_reg',
    'large': 'vitl14_reg',
    'giant': 'vitg14_reg',
}
EMBEDDING_DIMS = {
    'small': 384,
    'base': 768,
    'large': 1024,
    'giant': 1536,
}
SIZE = 'base'

# LoRA
USE_LORA = True
R = 3

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

SEED = 42  # random seed for reproducibility
EPOCHS = 50
LR = 3e-4  # learning rate
BATCH_SIZE = 12

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

NORMALIZE = True  # whether to L2-normalize per-pixel features for Maha+ score
REFERENCE_SPLIT = "trainval"  # split used both for class stats (means/cov) and threshold calibration, choices=["train", "trainval", "val"]
PERC = 0.996  # TPR-like parameter: fraction of pixels on REFERENCE_SPLIT predicted as ID
