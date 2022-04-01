import torch
from torchvision import transforms as tmfs

RAND_SIZE = 512
IN_CHANNELS = 3
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
NUM_CONV = 4
NUM_TRANSPOSE = 5
EPOCHS = 100
NUM_ITERATIONS = 5
MAX_STEPS_PER_EPOCH = 5000
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0
WEIGHT_CLIP = 0.01
IMG_FOLDER_PATH = "data/Clouds"
RESULT_DIR = "results"

default_transformations = tmfs.Compose(
        [
            tmfs.RandomHorizontalFlip(),
            tmfs.ToTensor()
        ]
    )

