import time
import numpy as np
import wandb

wandb.init(project='image-test')


IMG_WIDTH = 5
IMG_HEIGHT = 5

IMG_COUNT = 2
# Step count
N = 2


def gen_image(w=IMG_WIDTH, h=IMG_HEIGHT):
    return np.concatenate(
        (np.random.rand(h//2, w),
         np.zeros((h//2, w))),
        axis=0)

all_test = {
    "test_image_file_single": wandb.Image("test_summary_image_7_1.png"),
    "test_image_file_array": [wandb.Image("test_summary_image_7_1.png")],
    "test_image_data_single": wandb.Image(gen_image()),
    "test_image_data_array": [wandb.Image(gen_image()) for _ in range(IMG_COUNT)],
}

for i in range(0, N):
    wandb.log({"i": i}, step=i)

time.sleep(100)
