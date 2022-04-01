import os
import torch
import config as conf
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from model import Critic, Generator
from dataset import create_data_patches
import matplotlib.pyplot as plt


def main():
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    loader = create_data_patches(conf.IMG_FOLDER_PATH,
                                 conf.IMG_HEIGHT,
                                 conf.IMG_WIDTH,
                                 batch_size=conf.BATCH_SIZE,
                                 shuffle=True)

    print("DataLoader created...")

    gen = Generator(conf.RAND_SIZE,
                    conf.IN_CHANNELS,
                    num_transpose=conf.NUM_TRANSPOSE).to(DEVICE)
    print("Generator model loaded on device...")
    crit = Critic(conf.IN_CHANNELS,
                  h=conf.IMG_HEIGHT,
                  w=conf.IMG_WIDTH,
                  num_conv=conf.NUM_CONV).to(DEVICE)
    print("Critic Model loaded on device...")

    gen_optim = optim.RMSprop(gen.parameters(),
                              lr=conf.LEARNING_RATE,
                              weight_decay=conf.WEIGHT_DECAY)
    crit_optim = optim.RMSprop(crit.parameters(),
                               lr=conf.LEARNING_RATE,
                               weight_decay=conf.WEIGHT_DECAY)

    print("Optimizer loaded")
    result_dir = conf.RESULT_DIR

    print("Training initiated...")
    for epoch in range(conf.EPOCHS):
        print(f"Epoch : {epoch + 1}")
        crit.train()
        gen.train()
        for real_patch in loader:
            real_patch = real_patch.to(DEVICE)
            curr_batch_size = real_patch.shape[0]
            for _ in range(conf.NUM_ITERATIONS):
                real_op = crit(real_patch)
                crit_noise = torch.randn(curr_batch_size, conf.RAND_SIZE).to(DEVICE)
                fake_patch = gen(crit_noise)
                fake_op = crit(fake_patch)
                crit_loss = (torch.mean(fake_op) - torch.mean(real_op))
                crit.zero_grad()
                crit_loss.backward(retain_graph=True)
                crit_optim.step()
                # for param in crit.parameters():
                #     param.data.clamp_(-conf.WEIGHT_CLIP, conf.WEIGHT_CLIP)

            gen_op = crit(fake_patch)
            gen_loss = torch.mean(gen_op)
            gen.zero_grad()
            gen_loss.backward()
            gen_optim.step()

        gen.eval()
        crit.eval()
        print(f"Generator Loss : {gen_loss:.4f}  ||  Critic Loss : {crit_loss:.4f}")
        fake_img_grid = make_grid(fake_patch[:30], nrow=6, normalize=True).cpu().permute(1, 2, 0)
        fake_filename = f"{result_dir}/fake/{epoch + 1}.png"
        plt.figure(figsize=(15, 8))
        plt.imshow(fake_img_grid)
        plt.savefig(fake_filename)
        real_img_grid = make_grid(real_patch[:30], nrow=6, normalize=True).cpu().permute(1, 2, 0)
        real_filename = f"{result_dir}/real/{epoch + 1}.png"
        plt.figure(figsize=(15, 8))
        plt.imshow(real_img_grid)
        plt.savefig(real_filename)

    print("Training Completed...")


if __name__ == "__main__":
    main()
