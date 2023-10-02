import math
import os
import gc

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
from torchvision.utils import save_image


def num_to_groups(num, divisor): # samples, batch_size
    groups = num // divisor # 5000 // 1024 = 4
    remainder = num % divisor # 5000 % 1024 = 976
    arr = [divisor] * groups # [1024] * 4 = [1024, 1024, 1024, 1024]
    if remainder > 0:
        arr.append(remainder) # [1024, 1024, 1024, 1024, 976]
    return arr


class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        pipeline,
        inference_T,
        channels=3,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
        seed=42,
        save_path="./images",
        epoch=1000,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.pipeline = pipeline
        self.inference_T = inference_T
        self.stats_dir = stats_dir
        self.print_fn = print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False
        self.seed = seed
        self.save_path = save_path
        self.epoch = epoch

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        ckpt = np.load(self.stats_dir+'train_all.npz')
        self.m2, self.s2 = ckpt["mu"], ckpt["sigma"]
        self.print_fn("Dataset stats loaded from disk.")
        ckpt.close()
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()

        num_sampling_rounds = int(np.ceil(self.n_samples / self.batch_size)) # 50000 / 1024 = 49
        batches = num_to_groups(self.n_samples, self.batch_size) # [1024] * 48 + [976]
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        stacked_fake_features = []
        # for batch in tqdm(batches):
        for j in range(num_sampling_rounds):

            fake_samples = self.pipeline(
                batch_size = batches[j],
                num_inference_steps=self.inference_T,
                generator=torch.Generator(device=self.device).manual_seed(self.seed+j)
            ).images
            gc.collect()
            torch.cuda.empty_cache()

            # save the images
            test_dir = os.path.join(self.save_path, f'samples/uncond/{self.epoch:04d}')
            os.makedirs(test_dir, exist_ok=True)
            save_image(fake_samples, f'{test_dir}/{str(j)}.png')
            gc.collect()
            torch.cuda.empty_cache()

            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)

        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)
        fid = calculate_frechet_distance(m1, s1, self.m2, self.s2)

        return fid
