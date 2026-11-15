"""LeRobot dataset adapter with explicit domains, per-episode sampling and causal X."""

import numpy as np
import torch
from torch.utils.data import Dataset
from prepare import OVER, WRIST, TASK
from variants import transform


class VariantDataset(Dataset):
    def __init__(
        self, real, markers, recipe, marked, sim=None, sim_markers=None, training=True
    ):
        self.real = real
        self.sim = sim
        self.markers = [markers, sim_markers]
        self.recipe = recipe
        self.marked = marked
        self.training = training
        self.meta = real.meta
        self.episodes = real.episodes
        self.absolute_to_relative_idx = real.absolute_to_relative_idx
        self.num_frames = len(real) + (len(sim) if sim else 0)
        self.num_episodes = real.num_episodes + (sim.num_episodes if sim else 0)
        self.episode_indices = []
        for ds in [real, sim]:
            groups = []
            if ds is not None:
                episode_ids = np.array(ds.hf_dataset["episode_index"])
                groups = [
                    np.flatnonzero(episode_ids == e) for e in np.unique(episode_ids)
                ]
            self.episode_indices.append(groups)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, key):
        idx, augseed = key if isinstance(key, tuple) else (key, 0)
        domain = int(idx >= len(self.real))
        ds = self.sim if domain else self.real
        idx = idx - len(self.real) if domain else idx
        item = ds[idx]
        absolute = int(item["index"])
        point = self.markers[domain][absolute] if self.marked else None
        rng = np.random.default_rng(augseed)
        for camera in [OVER, WRIST]:
            value = item[camera]
            stack = value.ndim == 4
            if stack and len(value) != 1:
                raise ValueError("This campaign supports one observation only")
            if stack:
                value = value[0]
            rgb = value.permute(1, 2, 0).numpy()
            rgb = rgb if rgb.dtype == np.uint8 else np.rint(rgb * 255).astype(np.uint8)
            rgb = transform(
                rgb,
                self.recipe if self.training else 0,
                rng,
                overhead=camera == OVER,
                xy=point if camera == OVER else None,
            )
            output = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
            item[camera] = output.unsqueeze(0) if stack else output
        item["task"] = (
            "Pick up the marked cube and put it on the blue plate."
            if self.marked
            else TASK
        )
        return item


class DomainSampler:
    """Deterministic (domain, episode, frame) draws with resumable epoch offset.

    Each output carries an independent augmentation seed, so worker scheduling does
    not change transformations and resume does not require restoring worker RNGs.
    """

    def __init__(self, dataset, seed=0, sim_fraction=0.0, **kwargs):
        self.dataset = dataset
        self.seed = seed
        self.sim_fraction = sim_fraction
        self.epoch = 0
        self.start = 0
        if sim_fraction and not dataset.episode_indices[1]:
            raise ValueError("Missing simulation training episodes")

    def __len__(self):
        return len(self.dataset)

    def state_dict(self):
        return {"epoch": self.epoch, "start_index": self.start}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.start = state["start_index"]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        epoch, start = self.epoch, self.start
        self.epoch += 1
        self.start = 0
        return self._iterate(epoch, start)

    def _iterate(self, epoch, start):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        for position in range(len(self)):
            domain = int(rng.random() < self.sim_fraction)
            episodes = self.dataset.episode_indices[domain]
            frames = episodes[int(rng.integers(len(episodes)))]
            idx = int(frames[int(rng.integers(len(frames)))])
            if domain:
                idx += len(self.dataset.real)
            augseed = int(rng.integers(0, 2**63 - 1))
            if position >= start:
                yield idx, augseed
