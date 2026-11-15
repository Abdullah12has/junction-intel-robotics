import unittest
import numpy as np
from variants import TargetTracker, transform
from runtime import DomainSampler


class FakeData:
    def __init__(self):
        self.real = list(range(15))
        self.episode_indices = [
            [np.arange(0, 5), np.arange(5, 15)],
            [np.arange(0, 7), np.arange(7, 15)],
        ]

    def __len__(self):
        return 20000


class Tests(unittest.TestCase):
    def test_sampler_domain_proportions_and_resume(self):
        d = FakeData()
        s = DomainSampler(d, 7, 0.2)
        all_items = list(s)
        self.assertTrue(0.19 < np.mean([i >= 15 for i, _ in all_items]) < 0.21)
        resumed = DomainSampler(d, 7, 0.2)
        resumed.load_state_dict({"epoch": 0, "start_index": 710})
        self.assertEqual(list(resumed), all_items[710:])

    def test_real_only_never_samples_sim(self):
        self.assertTrue(all(i < 15 for i, _ in DomainSampler(FakeData(), 0, 0)))

    def test_marker_causal_loss_does_not_reappear(self):
        im = np.zeros((100, 100, 3), np.uint8)
        im[35:66, 35:66] = [240, 10, 20]
        t = TargetTracker(im, [50, 50])
        xy, _ = t.update(im)
        self.assertGreater(xy[0], 0)
        xy, _ = t.update(np.zeros_like(im))
        self.assertEqual(xy, [-1.0, -1.0])
        xy, _ = t.update(im)
        self.assertEqual(xy, [-1.0, -1.0])

    def test_marker_displacement_suppressed(self):
        im = np.zeros((100, 100, 3), np.uint8)
        im[35:66, 35:66] = [240, 10, 20]
        t = TargetTracker(im, [50, 50])
        t.update(im)
        moved = np.roll(im, 17, axis=1)
        xy, _ = t.update(moved)
        self.assertEqual(xy, [-1.0, -1.0])

    def test_aug_identity_and_reproducibility(self):
        im = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(im, transform(im, 0, np.random.default_rng(4))))
        for recipe in [1, 2, 3]:
            a = transform(
                im, recipe, np.random.default_rng(12), overhead=True, xy=[300, 250]
            )
            b = transform(
                im, recipe, np.random.default_rng(12), overhead=True, xy=[300, 250]
            )
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(a.shape, im.shape)

    def test_invalid_marker_no_overlay(self):
        im = np.zeros((480, 640, 3), np.uint8)
        self.assertTrue(
            np.array_equal(im, transform(im, 0, np.random.default_rng(1), xy=[-1, -1]))
        )


if __name__ == "__main__":
    unittest.main()
