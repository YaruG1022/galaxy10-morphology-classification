"""Regression checks for training-only exploratory statistics and gallery selection."""
import sys
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eda import collect_training, describe, image_statistics, select_examples


class ExploratoryDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls.temp.name)/"fixture.h5"
        cls.labels = np.repeat(np.arange(10), 5)
        cls.train = np.flatnonzero(np.arange(50)%5 < 4)
        pixels = np.empty((50, 256, 256, 3), dtype=np.uint8)
        for i in range(50):
            pixels[i].fill(255 if i%5 == 4 else 10+int(cls.labels[i]))
        with h5py.File(cls.raw, "w") as f:
            f.create_dataset("images", data=pixels, chunks=(5, 16, 16, 3))
            f["redshift"] = np.ones(50)
            f["pxscale"] = np.full(50, .262)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_held_out_bright_images_do_not_change_training_statistics(self):
        examples = select_examples(self.train, self.labels)
        train, stats, means, counts, pictures, _ = collect_training(self.raw, self.train, self.labels, examples)
        self.assertTrue(np.array_equal(train, self.train))
        self.assertTrue(np.all(counts == 4))
        self.assertEqual(set(pictures), set(self.train))
        for c in range(10):
            expected = (10+c)/255
            self.assertTrue(np.allclose(stats[self.labels[train] == c, :4], expected))
            self.assertTrue(np.allclose(stats[self.labels[train] == c, 4], 0))
            self.assertTrue(np.allclose(means[c], expected))

    def test_gallery_is_seeded_and_training_only(self):
        first = select_examples(self.train, self.labels)
        self.assertEqual(first, select_examples(self.train, self.labels))
        for c, indices in first.items():
            self.assertEqual(len(set(indices)), 4)
            self.assertTrue(set(indices).issubset(set(self.train)))
            self.assertTrue(np.all(self.labels[indices] == int(c)))

    def test_repeated_rows_and_held_out_gallery_are_rejected(self):
        examples = select_examples(self.train, self.labels)
        with self.assertRaises(ValueError):
            collect_training(self.raw, np.r_[self.train, self.train[0]], self.labels, examples)
        examples["0"] = [4]
        with self.assertRaises(ValueError):
            collect_training(self.raw, self.train, self.labels, examples)

    def test_pixel_definition_and_nonfinite_summary(self):
        image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        values = image_statistics(image)
        self.assertTrue(np.allclose(values, [.5, .5, .5, .5, .5]))
        self.assertEqual(describe(np.array([np.nan, 1., 3., np.inf]))["median"], 2.)
        self.assertIsNone(describe(np.array([np.nan]))["median"])


if __name__ == "__main__":
    unittest.main()
