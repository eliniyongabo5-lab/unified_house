"""Scientific integrity checks for the new runner (unittest, CPU only)."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import revised_data as rd
import revised_experiments as legacy
from benchmark_neural import BASE_SPEC, BenchmarkNet, load, train
from run_benchmarks import metrics, prepare, validate_output


class BenchmarkIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data, _, cls.transforms, cls.arrays = prepare(42)

    def test_split_groups_and_original_partition(self):
        frozen = self.data["slope"].row_id[self.data["slope"].split["test"]]
        for seed in (42, 137, 271):
            data, _, _, _ = prepare(seed)
            np.testing.assert_array_equal(data["slope"].row_id[data["slope"].split["test"]], frozen)
            for t in data.values():
                partitions = list(t.split.values())
                self.assertEqual(sum(map(len, partitions)), len(t.target))
                self.assertEqual(len(np.unique(np.concatenate(partitions))), len(t.target))
            groups = [set(data["rock"].groups[i]) for i in data["rock"].split.values()]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_transforms_ignore_test_values_and_labels(self):
        data = copy.deepcopy(self.data)
        for t in data.values():
            test = t.split["test"]
            t.raw[test] = 999999.
            t.target[test] = 1 - t.target[test] if t.name == "slope" else 999999.
        transforms = rd.fit_transforms(data)
        self.assertEqual(transforms, self.transforms)
        for task in data:
            tr = data[task].split["train"]
            np.testing.assert_array_equal(data[task].x[tr], self.data[task].x[tr])

    def test_training_never_reads_test_tensors(self):
        arrays = copy.deepcopy(self.arrays)
        for task, t in self.data.items():
            for tensor in arrays[task].values():
                tensor[t.split["test"]] = float("nan")
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a", Path(temporary) / "b"
            _, first = train(self.data, self.arrays, BASE_SPEC, 9, a, max_epochs=6, patience=2)
            _, second = train(self.data, arrays, BASE_SPEC, 9, b, max_epochs=6, patience=2)
            self.assertEqual(first["best_validation_loss"], second["best_validation_loss"])
            for key, value in load(self.data, a).state_dict().items():
                torch.testing.assert_close(value, load(self.data, b).state_dict()[key], rtol=0, atol=0)

    def test_original_architecture_forward_parity(self):
        dims = {k: t.x.shape[1] for k, t in self.data.items()}
        for shared, physics, gate in legacy.CONFIGS.values():
            legacy.seed_everything(7)
            original = legacy.Model(dims, shared, physics, gate)
            legacy.seed_everything(7)
            updated = BenchmarkNet(dims, shared, physics, gate)
            for task, t in self.data.items():
                a = self.arrays[task]
                args = (task, a["x"][:10], a["prior"][:10], a["valid"][:10], t.y_mean, t.y_std)
                torch.testing.assert_close(original(*args)[0], updated(*args)[0], rtol=0, atol=0)

    def test_missing_prior_is_bypassed(self):
        model = BenchmarkNet({"rock": 2}, physics=True, gate=True)
        x = torch.zeros(4, 2)
        valid = torch.zeros(4)
        first = model("rock", x, torch.zeros(4), valid, 10., 2.)[0]
        second = model("rock", x, torch.full((4,), 1e8), valid, 10., 2.)[0]
        torch.testing.assert_close(first, second)

    def test_failure_recall_uses_failure_class(self):
        result = metrics("slope", np.array([0, 0, 1, 1]), np.array([.1, .2, .3, .9]))
        self.assertEqual(result["failure_recall"], 1.)
        self.assertEqual(result["balanced_accuracy"], .75)

    def test_output_guards_prevent_historical_or_unidentified_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            protected = base / "corrected_v1"
            protected.mkdir()
            for output in (protected, protected / "nested"):
                with self.assertRaises(ValueError):
                    validate_output(output, protected)
            stale = base / "stale"
            stale.mkdir()
            (stale / "model.pt").write_text("unidentified")
            with self.assertRaises(ValueError):
                validate_output(stale, protected)
            validate_output(base / "new", protected)


if __name__ == "__main__":
    unittest.main()
