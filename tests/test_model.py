import unittest

import torch

from strideplan.model import WorkoutRanker


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = WorkoutRanker(3)

    def test_forward_backward_and_history_dependence(self):
        items = torch.tensor([[1, 2], [3, 2]])
        features = torch.zeros(2, 2, 5)
        ranking, effort = self.model(items, features, torch.tensor([2, 2]))
        self.assertEqual(ranking.shape, (2, 3))
        self.assertEqual(effort.shape, (2, 3))
        self.assertFalse(torch.allclose(ranking[0], ranking[1]))
        loss = torch.nn.functional.cross_entropy(ranking, torch.tensor([1, 2]))
        loss += torch.nn.functional.binary_cross_entropy_with_logits(effort, torch.zeros_like(effort))
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_padding_and_empty_history(self):
        features = torch.zeros(1, 1, 5)
        short = self.model(torch.tensor([[1]]), features, torch.tensor([1]))
        padded = self.model(torch.tensor([[1, 0, 0]]), torch.cat((features, torch.ones(1, 2, 5)), dim=1), torch.tensor([1]))
        for first, second in zip(short, padded):
            torch.testing.assert_close(first, second)
        ranking, effort = self.model(torch.zeros(1, 1, dtype=torch.long), features, torch.tensor([0]))
        torch.testing.assert_close(ranking, self.model.ranking_bias[None, :])
        self.assertTrue(torch.isfinite(effort).all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[4]]), torch.zeros(1, 1, 5), torch.tensor([1]))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1]]), torch.zeros(1, 1, 5), torch.tensor([2]))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1]]), torch.full((1, 1, 5), float("nan")), torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
