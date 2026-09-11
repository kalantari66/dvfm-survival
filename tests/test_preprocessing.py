import numpy as np

from utility.data import SurvivalData
from utility.splitting import preprocess_covariates


def test_numeric_zscore_uses_train_statistics_and_preserves_one_hot():
    def part(x):
        return SurvivalData(np.array(x, dtype=float), np.ones(len(x)),
                            np.ones(len(x)), ["age", "group_b"])
    train = part([[10, 0], [20, 1], [30, 0]])
    valid = part([[40, 1]])
    test = part([[50, 0]])
    train, valid, test = preprocess_covariates(
        train, valid, test, {"zscore_x": True, "numeric_features": ["age"]})
    np.testing.assert_allclose(train.X[:, 0].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(train.X[:, 0].std(), 1)
    np.testing.assert_allclose(valid.X[0, 0], (40 - 20) / np.std([10, 20, 30]))
    np.testing.assert_allclose(test.X[0, 0], (50 - 20) / np.std([10, 20, 30]))
    np.testing.assert_array_equal(train.X[:, 1], [0, 1, 0])
    assert valid.X[0, 1] == 1
    assert test.X[0, 1] == 0
