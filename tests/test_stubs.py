import pytest

from octmirror.baselines import hilbert_analytic_signal


@pytest.mark.parametrize("fn", [hilbert_analytic_signal])
def test_unreleased_methods_are_stubs(fn):
    with pytest.raises(NotImplementedError):
        fn()
