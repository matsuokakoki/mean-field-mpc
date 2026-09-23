from __future__ import annotations

import pandas as pd
import pytest

from mfcontrol.experiments.controllers import _select_alphas


def test_alpha_selection_rejects_nonvalidation_rows() -> None:
    with pytest.raises(AssertionError, match="validation rows only"):
        _select_alphas(
            {"control": {}},
            pd.DataFrame({"split": ["test"]}),
            scale=1.0,
            bin_seconds=300.0,
            service_rate=1.0,
            n=10,
            delay=1,
        )
