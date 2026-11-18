"""Suite-wide isolation.

The Fyers call counter persists to data/fyers_calls.json, and a cap recorded there by the
REAL board makes live._fyers_get refuse to send for 10 minutes. Tests that fake the broker's
reply would then be routed to the NSE fallback instead -- passing or failing depending on
whether the live dashboard happened to hit the cap recently. Every test gets a private,
empty counter."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_fyers_counter(tmp_path, monkeypatch):
    from eqbtst import config, live
    monkeypatch.setattr(live, "_CALLS_FILE", tmp_path / "fyers_calls_isolated.json")
    monkeypatch.setattr(config, "EQBTST_FYERS_BUDGET", 0)
    live._CALLS.clear()
    live._PENDING.update({"calls": 0, "by_kind": {}})
    live._SRC["nse_at"] = None
    yield
    live._CALLS.clear()
    live._PENDING.update({"calls": 0, "by_kind": {}})
