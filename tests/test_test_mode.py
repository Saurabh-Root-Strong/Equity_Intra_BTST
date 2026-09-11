"""Test mode pre-open. Offline, no network, no DuckDB.

Measured live at 09:05-09:08 IST: every quote came back lp == prev_close == high == low with
volume 0, so universe_mtf_scan's `h == l` guard dropped 211 of 211 names and test mode showed an
empty board -- in the one window it is most wanted, before the open. The fix keeps those rows
OFF-HOURS ONLY, and makes the mode part of every cache key so a pre-open board can never be
served as the live one at 09:15.
"""
import inspect
import io

from eqbtst import live


def _scan_src():
    return inspect.getsource(live.universe_mtf_scan)


def test_the_scan_takes_an_explicit_offhours_flag_defaulting_to_live():
    sig = inspect.signature(live.universe_mtf_scan)
    assert "allow_offhours" in sig.parameters
    assert sig.parameters["allow_offhours"].default is False, "live must be the default"


def test_a_zero_range_is_still_dropped_in_a_live_session():
    """During the session a zero range means a halted or frozen name, and it breaks bar_clr
    downstream. Relaxing that for everyone to fix test mode would reintroduce it live."""
    src = _scan_src()
    i = src.index("h is None or l is None or h == l")
    blk = src[i:i + 200]
    assert "if not allow_offhours:" in blk and "continue" in blk


def test_offhours_keeps_the_row_and_uses_the_quote_as_the_range():
    src = _scan_src()
    i = src.index("h is None or l is None or h == l")
    assert "h = l = c" in src[i:i + 200]


def test_a_missing_price_is_dropped_in_every_mode():
    """The relaxation is about the RANGE. A quote with no price or no previous close is not a
    row in any mode."""
    assert "if None in (c, pc):" in _scan_src()


def test_the_module_memo_key_separates_offhours_from_live():
    """Same 5-minute bucket, different mode, must be a different entry -- otherwise a live
    request at 09:15 could be answered with the board a test-mode request cached at 09:14."""
    src = _scan_src()
    assert "_bucket5()" in src and "allow_offhours" in src.split("hit = _UNISCAN_CACHE")[0]


def test_the_streamlit_cache_is_keyed_on_offhours_too():
    """_uni_scan was cached by nonce alone, and the nonce does NOT change when the market opens.
    Without the flag in its signature a pre-open board -- every price at yesterday's close, every
    day% zero -- would be served as the LIVE board. Far worse than an empty one."""
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "def _uni_scan(nonce: int, offhours: bool = False):" in src
    assert "live.universe_mtf_scan(allow_offhours=offhours)" in src
    assert "offhours=bool(test_mode and not live.market_open())" in src


def test_offhours_is_only_ever_true_with_the_market_shut():
    """The flag must fall to False the instant the session opens, which is what forces the fresh
    live scan. Tying it to test_mode alone would keep relaxing the guard all day."""
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "test_mode and not live.market_open()" in src
