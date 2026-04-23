from __future__ import annotations

from perplexity_at_home.utils import get_current_datetime_string


def test_get_current_datetime_string_includes_expected_prefix() -> None:
    value = get_current_datetime_string()

    assert value.startswith("The current date and time is: ")
    assert "EDT" in value or "EST" in value
