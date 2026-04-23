from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_datetime_string() -> str:
    """Return the current date and time as a formatted string."""
    return (
        f"The current date and time is: "
        f"{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    