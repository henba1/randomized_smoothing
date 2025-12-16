"""Signal handler for graceful shutdown and CSV file logging to Comet ML."""

import signal
from pathlib import Path
from typing import Any


_csv_writer = None
_tracker = None
_output_file = None


def _signal_handler(signum, frame):
    """Handle termination signals by logging CSV files before exit."""
    global _csv_writer, _tracker, _output_file
    print(f"\nReceived signal {signum}. Logging CSV files to Comet ML before exit...")
    try:
        if _csv_writer is not None and _tracker is not None:
            _csv_writer.log_to_comet(_tracker)
        if _output_file is not None and _tracker is not None:
            _tracker.log_asset(str(_output_file))
            print(f"Results txt file logged to Comet ML: {_output_file}")
    except Exception as e:
        print(f"Error logging files to Comet ML: {e}")
    raise SystemExit(1)


def setup_signal_handler(csv_writer: Any, tracker: Any, output_file: Path) -> None:
    """Set up signal handlers for graceful shutdown and CSV file logging.

    Registers signal handlers for SIGTERM and SIGINT to ensure CSV files
    are logged to Comet ML before the process exits (e.g., on job cancellation).

    Args:
        csv_writer: CSVResultWriter instance
        tracker: CometTracker instance
        output_file: Path to the output txt file
    """
    global _csv_writer, _tracker, _output_file
    _csv_writer = csv_writer
    _tracker = tracker
    _output_file = output_file

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
