"""Signal handler for graceful shutdown and CSV file logging to Comet ML."""

import signal
from pathlib import Path
from typing import Any, Callable, Optional


_csv_writer = None
_tracker = None
_output_file = None
_summary_params = None


def _signal_handler(signum, frame):
    """Handle termination signals by creating summary and logging CSV files before exit."""
    global _csv_writer, _tracker, _output_file, _summary_params
    print(f"\nReceived signal {signum}. Creating summary and logging CSV files before exit...")
    try:
        # Create summary dataframe if we have the necessary parameters
        if _summary_params is not None and _csv_writer is not None:
            print("Creating summary dataframe with current progress...")
            _csv_writer.create_summary(**_summary_params)
        
        if _csv_writer is not None and _tracker is not None:
            _csv_writer.log_to_comet(_tracker)
        if _output_file is not None and _tracker is not None:
            _tracker.log_asset(str(_output_file))
            print(f"Results txt file logged to Comet ML: {_output_file}")
    except Exception as e:
        print(f"Error during graceful shutdown: {e}")
    raise SystemExit(1)


def setup_signal_handler(
    csv_writer: Any,
    tracker: Any,
    output_file: Path,
    update_summary_params: Optional[Callable[[], dict]] = None,
) -> None:
    """Set up signal handlers for graceful shutdown and CSV file logging.

    Registers signal handlers for SIGTERM and SIGINT to ensure summary dataframe
    is created and CSV files are logged to Comet ML before the process exits
    (e.g., on job cancellation).

    Args:
        csv_writer: CSVResultWriter instance
        tracker: CometTracker instance
        output_file: Path to the output txt file
        update_summary_params: Optional callable that returns a dict with current
            summary parameters (total_num, correct, n_misclassified, n_abstain,
            sigma, alpha, N0, N, model_name). This will be called when a signal
            is received to get the latest state.
    """
    global _csv_writer, _tracker, _output_file, _summary_params
    _csv_writer = csv_writer
    _tracker = tracker
    _output_file = output_file
    _update_summary_params = update_summary_params

    def _handler_with_params(signum, frame):
        """Wrapper that updates summary params before calling the handler."""
        global _summary_params
        if _update_summary_params is not None:
            _summary_params = _update_summary_params()
        _signal_handler(signum, frame)

    signal.signal(signal.SIGTERM, _handler_with_params)
    signal.signal(signal.SIGINT, _handler_with_params)
