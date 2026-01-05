"""
Comprehensive tests for performance metrics tracking and persistence.
Tests all the metrics mentioned in the user's requirements.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from src.llm.performance_metrics import (
    PerformanceMetricsPersistence,
    ProcessingTimeMetrics,
    SessionMetrics,
    SystemPerformanceMetrics,
    TranscriptionMetrics,
    end_session_tracking,
    export_metrics_report,
    get_metrics_summary,
    get_performance_metrics,
    start_session_tracking,
    update_transcription_metrics,
)


class TestTranscriptionMetrics:
    """Test transcription-specific metrics tracking."""

    def test_transcription_metrics_initialization(self):
        """Test TranscriptionMetrics initializes with correct defaults."""
        metrics = TranscriptionMetrics()

        assert metrics.total_transcriptions == 0
        assert metrics.final_transcriptions == 0
        assert metrics.partial_transcriptions == 0
        assert metrics.total_confidence_score == 0.0
        assert metrics.average_confidence == 0.0
        assert metrics.total_words == 0
        assert metrics.total_audio_duration == 0.0
        assert metrics.average_word_time == 0.0

    def test_update_transcription_final(self):
        """Test updating metrics with final transcription."""
        metrics = TranscriptionMetrics()

        # Update with final transcription
        metrics.update_transcription(
            is_final=True, confidence=0.92, word_count=17, audio_duration=6.7
        )

        assert metrics.total_transcriptions == 1
        assert metrics.final_transcriptions == 1
        assert metrics.partial_transcriptions == 0
        assert metrics.total_confidence_score == 0.92
        assert metrics.average_confidence == 0.92
        assert metrics.total_words == 17
        assert metrics.total_audio_duration == 6.7
        assert metrics.average_word_time == 6.7 / 17

    def test_update_transcription_partial(self):
        """Test updating metrics with partial transcription."""
        metrics = TranscriptionMetrics()

        # Update with partial transcription
        metrics.update_transcription(
            is_final=False, confidence=0.5, word_count=5, audio_duration=2.0
        )

        assert metrics.total_transcriptions == 1
        assert metrics.final_transcriptions == 0
        assert metrics.partial_transcriptions == 1
        assert metrics.total_confidence_score == 0.5
        assert metrics.average_confidence == 0.5
        assert metrics.total_words == 5
        assert metrics.total_audio_duration == 2.0
        assert metrics.average_word_time == 2.0 / 5

    def test_update_transcription_multiple(self):
        """Test updating metrics with multiple transcriptions."""
        metrics = TranscriptionMetrics()

        # Add multiple transcriptions
        metrics.update_transcription(True, 0.92, 17, 6.7)
        metrics.update_transcription(False, 0.5, 3, 1.0)
        metrics.update_transcription(True, 0.88, 12, 5.5)

        assert metrics.total_transcriptions == 3
        assert metrics.final_transcriptions == 2
        assert metrics.partial_transcriptions == 1
        assert metrics.total_confidence_score == 2.3  # 0.92 + 0.5 + 0.88
        assert metrics.average_confidence == 2.3 / 3
        assert metrics.total_words == 32  # 17 + 3 + 12
        assert metrics.total_audio_duration == 13.2  # 6.7 + 1.0 + 5.5
        assert metrics.average_word_time == 13.2 / 32

    def test_transcription_metrics_to_dict(self):
        """Test TranscriptionMetrics serialization."""
        metrics = TranscriptionMetrics()
        metrics.update_transcription(True, 0.92, 17, 6.7)

        result = metrics.to_dict()

        assert result["total_transcriptions"] == 1
        assert result["final_transcriptions"] == 1
        assert result["partial_transcriptions"] == 0
        assert result["average_confidence"] == 0.92
        assert result["total_words"] == 17
        assert result["total_audio_duration"] == 6.7
        assert result["average_word_time"] == round(6.7 / 17, 4)


class TestProcessingTimeMetrics:
    """Test processing time metrics tracking."""

    def test_processing_metrics_initialization(self):
        """Test ProcessingTimeMetrics initializes with correct defaults."""
        metrics = ProcessingTimeMetrics()

        assert metrics.total_processing_time == 0.0
        assert metrics.audio_to_transcript_time == 0.0
        assert metrics.transcript_to_llm_time == 0.0
        assert metrics.llm_processing_time == 0.0
        assert metrics.tool_execution_time == 0.0
        assert metrics.response_generation_time == 0.0
        assert metrics.session_count == 0
        assert metrics.average_processing_time == 0.0

    def test_update_processing_times(self):
        """Test updating processing times."""
        metrics = ProcessingTimeMetrics()

        # Update with session times
        times = {
            "total_end_to_end": 6.67545,
            "audio_to_transcript": 2.5,
            "transcript_to_llm": 0.5,
            "llm_processing": 2.0,
            "tool_execution": 1.0,
            "response_generation": 0.67545,
        }

        metrics.update_processing_times(times)

        assert metrics.total_processing_time == 6.67545
        assert metrics.audio_to_transcript_time == 2.5
        assert metrics.transcript_to_llm_time == 0.5
        assert metrics.llm_processing_time == 2.0
        assert metrics.tool_execution_time == 1.0
        assert metrics.response_generation_time == 0.67545
        assert metrics.session_count == 1
        assert metrics.average_processing_time == 6.67545

    def test_update_processing_times_multiple_sessions(self):
        """Test updating processing times with multiple sessions."""
        metrics = ProcessingTimeMetrics()

        # Add multiple sessions
        metrics.update_processing_times({"total_end_to_end": 6.7})
        metrics.update_processing_times({"total_end_to_end": 8.2})
        metrics.update_processing_times({"total_end_to_end": 5.5})

        assert metrics.total_processing_time == 20.4  # 6.7 + 8.2 + 5.5
        assert metrics.session_count == 3
        assert metrics.average_processing_time == 20.4 / 3

    def test_processing_metrics_to_dict(self):
        """Test ProcessingTimeMetrics serialization."""
        metrics = ProcessingTimeMetrics()
        metrics.update_processing_times({"total_end_to_end": 6.67545})

        result = metrics.to_dict()

        assert result["total_processing_time"] == round(6.67545, 4)
        assert result["session_count"] == 1
        assert result["average_processing_time"] == round(6.67545, 4)


class TestSessionMetrics:
    """Test session-level metrics tracking."""

    def test_session_metrics_initialization(self):
        """Test SessionMetrics initializes correctly."""
        session = SessionMetrics(session_id="test_session", start_time=datetime.now())

        assert session.session_id == "test_session"
        assert session.start_time is not None
        assert session.end_time is None
        assert session.duration == 0.0
        assert session.transcription_count == 0
        assert session.final_transcription_count == 0
        assert session.total_words == 0
        assert session.average_confidence == 0.0
        assert session.processing_time == 0.0

    def test_finalize_session(self):
        """Test session finalization."""
        start_time = datetime.now() - timedelta(seconds=10)
        session = SessionMetrics(session_id="test_session", start_time=start_time)

        # Add a small delay to ensure duration is calculated

        time.sleep(0.01)

        session.finalize_session()

        assert session.end_time is not None
        assert session.end_time > session.start_time
        assert session.duration >= 0.01  # Should be at least the sleep time

    def test_session_metrics_to_dict(self):
        """Test SessionMetrics serialization."""
        start_time = datetime.now()
        session = SessionMetrics(session_id="test_session", start_time=start_time)

        # Add a small delay to ensure duration is calculated

        time.sleep(0.01)

        session.finalize_session()

        result = session.to_dict()

        assert result["session_id"] == "test_session"
        assert result["start_time"] == start_time.isoformat()
        assert result["end_time"] is not None
        assert result["duration"] >= 0.01  # Should be at least the sleep time


class TestSystemPerformanceMetrics:
    """Test comprehensive system performance metrics."""

    def test_system_metrics_initialization(self):
        """Test SystemPerformanceMetrics initializes correctly."""
        metrics = SystemPerformanceMetrics()

        assert isinstance(metrics.transcription_metrics, TranscriptionMetrics)
        assert isinstance(metrics.processing_metrics, ProcessingTimeMetrics)
        assert metrics.active_sessions == {}
        assert metrics.completed_sessions == []
        assert isinstance(metrics.system_start_time, datetime)
        assert isinstance(metrics.last_updated, datetime)

    def test_start_and_end_session(self):
        """Test session lifecycle tracking."""
        metrics = SystemPerformanceMetrics()

        # Start session
        session = metrics.start_session("test_session")

        assert isinstance(session, SessionMetrics)
        assert session.session_id == "test_session"
        assert "test_session" in metrics.active_sessions
        assert len(metrics.completed_sessions) == 0

        # End session
        metrics.end_session("test_session", 6.7)

        assert "test_session" not in metrics.active_sessions
        assert len(metrics.completed_sessions) == 1
        assert metrics.completed_sessions[0].session_id == "test_session"
        assert metrics.processing_metrics.session_count == 1
        assert metrics.processing_metrics.average_processing_time == 6.7

    def test_update_transcription_in_session(self):
        """Test updating transcription metrics for active session."""
        metrics = SystemPerformanceMetrics()

        # Start session and update transcription
        metrics.start_session("test_session")
        metrics.update_transcription("test_session", True, 0.92, 17, 6.7)

        # Check global metrics
        assert metrics.transcription_metrics.total_transcriptions == 1
        assert metrics.transcription_metrics.final_transcriptions == 1
        assert metrics.transcription_metrics.average_confidence == 0.92

        # Check session metrics
        session = metrics.active_sessions["test_session"]
        assert session.transcription_count == 1
        assert session.final_transcription_count == 1
        assert session.total_words == 17
        assert session.average_confidence == 0.92

    def test_get_current_summary(self):
        """Test getting current performance summary."""
        metrics = SystemPerformanceMetrics()

        # Add some data
        metrics.start_session("test_session")
        metrics.update_transcription("test_session", True, 0.92, 17, 6.7)
        metrics.end_session("test_session", 6.67545)

        summary = metrics.get_current_summary()

        # Check system metrics
        assert "system_metrics" in summary
        assert summary["system_metrics"]["active_sessions"] == 0
        assert summary["system_metrics"]["completed_sessions"] == 1

        # Check transcription metrics
        assert "transcription_metrics" in summary
        tm = summary["transcription_metrics"]
        assert tm["total_transcriptions"] == 1
        assert tm["final_transcriptions"] == 1
        assert tm["average_confidence"] == 0.92
        assert tm["total_words"] == 17
        assert tm["average_word_time"] == round(6.7 / 17, 4)

        # Check processing metrics
        assert "processing_metrics" in summary
        pm = summary["processing_metrics"]
        assert pm["session_count"] == 1
        assert pm["average_processing_time"] == round(6.67545, 4)

        # Check session summary
        assert "session_summary" in summary
        ss = summary["session_summary"]
        assert ss["total_sessions"] == 1
        assert ss["average_session_confidence"] == 0.92


class TestPerformanceMetricsPersistence:
    """Test metrics persistence functionality."""

    def test_save_and_load_metrics(self):
        """Test saving and loading metrics to/from file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # Create metrics with data
            metrics = SystemPerformanceMetrics()
            metrics.start_session("test_session")
            metrics.update_transcription("test_session", True, 0.92, 17, 6.7)
            metrics.end_session("test_session", 6.67545)

            # Save metrics
            persistence = PerformanceMetricsPersistence(temp_file)
            success = persistence.save_metrics(metrics)
            assert success is True

            # Load metrics
            loaded_metrics = persistence.load_metrics()

            # Verify loaded data
            assert loaded_metrics.transcription_metrics.total_transcriptions == 1
            assert loaded_metrics.transcription_metrics.final_transcriptions == 1
            assert loaded_metrics.transcription_metrics.average_confidence == 0.92
            assert loaded_metrics.processing_metrics.session_count == 1
            assert loaded_metrics.processing_metrics.average_processing_time == round(6.67545, 4)

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_metrics_when_file_not_exists(self):
        """Test loading metrics when file doesn't exist returns new metrics."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        # Delete the file so it doesn't exist
        os.unlink(temp_file)

        try:
            persistence = PerformanceMetricsPersistence(temp_file)
            metrics = persistence.load_metrics()

            # Should return new empty metrics
            assert isinstance(metrics, SystemPerformanceMetrics)
            assert metrics.transcription_metrics.total_transcriptions == 0
            assert metrics.processing_metrics.session_count == 0

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_report(self):
        """Test exporting metrics report."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # Create metrics with data
            metrics = SystemPerformanceMetrics()
            metrics.start_session("test_session")
            metrics.update_transcription("test_session", True, 0.92, 17, 6.7)
            metrics.end_session("test_session", 6.67545)

            # Export report
            persistence = PerformanceMetricsPersistence()
            success = persistence.export_metrics_report(metrics, temp_file)
            assert success is True

            # Verify report file exists and has correct structure
            assert os.path.exists(temp_file)
            with open(temp_file, "r") as f:
                report = json.load(f)

            assert "report_generated" in report
            assert "performance_summary" in report
            assert "detailed_metrics" in report

            # Check summary data
            summary = report["performance_summary"]
            assert summary["transcription_metrics"]["total_transcriptions"] == 1
            assert summary["transcription_metrics"]["average_confidence"] == 0.92
            assert summary["processing_metrics"]["average_processing_time"] == round(6.67545, 4)

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_performance_metrics_singleton(self):
        """Test get_performance_metrics returns singleton instance."""
        # Reset global state
        import src.llm.performance_metrics as pm

        pm._global_metrics = None
        pm._metrics_persistence = None

        # Get metrics twice
        metrics1 = get_performance_metrics()
        metrics2 = get_performance_metrics()

        # Should be the same instance
        assert metrics1 is metrics2
        assert isinstance(metrics1, SystemPerformanceMetrics)

    def test_global_session_tracking_functions(self, monkeypatch):
        """Test global session tracking functions."""
        # Reset global state completely
        # Isolate persistence to a temp file to avoid cross-test contamination
        import tempfile

        import src.llm.performance_metrics as pm

        orig_cls = pm.PerformanceMetricsPersistence
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_path = tmp.name
        tmp.close()
        monkeypatch.setattr(
            pm,
            "PerformanceMetricsPersistence",
            lambda metrics_file="performance_metrics.json": orig_cls(tmp_path),
        )

        pm._global_metrics = None
        pm._metrics_persistence = None

        # Start session
        session = start_session_tracking("test_session")
        assert session.session_id == "test_session"

        # Update transcription
        update_transcription_metrics("test_session", True, 0.92, 17, 6.7)

        # End session
        end_session_tracking("test_session", 6.67545)

        # Check summary
        summary = get_metrics_summary()
        assert summary["transcription_metrics"]["total_transcriptions"] == 1
        assert summary["transcription_metrics"]["average_confidence"] == 0.92
        assert summary["processing_metrics"]["average_processing_time"] == round(6.67545, 4)

    def test_export_metrics_report_function(self):
        """Test global export metrics report function."""
        # Reset global state
        import src.llm.performance_metrics as pm

        pm._global_metrics = None
        pm._metrics_persistence = None

        # Add some data
        start_session_tracking("test_session")
        update_transcription_metrics("test_session", True, 0.92, 17, 6.7)
        end_session_tracking("test_session", 6.67545)

        # Export report
        success = export_metrics_report("test_report.json")
        assert success is True

        # Clean up
        if os.path.exists("test_report.json"):
            os.unlink("test_report.json")


class TestRequiredMetricsValidation:
    """Test that all required metrics from user's specification are tracked."""

    def test_all_required_metrics_are_tracked(self, monkeypatch):
        """Test that all metrics mentioned in user's requirements are tracked."""
        # Reset global state completely
        # Isolate persistence to a temp file to avoid cross-test contamination
        import tempfile

        import src.llm.performance_metrics as pm

        orig_cls = pm.PerformanceMetricsPersistence
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_path = tmp.name
        tmp.close()
        monkeypatch.setattr(
            pm,
            "PerformanceMetricsPersistence",
            lambda metrics_file="performance_metrics.json": orig_cls(tmp_path),
        )

        pm._global_metrics = None
        pm._metrics_persistence = None

        # Simulate a complete session
        start_session_tracking("test_session")

        # Update transcription metrics
        update_transcription_metrics("test_session", True, 0.92, 17, 6.7)

        # End session with processing time
        end_session_tracking("test_session", 6.67545)

        # Get summary and validate all required metrics
        summary = get_metrics_summary()

        # Validate transcription metrics
        tm = summary["transcription_metrics"]
        assert "total_transcriptions" in tm  # Total Transcriptions: 1
        assert "final_transcriptions" in tm  # Final Transcriptions: 1
        assert "average_confidence" in tm  # Average Confidence: 0.92
        assert "total_words" in tm  # Total Words: 17
        assert "average_word_time" in tm  # Avg Word Time: 0.39268s

        # Validate processing metrics
        pm = summary["processing_metrics"]
        assert "average_processing_time" in pm  # Avg Processing Time: 6.67545s

        # Validate session summary
        ss = summary["session_summary"]
        assert "total_sessions" in ss
        assert "average_session_duration" in ss  # Session Duration: 6.7s

        # Verify the specific values match the user's example
        assert tm["total_transcriptions"] >= 1  # At least 1 transcription
        assert tm["final_transcriptions"] >= 1  # At least 1 final transcription
        assert tm["average_confidence"] == 0.92
        assert pm["average_processing_time"] == round(6.67545, 4)
        assert tm["total_words"] == 17
        assert tm["average_word_time"] == round(6.7 / 17, 4)  # 0.39268s
        # Session duration will be tracked in the session metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
