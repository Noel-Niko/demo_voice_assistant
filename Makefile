# Root level Makefile
 
.PHONY: install dev test test-ci test-vad lint format run stop models demo check-deps install-deps clean help
 
install:
	uv sync
 
dev:
	uv sync --dev
 
check-deps:
	uv run python scripts/check_dependencies.py
 
install-deps:
	uv run python scripts/check_dependencies.py --install-all
 
test: check-deps
	uv run pytest tests/ -v
 
test-ci:
	uv run pytest tests/ -v
 
test-vad: check-deps
	uv run python tests/run_vad_tests.py
 
lint:
	uv run ruff check src/ tests/
	uv run mypy src/
 
format:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/
 
run:
	@echo "Stopping any existing server processes..."
	@echo "Checking for webrtc_server processes..."
	@ps aux | grep webrtc_server | grep -v grep || echo "No webrtc_server processes found"
	@echo "Checking for processes using port 9999..."
	@lsof -ti:9999 | xargs -I {} echo "Found process {} using port 9999" || echo "No processes using port 9999"
	@pkill -f webrtc_server || true
	@lsof -ti:9999 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo "Checking again for processes using port 9999 after cleanup..."
	@lsof -ti:9999 | xargs -I {} echo "Still found process {} using port 9999" || echo "Port 9999 is now free"
	@echo "Starting WebRTC voice server..."
	@set -a; [ -f .env ] && . .env; set +a; uv run uvicorn src.gateway.webrtc_server:app --reload --host 0.0.0.0 --port 9999
 
stop:
	@echo "Stopping server processes..."
	@pkill -f webrtc_server || true
	@lsof -ti:9999 | xargs kill -9 2>/dev/null || true
	@sleep 2
	@echo "All server processes stopped."
 
models:
	@echo "Available ASR Models:"
	@echo ""
	@echo "Google Cloud Speech-to-Text v2 Models (set MODEL_NAME env var):"
	@echo "    latest_long         - Best quality for general speech"
	@echo "    latest_short        - Faster, lower latency"
	@echo "    telephony           - Optimized for phone audio"
	@echo "    medical_conversation - Optimized for medical dictation"
	@echo "    chirp               - High quality (availability may vary)"
	@echo "    chirp_2             - Best quality (availability may vary)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  MODEL_NAME=latest_long make run"
 
demo:
	uv run python demo/webrtc_client.py
 
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
 
help:
	@echo "Available commands:"
	@echo "  make install        # Install dependencies"
	@echo "  make dev            # Install dev dependencies"
	@echo "  make run            # Run server (reloads .env each time)"
	@echo "  make stop           # Stop server"
	@echo "  make test           # Run tests"
	@echo "  make lint           # Run linting"
	@echo "  make format         # Format code"
	@echo "  make models         # List available models"
	@echo "  make demo           # Run demo client"
	@echo "  make clean          # Clean cache files"
