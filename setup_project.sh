#!/bin/bash
# Grainger Voice Assistant - Project Setup Script
# Run this to create the project structure

set -e

echo "🚀 Setting up Grainger Voice Assistant project..."

# Create project root
PROJECT_NAME="Voice_Assistant"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Create directory structure
echo "📁 Creating directory structure..."

mkdir -p src/{asr,tts,agent,gateway,handoff,utils}
mkdir -p data/{mro_vocabulary,training_audio,synthetic_utterances,pending_orders}
mkdir -p models/{whisper-mro,xtts-mro}
mkdir -p notebooks
mkdir -p tests
mkdir -p demo
mkdir -p docs

# Create __init__.py files
touch src/__init__.py
touch src/asr/__init__.py
touch src/tts/__init__.py
touch src/agent/__init__.py
touch src/gateway/__init__.py
touch src/handoff/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create placeholder files
echo "📝 Creating placeholder files..."

# pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "grainger-voice-assistant"
version = "0.1.0"
description = "Voice-enabled product search assistant for Grainger"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "black",
    "ruff",
    "mypy",
]

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
EOF

# Makefile
cat > Makefile << 'EOF'
.PHONY: install dev test lint format run demo clean

install:
	uv sync

dev:
	uv sync --dev

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

run:
	uvicorn src.gateway.voice_gateway:app --reload --host 0.0.0.0 --port 8000

demo:
	python demo/gradio_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
EOF

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Type checking
.mypy_cache/

# Models (large files)
models/*.pth
models/*.bin
models/*.safetensors

# Data (large files)
data/training_audio/*.wav
data/synthetic_utterances/*.wav

# Environment
.env
.env.local

# Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
EOF

# .env.example
cat > .env.example << 'EOF'
# Anthropic API (for LLM)
ANTHROPIC_API_KEY=your_api_key_here

# Twilio (for phone handoff)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
WEBHOOK_BASE_URL=https://your-domain.com

# Model paths
ASR_MODEL_PATH=models/whisper-mro
TTS_MODEL_PATH=models/xtts-mro/model.pth
TTS_CONFIG_PATH=models/xtts-mro/config.json
SPEAKER_WAV_PATH=data/reference_voice.wav

# MCP Server
MCP_SERVER_COMMAND=python,mcp_server.py
EOF

# README.md
cat > README.md << 'EOF'
# Grainger Voice Product Assistant

Voice-enabled product search and ordering system for industrial MRO supplies.

## Features

- 🎤 Real-time speech recognition with fine-tuned Whisper
- 🔍 Semantic product search via MCP tools
- 🔊 Natural TTS with correct MRO pronunciation
- 📞 Phone handoff to customer service

## Quick Start

```bash
# Install dependencies
make install

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
make run

# Launch demo UI
make demo
```

## Project Structure

```
Voice_Assistant/
├── src/
│   ├── asr/          # Speech recognition
│   ├── tts/          # Text-to-speech
│   ├── agent/        # LangGraph voice agent
│   ├── gateway/      # WebSocket server
│   └── handoff/      # Phone integration
├── data/             # Training data, vocabulary
├── models/           # Fine-tuned model checkpoints
├── demo/             # Gradio demo UI
└── tests/            # Unit tests
```

## Architecture

[Architecture diagram here]

## Documentation

See [docs/](docs/) for detailed documentation.

## License

MIT
EOF

# Create empty test files
cat > tests/test_asr.py << 'EOF'
"""Tests for ASR module."""
import pytest

def test_whisper_base():
    """Test base Whisper transcription."""
    # TODO: Implement
    pass

def test_vad():
    """Test voice activity detection."""
    # TODO: Implement
    pass
EOF

cat > tests/test_agent.py << 'EOF'
"""Tests for agent module."""
import pytest

def test_intent_classification():
    """Test intent classification."""
    # TODO: Implement
    pass

def test_order_management():
    """Test order management."""
    # TODO: Implement
    pass
EOF

# Create MRO vocabulary starter file
cat > data/mro_vocabulary/fasteners.txt << 'EOF'
# Hex Bolts
quarter inch hex bolt
quarter inch by one inch hex bolt
quarter inch by two inch hex bolt
three eighths inch hex bolt
three eighths by one inch hex bolt
half inch hex bolt
five eighths inch hex bolt

# Cap Screws
quarter inch cap screw
socket head cap screw
hex head cap screw
button head cap screw

# Thread Specifications
1/4-20 hex head
1/4-28 fine thread
5/16-18 hex bolt
5/16-24 fine thread
3/8-16 hex head bolt
3/8-24 fine thread cap screw
1/2-13 hex bolt
1/2-20 fine thread

# Materials
stainless steel
stainless 304
stainless 316
zinc plated
galvanized
black oxide
grade 5
grade 8

# Metric
M6 by 1.0
M8 by 1.25
M10 by 1.5
M12 by 1.75
EOF

echo "✅ Project structure created!"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. python -m venv venv && source venv/bin/activate"
echo "3. pip install uv && uv sync"
echo "4. cp .env.example .env && edit .env"
echo ""
echo "📚 See docs/grainger-voice-assistant-learning-plan.md for the full tutorial guide"
