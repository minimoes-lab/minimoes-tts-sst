# Contributing to Streaming Avatar API

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/your-username/streaming-avatar-api.git
cd streaming-avatar-api
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Project Structure

```
├── api.py                    # Main FastAPI application
├── streaming/               # Streaming components
│   ├── qwen_tts_worker.py  # TTS generation
│   ├── blendshape_worker.py # Facial animation
│   └── kyutai_coordinator.py # Streaming coordination
├── utils/                   # Utilities
│   └── model/              # Blendshape model
├── requirements.txt        # Dependencies
└── Dockerfile             # Docker configuration
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and small
- Use meaningful variable names

## Testing

Run tests before submitting:
```bash
pytest
```

## Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Test thoroughly
4. Commit with clear messages:
```bash
git commit -m "Add: feature description"
```

5. Push to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create a Pull Request

## Pull Request Guidelines

- Describe what your PR does
- Reference any related issues
- Include tests if applicable
- Update documentation if needed
- Ensure all tests pass
- Keep PRs focused on a single feature/fix

## Reporting Issues

When reporting issues, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and logs

## Feature Requests

We welcome feature requests! Please:
- Check if it's already requested
- Describe the use case
- Explain why it would be useful
- Suggest implementation if possible

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing! 🎉
