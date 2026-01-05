"""
Tests for markdown stripping in TTS response reader.
"""

from src.tts.response_reader import strip_markdown


def test_strip_headers():
    """Test stripping markdown headers."""
    text = "### 1. **Brazos Walking Stick**"
    result = strip_markdown(text)
    # Headers, numbered lists, and bold are all stripped
    assert result == "Brazos Walking Stick"

    text = "## Header 2\n### Header 3"
    result = strip_markdown(text)
    assert "##" not in result
    assert "###" not in result


def test_strip_bold():
    """Test stripping bold formatting."""
    text = "**SKU:** 415L16"
    result = strip_markdown(text)
    assert result == "SKU: 415L16"
    assert "**" not in result


def test_strip_italic():
    """Test stripping italic formatting."""
    text = "*Type:* Standard Single Walking Stick"
    result = strip_markdown(text)
    assert "*" not in result


def test_strip_lists():
    """Test stripping list markers."""
    text = "- Item 1\n- Item 2\n* Item 3"
    result = strip_markdown(text)
    assert result == "Item 1\nItem 2\nItem 3"

    text = "1. First\n2. Second"
    result = strip_markdown(text)
    assert "1." not in result
    assert "2." not in result


def test_strip_links():
    """Test stripping markdown links."""
    text = "[Click here](https://example.com)"
    result = strip_markdown(text)
    assert result == "Click here"
    assert "https://" not in result


def test_strip_code():
    """Test stripping code blocks and inline code."""
    text = "Use `code` here"
    result = strip_markdown(text)
    assert "`" not in result

    text = "```python\nprint('hello')\n```"
    result = strip_markdown(text)
    assert "```" not in result
    assert "python" not in result


def test_complex_markdown():
    """Test stripping complex markdown with multiple elements."""
    text = """### 1. **Brazos Walking Stick**
- **SKU:** 415L16
- **Type:** Standard Single Walking Stick
- **Material:** Wood
- **Height:** Adjustable up to 55 inches
- **Load Capacity:** 200 lbs
- **Features:** Includes a rubber bottom tip for better grip and stability.
- **Color:** Natural wood finish"""

    result = strip_markdown(text)

    # Should not contain markdown
    assert "###" not in result
    assert "**" not in result
    assert "- " not in result.split("\n")[0]  # First line shouldn't start with -

    # Should contain the actual content
    assert "Brazos Walking Stick" in result
    assert "SKU: 415L16" in result
    assert "200 lbs" in result


def test_empty_text():
    """Test handling empty text."""
    assert strip_markdown("") == ""
    assert strip_markdown(None) is None


def test_no_markdown():
    """Test text without markdown passes through."""
    text = "This is plain text with no markdown."
    result = strip_markdown(text)
    assert result == text


def test_multiple_newlines():
    """Test cleaning up multiple newlines."""
    text = "Line 1\n\n\n\nLine 2"
    result = strip_markdown(text)
    assert "\n\n\n" not in result
    assert "Line 1" in result
    assert "Line 2" in result
