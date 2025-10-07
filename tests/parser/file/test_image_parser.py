from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from application.parser.file.image_parser import ImageParser


def test_image_init_parser():
    parser = ImageParser()
    assert isinstance(parser._init_parser(), dict)
    parser.init_parser()
    assert parser.parser_config_set


@patch("application.parser.file.image_parser.settings")
def test_image_parser_remote_true(mock_settings):
    mock_settings.PARSE_IMAGE_REMOTE = True
    parser = ImageParser()

    mock_response = MagicMock()
    mock_response.json.return_value = {"markdown": "# From Image"}

    with patch("application.parser.file.image_parser.requests.post", return_value=mock_response) as mock_post:
        with patch("builtins.open", mock_open()):
            result = parser.parse_file(Path("img.png"))

    assert result == "# From Image"


@patch("application.parser.file.image_parser.settings")
def test_image_parser_remote_false(mock_settings):
    mock_settings.PARSE_IMAGE_REMOTE = False
    parser = ImageParser()

    with patch("application.parser.file.image_parser.requests.post") as mock_post:
        result = parser.parse_file(Path("img.png"))

    assert result == ""

