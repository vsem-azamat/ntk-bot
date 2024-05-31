import pytest
from unittest.mock import patch, MagicMock
from apps.parse_functions import get_ntk_quantity


@pytest.mark.asyncio
@patch('aiohttp.ClientSession.get')
async def test_get_ntk_quantity(mock_get):
    # Mocking the response of the web page
    mock_response = MagicMock()
    mock_response.status = 200
    async def mock_text():
        return """
        <div class="panel-body text-center lead">42</div>
        """
    mock_response.text = mock_text
    mock_get.return_value.__aenter__.return_value = mock_response

    # Run the function
    result = await get_ntk_quantity()

    # Assert the result
    assert result == 42
