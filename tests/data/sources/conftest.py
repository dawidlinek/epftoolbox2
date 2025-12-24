import os
import pytest


@pytest.fixture
def entsoe_api_key():
    """
    Get ENTSOE API key from environment variable.

    To run integration tests, set: ENTSOE_API_KEY=your-key-here
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        pytest.skip("ENTSOE_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def sample_xml_load():
    """Sample XML response for load data"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument>
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2024-01-01T00:00Z</start>
                <end>2024-01-01T01:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <quantity>1000</quantity>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>"""


@pytest.fixture
def sample_xml_price():
    """Sample XML response for price data"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument>
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2024-01-01T00:00Z</start>
                <end>2024-01-01T01:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>50.5</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>"""


@pytest.fixture
def sample_xml_generation():
    """Sample XML response for generation data"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument>
    <TimeSeries>
        <MktPSRType>
            <psrType>B16</psrType>
        </MktPSRType>
        <Period>
            <timeInterval>
                <start>2024-01-01T00:00Z</start>
                <end>2024-01-01T01:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <quantity>500</quantity>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>"""
