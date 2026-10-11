import pytest


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


@pytest.fixture
def sample_xml_load_none():
    """Fixture that returns None to simulate no data available"""
    return None


@pytest.fixture
def sample_xml_with_daily_resolution():
    """Sample XML response with daily (1D) resolution"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument>
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2024-01-01T00:00Z</start>
                <end>2024-01-02T00:00Z</end>
            </timeInterval>
            <resolution>P1D</resolution>
            <Point>
                <position>1</position>
                <quantity>24000</quantity>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>"""


@pytest.fixture
def sample_xml_with_weekly_resolution():
    """Sample XML response with weekly (7D) resolution"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument>
    <TimeSeries>
        <Period>
            <timeInterval>
                <start>2024-01-01T00:00Z</start>
                <end>2024-01-08T00:00Z</end>
            </timeInterval>
            <resolution>P7D</resolution>
            <Point>
                <position>1</position>
                <quantity>168000</quantity>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>"""


@pytest.fixture
def sample_weather_response():
    """Sample Previous Runs API response: 3 days of hourly valid times (UTC) from 2024-01-01.

    Values encode 100 * bucket + position (temperature) and bucket + position / 100 (rain), so a wrong offset or
    direction in the d+i mapping changes the numbers instead of passing silently.
    """
    n = 72
    return {
        "hourly": {
            "time": [f"2024-01-{1 + j // 24:02d}T{j % 24:02d}:00" for j in range(n)],
            "temperature_2m_previous_day1": [100.0 + j for j in range(n)],
            "temperature_2m_previous_day2": [200.0 + j for j in range(n)],
            "rain_previous_day1": [1.0 + j / 100 for j in range(n)],
            "rain_previous_day2": [2.0 + j / 100 for j in range(n)],
        }
    }


@pytest.fixture
def sample_weather_response_incomplete():
    """Sample JSON response with incomplete/missing data"""
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m_previous_day1": [10.0],
            "temperature_2m_previous_day2": [11.0],
        }
    }
