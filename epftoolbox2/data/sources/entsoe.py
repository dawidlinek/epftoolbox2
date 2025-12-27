import warnings
from typing import List, Dict
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
import logging
from .base import DataSource

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class NoMatchingDataError(Exception):
    """Raised when ENTSOE API returns no matching data"""

    pass


AREAS = {
    "DE_50HZ": "10YDE-VE-------2",
    "AL": "10YAL-KESH-----5",
    "DE_AMPRION": "10YDE-RWENET---I",
    "AT": "10YAT-APG------L",
    "BY": "10Y1001A1001A51S",
    "BE": "10YBE----------2",
    "BA": "10YBA-JPCC-----D",
    "BG": "10YCA-BULGARIA-R",
    "CZ_DE_SK": "10YDOM-CZ-DE-SKK",
    "HR": "10YHR-HEP------M",
    "CWE": "10YDOM-REGION-1V",
    "CY": "10YCY-1001A0003J",
    "CZ": "10YCZ-CEPS-----N",
    "DE_AT_LU": "10Y1001A1001A63L",
    "DE_LU": "10Y1001A1001A82H",
    "DK": "10Y1001A1001A65H",
    "DK_1": "10YDK-1--------W",
    "DK_1_NO_1": "46Y000000000007M",
    "DK_2": "10YDK-2--------M",
    "DK_CA": "10Y1001A1001A796",
    "EE": "10Y1001A1001A39I",
    "FI": "10YFI-1--------U",
    "MK": "10YMK-MEPSO----8",
    "FR": "10YFR-RTE------C",
    "DE": "10Y1001A1001A83F",
    "GR": "10YGR-HTSO-----Y",
    "HU": "10YHU-MAVIR----U",
    "IS": "IS",
    "IE_SEM": "10Y1001A1001A59C",
    "IE": "10YIE-1001A00010",
    "IT": "10YIT-GRTN-----B",
    "IT_SACO_AC": "10Y1001A1001A885",
    "IT_CALA": "10Y1001C--00096J",
    "IT_SACO_DC": "10Y1001A1001A893",
    "IT_BRNN": "10Y1001A1001A699",
    "IT_CNOR": "10Y1001A1001A70O",
    "IT_CSUD": "10Y1001A1001A71M",
    "IT_FOGN": "10Y1001A1001A72K",
    "IT_GR": "10Y1001A1001A66F",
    "IT_MACRO_NORTH": "10Y1001A1001A84D",
    "IT_MACRO_SOUTH": "10Y1001A1001A85B",
    "IT_MALTA": "10Y1001A1001A877",
    "IT_NORD": "10Y1001A1001A73I",
    "IT_NORD_AT": "10Y1001A1001A80L",
    "IT_NORD_CH": "10Y1001A1001A68B",
    "IT_NORD_FR": "10Y1001A1001A81J",
    "IT_NORD_SI": "10Y1001A1001A67D",
    "IT_PRGP": "10Y1001A1001A76C",
    "IT_ROSN": "10Y1001A1001A77A",
    "IT_SARD": "10Y1001A1001A74G",
    "IT_SICI": "10Y1001A1001A75E",
    "IT_SUD": "10Y1001A1001A788",
    "RU_KGD": "10Y1001A1001A50U",
    "LV": "10YLV-1001A00074",
    "LT": "10YLT-1001A0008Q",
    "LU": "10YLU-CEGEDEL-NQ",
    "LU_BZN": "10Y1001A1001A82H",
    "MT": "10Y1001A1001A93C",
    "ME": "10YCS-CG-TSO---S",
    "GB": "10YGB----------A",
    "GE": "10Y1001A1001B012",
    "GB_IFA": "10Y1001C--00098F",
    "GB_IFA2": "17Y0000009369493",
    "GB_ELECLINK": "11Y0-0000-0265-K",
    "UK": "10Y1001A1001A92E",
    "NL": "10YNL----------L",
    "NO_1": "10YNO-1--------2",
    "NO_1A": "10Y1001A1001A64J",
    "NO_2": "10YNO-2--------T",
    "NO_2_NSL": "50Y0JVU59B4JWQCU",
    "NO_2A": "10Y1001C--001219",
    "NO_3": "10YNO-3--------J",
    "NO_4": "10YNO-4--------9",
    "NO_5": "10Y1001A1001A48H",
    "NO": "10YNO-0--------C",
    "PL_CZ": "10YDOM-1001A082L",
    "PL": "10YPL-AREA-----S",
    "PT": "10YPT-REN------W",
    "MD": "10Y1001A1001A990",
    "RO": "10YRO-TEL------P",
    "RU": "10Y1001A1001A49F",
    "SE_1": "10Y1001A1001A44P",
    "SE_2": "10Y1001A1001A45N",
    "SE_3": "10Y1001A1001A46L",
    "SE_4": "10Y1001A1001A47J",
    "RS": "10YCS-SERBIATSOV",
    "SK": "10YSK-SEPS-----K",
    "SI": "10YSI-ELES-----O",
    "GB_NIR": "10Y1001A1001A016",
    "ES": "10YES-REE------0",
    "SE": "10YSE-1--------K",
    "CH": "10YCH-SWISSGRIDZ",
    "DE_TENNET": "10YDE-EON------1",
    "DE_TRANSNET": "10YDE-ENBW-----N",
    "TR": "10YTR-TEIAS----W",
    "UA": "10Y1001C--00003F",
    "UA_DOBTPP": "10Y1001A1001A869",
    "UA_BEI": "10YUA-WEPS-----0",
    "UA_IPS": "10Y1001C--000182",
    "XK": "10Y1001C--00100H",
    "DE_AMP_LU": "10Y1001C--00002H",
}

PSRTYPE_MAPPINGS = {
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B06": "Fossil Oil",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and pondage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other",
}


def lookup_area(code: str) -> tuple[str, str]:
    code_upper = code.upper()
    if code_upper in AREAS:
        return (code_upper, AREAS[code_upper])
    raise ValueError(f"Invalid country code: {code}")


class EntsoeSource(DataSource):
    """
    ENTSOE data source for electricity market data

    Supports fetching:
    - Load: actual load, day-ahead forecast, week-ahead forecast (min/max)
    - Generation: actual by type, wind/solar forecasts
    - Price: day-ahead market prices

    Example:
        >>> source = EntsoeSource('PL', api_key='your-api-key', type=['load', 'generation', 'price'])
        >>> df = source.fetch(start=pd.Timestamp('2024-01-01', tz='Europe/Warsaw'),
        ...                   end=pd.Timestamp('2024-01-07', tz='Europe/Warsaw'))
        >>> # df contains all columns: load_actual, load_forecast, generation_*, price, etc.
    """

    API_URL = "https://web-api.tp.entsoe.eu/api"

    def __init__(self, country_code: str, api_key: str, type: List[str]):
        self.area_name, self.area_code = lookup_area(country_code)
        self.api_key = api_key
        self.types = type
        self.session = requests.Session()

        self.console = Console()
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.addHandler(RichHandler(console=self.console, rich_tracebacks=True))
            self.logger.setLevel(logging.INFO)

        self._validate_config()

    def __del__(self):
        """Cleanup session on object destruction"""
        if hasattr(self, "session") and self.session:
            self.session.close()

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        if not self.types:
            raise ValueError("At least one data type must be specified")

        valid_types = {"load", "generation", "price"}
        for t in self.types:
            if t not in valid_types:
                raise ValueError(f"Invalid type '{t}'. Must be one of: {valid_types}")

        return True

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        start_time = time.time()

        chunks = self._generate_chunks(start, end, months=3)

        self.logger.info(f"ENTSOE [{self.area_name}]: Start downloading {', '.join(self.types)} data")

        all_results = {dtype: [] for dtype in self.types}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]ENTSOE [{self.area_name}]: Downloading {', '.join(self.types)}...",
                total=len(chunks),
            )

            for chunk_start, chunk_end in chunks:
                date_range = f"{chunk_start.date()} to {chunk_end.date()}"
                progress.update(task, description=f"[cyan]ENTSOE [{self.area_name}]: {date_range}")

                chunk_data = self._fetch_chunk(chunk_start, chunk_end)

                for dtype in self.types:
                    if dtype in chunk_data:
                        all_results[dtype].append(chunk_data[dtype])
                progress.advance(task)

        dataframes = []
        for dtype in self.types:
            if all_results[dtype]:
                df = pd.concat(all_results[dtype]).sort_index()
                df = df[~df.index.duplicated(keep="first")]
                dataframes.append(df)

        elapsed = time.time() - start_time
        self._log_success(elapsed)

        if dataframes:
            return pd.concat(dataframes, axis=1)
        return pd.DataFrame()

    def _generate_chunks(self, start: pd.Timestamp, end: pd.Timestamp, months: int = 3) -> list:
        if months <= 0:
            raise ValueError(f"months parameter must be positive, got {months}")

        chunks = []
        current = start

        while current < end:
            next_chunk = current + pd.DateOffset(months=months)
            chunk_end = min(next_chunk, end)
            chunks.append((current, chunk_end))
            current = next_chunk

        return chunks

    def _fetch_chunk(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        result = {}

        if "load" in self.types:
            result["load"] = self._fetch_load(start, end)

        if "generation" in self.types:
            result["generation"] = self._fetch_generation(start, end)

        if "price" in self.types:
            result["price"] = self._fetch_price(start, end)

        return result

    def _fetch_load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dfs = []

        xml = self._api_request(
            {
                "documentType": "A65",
                "processType": "A16",
                "outBiddingZone_Domain": self.area_code,
                "out_Domain": self.area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_loads(xml, "A16"))

        xml = self._api_request(
            {
                "documentType": "A65",
                "processType": "A01",
                "outBiddingZone_Domain": self.area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_loads(xml, "A01"))

        xml = self._api_request(
            {
                "documentType": "A65",
                "processType": "A31",
                "outBiddingZone_Domain": self.area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_loads(xml, "A31"))

        result = pd.concat(dfs, axis=1)
        return result.truncate(before=start, after=end)

    def _fetch_generation(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dfs = []

        xml = self._api_request(
            {
                "documentType": "A75",
                "processType": "A16",
                "in_Domain": self.area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_generation(xml).add_prefix("generation_"))

        xml = self._api_request(
            {
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": self.area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_generation(xml).add_prefix("generation_"))

        xml = self._api_request(
            {
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": self.area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_generation(xml).add_prefix("generation_").add_suffix("_forecast"))

        result = pd.concat(dfs, axis=1)
        return result.truncate(before=start, after=end)

    def _fetch_price(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        xml = self._api_request(
            {
                "documentType": "A44",
                "in_Domain": self.area_code,
                "out_Domain": self.area_code,
            },
            start,
            end,
        )

        price_dict = self._parse_prices(xml)

        series = pd.Series()
        if price_dict["60min"] is not None and len(price_dict["60min"]) > 0:
            series = price_dict["60min"]
        elif price_dict["15min"] is not None and len(price_dict["15min"]) > 0:
            series = price_dict["15min"]

        series = series.truncate(before=start, after=end)

        return pd.DataFrame({"price": series})

    def _api_request(self, params: Dict, start: pd.Timestamp, end: pd.Timestamp) -> str:
        start_str = start.strftime("%Y%m%d%H00")
        end_str = end.strftime("%Y%m%d%H00")

        params.update(
            {
                "securityToken": self.api_key,
                "periodStart": start_str,
                "periodEnd": end_str,
            }
        )

        response = self.session.get(url=self.API_URL, params=params, timeout=30)

        if "No matching data found" in response.text:
            return None

        response.raise_for_status()

        return response.text

    def _parse_loads(self, xml_text: str, process_type: str) -> pd.DataFrame:
        if xml_text is None:
            return pd.DataFrame()
        if process_type in ["A01", "A16"]:
            series = []
            for soup in self._extract_timeseries(xml_text):
                s = self._parse_timeseries_generic(soup, merge=True)
                series.append(s)
            series = pd.concat(series).sort_index() if series else pd.Series()
            col_name = "load_forecast" if process_type == "A01" else "load_actual"
            return pd.DataFrame({col_name: series})
        else:
            series_min_list = []
            series_max_list = []
            for soup in self._extract_timeseries(xml_text):
                t = self._parse_timeseries_generic(soup, merge=True)
                bsn_type = soup.find("businesstype")
                if bsn_type and bsn_type.text == "A60":
                    series_min_list.append(t)
                elif bsn_type and bsn_type.text == "A61":
                    series_max_list.append(t)

            series_min = pd.concat(series_min_list) if series_min_list else pd.Series(dtype=float)
            series_max = pd.concat(series_max_list) if series_max_list else pd.Series(dtype=float)

            return pd.DataFrame(
                {
                    "load_forecast_daily_min": series_min,
                    "load_forecast_daily_max": series_max,
                }
            )

    def _parse_generation(self, xml_text: str) -> pd.DataFrame:
        if xml_text is None:
            return pd.DataFrame()
        all_series = {}
        for soup in self._extract_timeseries(xml_text):
            series = self._parse_timeseries_generic(soup, merge=True)

            _psrtype = soup.find("psrtype")
            if _psrtype is not None:
                psrtype = _psrtype.text
                psrtype_name = PSRTYPE_MAPPINGS.get(psrtype, psrtype)
            else:
                psrtype_name = "unknown"

            psrtype_snake = psrtype_name.lower().replace(" ", "_").replace("/", "_")
            series.name = psrtype_snake

            if series.name in all_series:
                all_series[series.name] = pd.concat([all_series[series.name], series]).sort_index()
            else:
                all_series[series.name] = series

        for name in all_series:
            ts = all_series[name]
            all_series[name] = ts[~ts.index.duplicated(keep="first")]

        df = pd.DataFrame.from_dict(all_series)
        df.sort_index(inplace=True)
        return df

    def _parse_prices(self, xml_text: str) -> Dict[str, pd.Series]:
        series = {"15min": [], "30min": [], "60min": []}
        if xml_text is None:
            return series

        for soup in self._extract_timeseries(xml_text):
            soup_series = self._parse_timeseries_generic(soup, label="price.amount")
            for key in series.keys():
                if soup_series[key] is not None:
                    series[key].append(soup_series[key])

        for freq, freq_series in series.items():
            try:
                series[freq] = pd.concat(freq_series).sort_index()
            except ValueError:
                series[freq] = pd.Series()
        return series

    def _parse_timeseries_generic(self, soup, label="quantity", merge=False):
        series = {"15min": [], "30min": [], "60min": [], "1D": [], "7D": []}

        for period in soup.find_all("period"):
            data = {}

            start_elem = period.find("start")
            resolution_elem = period.find("resolution")

            if start_elem is None or resolution_elem is None:
                continue

            start = pd.Timestamp(start_elem.text)
            delta_text = self._resolution_to_timedelta(resolution_elem.text)
            delta = pd.Timedelta(delta_text)

            for point in period.find_all("point"):
                value_elem = point.find(label)
                position_elem = point.find("position")

                if value_elem is None or position_elem is None:
                    continue

                value = value_elem.text.replace(",", "")
                position = int(position_elem.text)
                data[start + (position - 1) * delta] = value

            time_series = pd.Series(data).sort_index()

            if delta_text not in series:
                series[delta_text] = []
            series[delta_text].append(time_series)

        for freq, freq_series in series.items():
            if len(freq_series) > 0:
                series[freq] = pd.concat(freq_series).sort_index().astype(float)
            else:
                series[freq] = None

        if merge:
            return pd.concat([s for s in series.values() if s is not None])
        else:
            return series

    def _extract_timeseries(self, xml_text: str):
        if not xml_text:
            return
        soup = BeautifulSoup(xml_text, "html.parser")
        for timeseries in soup.find_all("timeseries"):
            yield timeseries

    def _resolution_to_timedelta(self, res_text: str) -> str:
        resolutions = {
            "PT60M": "60min",
            "PT15M": "15min",
            "PT30M": "30min",
            "P1D": "1D",
            "P7D": "7D",
        }
        delta = resolutions.get(res_text)
        if delta is None:
            raise NotImplementedError(f"Unknown resolution format: {res_text}")
        return delta

    def _log_success(self, elapsed: float):
        self.logger.info(f"ENTSOE [{self.area_name}]: Download completed successfully in {elapsed:.2f} sec")
