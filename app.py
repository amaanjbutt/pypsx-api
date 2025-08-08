"""
FastAPI service exposing Pakistan Stock Exchange analytics.

This API wraps the `pyPSX` library and a handful of lightweight web
scrapers to provide JSON endpoints for symbol lookup, historical data,
intraday data, announcements and simple comparative analytics. It is
designed for demonstration purposes and does not scrape or relay raw
PSX data beyond what is needed to compute derived metrics. Where
possible, scraped pages are obtained via `requests` and parsed using
BeautifulSoup rather than Selenium to reduce overhead. However, if
network access fails or pages change structure, endpoints will return
empty results or raise an HTTP 400 error.

To run the API locally:

    pip install fastapi uvicorn pandas requests beautifulsoup4
    uvicorn app:app --reload

Then visit `http://localhost:8000/docs` for interactive API docs.

Note: The `pypsx` package must be installed in the same environment.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

from psx.core import PSXTicker

app = FastAPI(title="PyPSX API", version="0.1.0")

# Minimal mapping of PSX symbols to company names. Extend this
# dictionary as needed or replace it with a file/database lookup.
symbol_name_map: Dict[str, str] = {
    'HBL': 'Habib Bank Limited',
    'MCB': 'MCB Bank Limited',
    'OGDC': 'Oil and Gas Development Company Limited',
    'FFC': 'Fauji Fertilizer Company Limited',
    'PSO': 'Pakistan State Oil Company Limited',
    'DGKC': 'D.G. Khan Cement Company Limited',
    'OGDC': 'Oil & Gas Development Co. Ltd.',
    'UBL': 'United Bank Limited',
    'ENGROH': 'Engro Holding Limited'
}

# Sorted list of available symbols. Additional symbols can be added
# here or loaded from a file (e.g. psx_symbols.txt) at startup.
symbols: List[str] = sorted(symbol_name_map.keys())

@app.get("/symbols", summary="List available PSX symbols")
async def get_symbols() -> List[Dict[str, str]]:
    """Return a list of trading symbols and their company names."""
    return [{"code": code, "name": symbol_name_map.get(code, code)} for code in symbols]


@app.get("/historical", summary="Historical OHLCV data")
async def get_historical(
    symbol: str = Query(..., description="PSX ticker symbol"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
) -> List[Dict[str, Any]]:
    """Return daily OHLCV records for a symbol between two dates.

    Date strings are parsed using `datetime.fromisoformat` and must be
    valid ISO date representations. If the PSX API is unreachable or
    returns no data, a 400 error will be raised.
    """
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    ticker = PSXTicker(symbol)
    try:
        df = ticker.get_historical_data(start_date=start, end_date=end)
        if df.empty:
            raise ValueError("No data returned")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch historical data: {e}")
    # Reset index to include the date as a field
    df = df.reset_index().rename(columns={df.index.name or df.columns[0]: 'Date'})
    # Standardise column names to title case
    df.columns = [str(c).title() for c in df.columns]
    return df.to_dict(orient="records")


@app.get("/intraday", summary="Intraday price and volume data")
async def get_intraday(symbol: str = Query(..., description="PSX ticker symbol")) -> List[Dict[str, Any]]:
    """Return current day's intraday price and volume series for a symbol.

    The results include timestamp (ISO format), price, volume and
    calculated VWAP. If intraday data is not available, a 400 error is
    raised.
    """
    ticker = PSXTicker(symbol)
    try:
        df = ticker.get_intraday_data()
        if df.empty:
            raise ValueError("No intraday data returned")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch intraday data: {e}")
    # Normalize column names
    df = df.rename(columns=lambda x: x.strip().lower())
    if 'timestamp' in df.columns:
        df['Time'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['Time'] = pd.to_datetime(df.index)
    df = df.rename(columns={'price': 'Price', 'volume': 'Volume'})
    df = df[['Time', 'Price', 'Volume']].sort_values('Time')
    # Compute VWAP
    df['VWAP'] = (df['Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    # Convert times to ISO strings for JSON serialization
    # Convert timestamps to ISO strings. pandas does not expose
    # `.dt.isoformat()` so we cast to string instead. This yields
    # strings like '2025-08-08 09:35:00', which are acceptable for
    # JSON output. Clients can parse these as needed.
    df['Time'] = df['Time'].astype(str)
    return df.to_dict(orient="records")


@app.get("/announcements", summary="Latest company announcements")
async def get_announcements(
    symbol: str = Query(..., description="PSX ticker symbol"),
    max_results: int = Query(5, ge=1, le=20, description="Maximum number of announcements to return")
) -> List[Dict[str, str]]:
    """Return the latest announcements for a given symbol.

    This endpoint scrapes the PSX announcements page directly. If the
    site is unreachable or parsing fails, an empty list is returned.
    """
    url = "https://dps.psx.com.pk/announcements/companies"
    announcements: List[Dict[str, str]] = []
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"id": "announcementsTable"})
        if not table:
            return []
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
            row_symbol = cols[2].text.strip()
            if row_symbol.upper() != symbol.upper():
                continue
            date = cols[0].text.strip()
            time_ = cols[1].text.strip()
            company = cols[3].text.strip()
            title = cols[4].text.strip()
            # Determine PDF link
            pdf_link = None
            pdf_tag = cols[5].find("a", href=True, string=lambda x: x and 'PDF' in x)
            if pdf_tag:
                pdf_link = "https://dps.psx.com.pk" + pdf_tag['href']
            announcements.append({
                "date": date,
                "time": time_,
                "symbol": row_symbol,
                "company": company,
                "title": title,
                "pdf_link": pdf_link
            })
            if len(announcements) >= max_results:
                break
    except Exception:
        # If scraping fails, return an empty list rather than erroring
        return []
    return announcements


@app.get("/comparative", summary="Comparative analysis for multiple symbols")
async def get_comparative(
    symbols: List[str] = Query(..., description="Comma-separated list of up to 4 ticker symbols"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
) -> Dict[str, Any]:
    """Return basic comparative analytics for up to four symbols.

    For each symbol this endpoint returns closing prices, daily
    returns, 50-day moving average, 200-day moving average, traded
    volume and price volatility over the specified date range. A
    summary of average return and volatility is also included.
    """
    if len(symbols) == 0 or len(symbols) > 4:
        raise HTTPException(status_code=400, detail="Please provide between 1 and 4 symbols.")
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    combined = {}
    summary = []
    for sym in symbols:
        try:
            data = PSXTicker(sym).get_historical_data(start_date=start, end_date=end)
            if data.empty:
                raise ValueError("No data returned")
            data.index = pd.to_datetime(data.index)
            data = data.loc[(data.index >= start) & (data.index <= end)]
            data = data.rename(columns=lambda c: c.upper())
            prices = data['CLOSE'].astype(float)
            # Compute derived metrics
            daily_return = prices.pct_change().fillna(0)
            ma50 = prices.rolling(window=50).mean().fillna(method='bfill')
            ma200 = prices.rolling(window=200).mean().fillna(method='bfill')
            volume = data['VOLUME'].astype(float)
            volatility = prices.rolling(window=50).std().fillna(0)
            # Build record list for this symbol
            records = []
            for dt, price in prices.items():
                records.append({
                    'date': dt.date().isoformat(),
                    'close': price,
                    'return': daily_return.loc[dt],
                    'ma50': ma50.loc[dt],
                    'ma200': ma200.loc[dt],
                    'volume': volume.loc[dt],
                    'volatility': volatility.loc[dt]
                })
            combined[sym] = records
            avg_return = daily_return.mean()
            avg_volatility = volatility.mean()
            summary.append({
                'symbol': sym,
                'avg_return': avg_return,
                'avg_volatility': avg_volatility
            })
        except Exception:
            continue
    return {'series': combined, 'summary': summary}