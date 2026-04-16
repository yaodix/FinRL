"""Data source abstractions and concrete fetchers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import os
from dotenv import load_dotenv

import pandas as pd

raw_columns = ["时间", "代码", "名称", "开盘价", "最高价", "最低价", "收盘价", "成交量", "成交额", "涨幅", "振幅"]
ticflow_columns = ["symbol", "name", "trade_time", "open", "high", "low", "close", "volume", "amount"]

map_columns = {"时间": "trade_time", "代码": "symbol", "名称": "name", "开盘价": "open", \
    "最高价": "high", "最低价": "low", "收盘价": "close", "成交量": "volume", "成交额": "amount"}


def merge_trans_raw_data(raw_dir: Path, output_file: Path) -> None:
    """Merge multiple raw TickFlow CSV files into one, sorted by timestamp."""
    all_files = list(raw_dir.glob("*.csv"))
    if not all_files:
        raise ValueError(f"No CSV files found in {raw_dir}")

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        print(f"Read {len(df)} rows from {file.name}")
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.sort_values(by="时间", inplace=True)
    
    # raw修改修改tickflow一致的列名和顺序
    merged_df.drop(columns=[col for col in merged_df.columns if col not in map_columns.keys()], inplace=True)
    merged_df = merged_df.rename(columns=map_columns)[ticflow_columns]
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(merged_df)} rows into {output_file}")

def compare_ohlcv(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    '''
    比较trade_time相同的行的ohlcv数据，返回不同的行和列
    '''
    df1.set_index("trade_time", inplace=True)
    df2.set_index("trade_time", inplace=True)
    common_times = df1.index.intersection(df2.index)
    diff_info = []
    for time in common_times:
        row1 = df1.loc[time]
        row2 = df2.loc[time]
        for col in ["open", "high", "low", "close", "volume"]:
            if row1[col] != row2[col]:
                diff_info.append({"trade_time": time, "column": col, "df1_value": row1[col], "df2_value": row2[col]})

    # 统计不同偏差绝对值的数量
    diff_count = {}
    for diff in diff_info:
        key = (diff["column"], abs(diff["df1_value"] - diff["df2_value"]))
        diff_count[key] = diff_count.get(key, 0) + 1
    print("不同数据统计:")
    for key, count in diff_count.items():
        print(f"Column: {key[0]} Absolute Difference: {key[1]} Count: {count}")    
    
    return pd.DataFrame(diff_info)
    
    

class TickflowFetcher():
    """Fetches OHLCV data from TickFlow API."""

    def __init__(self, api_key: str, local_data_path: Path, raw_cache_dir: Optional[Path] = None):
        self.api_key = api_key
        self.local_data_path = local_data_path
        self.raw_cache_dir = raw_cache_dir

    def fetch(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol and interval between start_time and end_time."""
        from tickflow import TickFlow

        tf = TickFlow(api_key=self.api_key)

        # Convert datetimes to milliseconds timestamps
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        # Optional caching of raw API response
        if self.raw_cache_dir:
            cache_file = self.raw_cache_dir / f"{symbol}_{interval}_{start_ms}_{end_ms}.csv"
            if cache_file.exists() :
                return pd.read_csv(cache_file)

        df = tf.klines.get(
            symbol,
            period=interval,
            start_time=start_ms,
            end_time=end_ms,
            as_dataframe=True,
            count=100000
        )

        if self.raw_cache_dir:
            df.to_csv(cache_file, index=False)

        return df
    
    def fetch_incremental(self, symbol: str, interval: str) -> pd.DataFrame:
        """
            Fetch new data to update existing dataset, 
        """
        if not self.local_data_path.exists():
            print(f"No existing data found at {self.local_data_path}. Fetching full dataset.")
            raise FileNotFoundError(f"Local data file not found: {self.local_data_path}")
        
        local_data = pd.read_csv(self.local_data_path)
        last_time_str = local_data["trade_time"].max()
        
        fetch_data = self.fetch(
            symbol=symbol,
            interval=interval,
            start_time=datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=1),
            end_time=datetime.now()
        )
        fetch_data.drop(columns=[col for col in fetch_data.columns if col not in ticflow_columns], inplace=True)
        
        if fetch_data.empty:
            print("No new data to fetch.")
            return local_data
        else:
            print(f"Fetched {len(fetch_data)} new rows.")
            fetch_data.to_csv(self.local_data_path, mode='a', header=not self.local_data_path.exists(), index=False)
            return pd.concat([local_data, fetch_data], ignore_index=True)


if __name__ == "__main__":
    project_dir = Path("/home/yao/Work/myproject/FinRL/etf_t0_quant")
    local_data_path = Path("/home/yao/Work/myproject/FinRL/etf_t0_quant/workdata/159740_sync_30m/159740_sync_30m.csv")
    raw_cache_dir = project_dir / "workdata" / "tickflow_raw_cache"
    load_dotenv()
    tickflow_apikey = os.getenv("TICKFLOW_API_KEY")

    
    fetcher = TickflowFetcher(api_key=tickflow_apikey, local_data_path=local_data_path, raw_cache_dir=raw_cache_dir)
    df_add = fetcher.fetch_incremental(symbol="159740.SZ", interval="30m")
    
    all_tickflow_data = fetcher.fetch(
        symbol="159740.SZ",
        interval="30m",
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2026, 12, 1)
    )
    print(f"Total rows fetched from TickFlow: {len(all_tickflow_data)}")
    
    # print(all_tickflow_data.tail(10))
    # print(df_add.tail(10))
    
    diff_info = compare_ohlcv(df_add, all_tickflow_data)
    diff_info.to_csv(project_dir / "data_comparison.csv", index=False)
    print(diff_info)
    
    