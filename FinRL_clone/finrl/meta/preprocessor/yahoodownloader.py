"""Contains classes and methods to collect data from Yahoo Finance API"""

import pandas as pd
import yfinance as yf

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API

        Args:
            proxy (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: A dataframe with date, open, high, low, close, volume, tick for specified stock tickers
        """
        data_df = pd.DataFrame()
        num_failures = 0
        for tick in self.ticker_list:
            temp_df = yf.download(tick, start=self.start_date, end=self.end_date, proxy=proxy)
            temp_df["tick"] = tick
            
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        
        # Reseting the index, using numbers as index instead of data
        data_df = data_df.reset_index()
        try:
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tick",
            ]
            data_df["close"] = data_df["adjcp"]
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        
        data_df["day"] = data_df["date"].dt.day_of_week
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of the Dataframe: ", data_df.shape)
        # print("Display Dataframe: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tick"]).reset_index(drop=True)   # Resetting the index and dropping the old one

        return data_df