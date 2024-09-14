"""
    Run:
        for output: pytest -s <file location>
        without output: pytest <file location>
"""

import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "GOOG"]

def test_downlaod(ticker_list):
    """testing function for yahoodownloader.py

    Args:
        ticker_list (list): obtained from ticker_list() automatically

    Returns:
        pandas.DataFrame: dataframe containing OHCL values for ticker_list
    """
    df = YahooDownloader(
        start_date="2019-01-01", end_date="2019-02-01", ticker_list=ticker_list
    ).fetch_data()
    assert isinstance(df, pd.DataFrame)
    print(df)
    return df

def test_clean_data(ticker_list):
    """testing function for preprocessors.py

    Args:
        ticker_list (list): obtained from ticker_list() automatically
    """
    fe = FeatureEngineer()
    cleaned_df = fe.add_technical_indicator(fe.clean_data(test_downlaod(ticker_list)))
    print(cleaned_df)
