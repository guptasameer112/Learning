import os
import sys
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

def data_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tick"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

class FeatureEngineer:
    # TODO: implement vix, turbulence, and user_defined
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """
    def __init__(
            self,
            use_technical_indicator=True,
            tech_indicator_list=config.INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def clean_data(self, data):
        """Clean the raw dataframe and deal with missing values

        Args:
            data (pandas.Dataframe): source dataframe

        Returns:
            pd.Dataframe: cleaned dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tick"], ignore_index=True)
        df.index = df.date.factorize()[0]   # returns code, uniques
        merged_closes = df.pivot_table(index="date", columns="tick", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        ticks = merged_closes.columns
        df = df[df.tick.isin(ticks)]    # removes tickers which have any date value missing, "None" does not count as missing
        
        return df
    
    def add_technical_indicator(self, data):
        """Calculates technical indicators using stockstats package

        Args:
            data (pandas.DataFrame): source dataframe

        Returns:
            pandas.DataFrame: technical indicators appended dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tick", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tick.unique()
        
        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tick == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tick"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tick == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            # print(indicator_df)
            df = df.merge(
                indicator_df[["tick", "date", indicator]], on=["tick", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tick"])
        return df
    
    def add_vix(self, data):
        """add VIX to the dataframe

        Args:
            data (pandas.DataFrame): source dataframe

        Returns:
            pd.DataFrame: dataframe with VIX
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tick"]).reset_index(drop=True)
        return df
    
    def calculate_turbulence(self, data: pd.DataFrame):
        """Calculates turbulence for each day

        Args:
            data (pd.DataFrame): source dataframe

        Raises:
            Exception: turbulence could not be computed

        Returns:
            list: list with turbulent values for the year
        """
        df = data.copy()
        unique_date = df.date.unique()
        df_price_pivot = df.pivot(index="date", columns="tick", values="close")
        df_price_pivot = df_price_pivot.pct_change()

        # Start after a year
        start = 252
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # Using one year rolling window to calculate covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            # calculate the covariance matrix and the Mahalanobis distance to determine the turbulence value for each day
            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis = 0
            )
            
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

    def add_turbulence(self, data):
        """add turbulence index

        Args:
            data (pandas.DataFrame): source dataframe

        Returns:
            pandas.DataFrame: dataframe with turbulence values for each day
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tick"]).reset_index(drop=True)
        return df

    def preprocess_data(self, df):
        """Processes data to perform feature engineering

        Args:
            df (pd.DataFrame): source dataframe

        Returns:
            DataMatrices: a datamatrices object including technical indicators
        """
        df = self.clean_data(df)

        # technical indicators
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")
        
        # vix
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")
        
        # turbulence index
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")
        
        # user defined features
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")
        df = df.ffill().bfill()
        return df
        