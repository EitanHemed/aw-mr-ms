import typing

import pandas as pd
from mr_utils import constants as cn


def coerce_pavlovia_timestamp(s: pd.Series) -> pd.Series:
    """
    Convert a series of strings into datetime.

    Two formats are supported, Pavlovia datetime format can be one of two
    options.

    :param s: pd.Series
        A series of strings containing time stamps.
    :return: S: pd.Series
        A series of strings datetime objects.
    """
    return pd.to_datetime(s, format=cn.DATE_FORMAT,
                          errors='coerce').fillna(
        pd.to_datetime(s, format=cn.DATE_FORMAT_FALLBACK,
                       errors='coerce'))


def find_outlier_rows(df: pd.DataFrame, low_boundary: float,
                      high_boundary: float) -> pd.Series:
    """"
    Return whether rows on a specific column are within a range of
    cumulative frqeuency values.

    :param df: pd.DataFrame
    :return: S: pd.Series
        A series of bools denoting whether trials are within the required
        range.

    """
    return ~df[cn.COLUMN_NAME_TOTAL_ROUTINE_RT].between(
        df[cn.COLUMN_NAME_TOTAL_ROUTINE_RT].quantile(
            q=low_boundary, interpolation='higher'),
        df[cn.COLUMN_NAME_TOTAL_ROUTINE_RT].quantile(
            q=high_boundary, interpolation='lower'))



def remove_duplicate_series(df):

    # TODO - this needs to be generalized
    second_run_sessions = (
        df.loc[df.groupby(cn.COLUMN_NAME_UID)[
                   cn.COLUMN_NAME_SESSION_TIMESTAMP].transform(
            'nunique').gt(1),
        ].groupby(cn.COLUMN_NAME_UID)[
            cn.COLUMN_NAME_SESSION_TIMESTAMP].max().to_dict()
    )

    return df.loc[
        ~((df[cn.COLUMN_NAME_UID].isin(second_run_sessions.keys())) &
          (df[cn.COLUMN_NAME_SESSION_TIMESTAMP].isin(
              second_run_sessions.values())))]


def calc_delta(df: pd.DataFrame, grouper: str, vals_column: str,
               agg_func: typing.Union[
                   str, typing.Callable] = 'median') -> pd.Series:
    """

    :param df: pd.DataFrame
        The dataframe containing the values to aggregate.
    :param grouper: str
        String of the column name of the two levels to .
    :param vals_column:
    :param agg_func:
        Function to use in aggregation.
    :return: df
        pd.DataFrame with the columns of `grouper` and `vals_column`.
    """
    return df.groupby([grouper])[vals_column].agg(agg_func).diff().dropna()
