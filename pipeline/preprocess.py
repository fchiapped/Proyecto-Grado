import pandas as pd


def preprocess_df(df, datetime_col='date_time'):
    """基础预处理：解析时间，去重，按时间排序，剔除全空行。"""
    if df.empty:
        return df
    df = df.copy()
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    # drop rows that are all NaN
    df = df.dropna(how='all')
    # drop exact duplicates
    df = df.drop_duplicates()
    if datetime_col in df.columns:
        df = df.sort_values(datetime_col).reset_index(drop=True)
    return df
