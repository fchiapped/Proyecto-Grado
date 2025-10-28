import pandas as pd


def get_new_rows(csv_path, last_ts=None, datetime_col='date_time', chunksize=200000):
    """增量读取 CSV 中比 last_ts 更晚的行。
    如果 last_ts 为 None，则返回整个文件（谨慎）。
    返回 (df_new, new_last_ts)
    """
    last_ts = pd.to_datetime(last_ts) if last_ts is not None else None
    new_chunks = []
    try:
        for chunk in pd.read_csv(csv_path, parse_dates=[datetime_col], chunksize=chunksize):
            if last_ts is None:
                new_chunks.append(chunk)
            else:
                mask = chunk[datetime_col] > last_ts
                if mask.any():
                    new_chunks.append(chunk.loc[mask])
    except StopIteration:
        pass
    except Exception as e:
        raise

    if not new_chunks:
        return pd.DataFrame(), last_ts

    df_new = pd.concat(new_chunks, ignore_index=True)
    df_new = df_new.sort_values(datetime_col).reset_index(drop=True)
    new_last = df_new[datetime_col].max()
    return df_new, new_last
