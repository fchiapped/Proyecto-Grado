import os
import yaml
import pandas as pd
from pipeline.ingest import get_new_rows
from pipeline.preprocess import preprocess_df
from pipeline.detector import detect_chunk

STATE_FILE = 'pipeline/state.yml'


def load_config(path='pipeline/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        yaml.safe_dump(state, f)


def append_flags(flags_df, flags_path):
    os.makedirs(os.path.dirname(flags_path), exist_ok=True)
    if flags_df.empty:
        return
    if not os.path.exists(flags_path):
        flags_df.to_csv(flags_path, index=False)
    else:
        flags_df.to_csv(flags_path, index=False, header=False, mode='a')


def run_once():
    cfg = load_config()
    plants = cfg.get('plants', [])
    state = load_state()
    total_flags = 0

    for p in plants:
        name = p['name']
        csv = p['csv_path']
        dt_col = p.get('datetime_col', 'date_time')
        cols = p.get('columns', [])
        last_ts = state.get(name)

        df_new, new_last = get_new_rows(csv, last_ts=last_ts, datetime_col=dt_col)
        if df_new is None or df_new.empty:
            print(f'No new rows for {name}')
            continue

        df_new = preprocess_df(df_new, datetime_col=dt_col)

        if not cols:
            cols = [c for c in df_new.columns if c != dt_col and pd.api.types.is_numeric_dtype(df_new[c])]

        all_flags = []
        for col in cols:
            df_col = df_new[[dt_col, col]].dropna(subset=[col]).copy()
            if df_col.empty:
                continue
            detected = detect_chunk(df_col, col)
            if not detected.empty:
                flags = detected[detected['label'] != 'normal'].copy()
                if not flags.empty:
                    flags['plant'] = name
                    flags['metric'] = col
                    all_flags.append(flags[[dt_col, 'plant', 'metric', 'outlier_score', 'change_score', 'label']])

        if all_flags:
            df_flags = pd.concat(all_flags, ignore_index=True)
            flags_path = p.get('flags_path', f"df_procesados/flags_{name}.csv")
            append_flags(df_flags, flags_path)
            print(f"Appended {len(df_flags)} flags for {name} to {flags_path}")
            total_flags += len(df_flags)

        if new_last is not None:
            state[name] = str(new_last)
            save_state(state)

    print(f'Total flags appended: {total_flags}')

if __name__ == '__main__':
    run_once()
