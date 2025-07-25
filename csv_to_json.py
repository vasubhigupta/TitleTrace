import pandas as pd

def extract_headings_from_df(df: pd.DataFrame):
    if 'predicted_label' not in df.columns:
        raise ValueError("Missing 'predicted_label' in DataFrame.")

    filtered_df = df[df['predicted_label'].isin(['H1', 'H2', 'H3', 'Title'])]

    output_df = filtered_df[['text', 'predicted_label', 'page_number']].copy()
    output_df.rename(columns={'predicted_label': 'type'}, inplace=True)

    return output_df.to_dict('records')
