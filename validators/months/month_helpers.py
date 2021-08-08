import pandas as pd


def _amount_to_float(amount):
    """Inserts a decimal before the last two characters of amount and returns a float."""
    if amount is None:
        return None
    return float(f'{amount[:-2]}.{amount[-2:]}')


def _format_amount_cols(df):
    """Returns a copy of df, with transactions formatted as floats."""
    temp_df = df.copy()
    temp_df['to_amount'] = temp_df['to_amount'].apply(lambda x: _amount_to_float(x))
    temp_df['from_amount'] = temp_df['from_amount'].apply(lambda x: _amount_to_float(x))
    temp_df.fillna(value=0, inplace=True)
    return temp_df


def _calc_net(df, calc_cumnet=False):
    """Calculates net gain or loss for months, and adds results to new column inplace.
    If calc_cumnet is True, then also adds a cumulative net column."""
    df['_net'] = df.apply(lambda x: x['to_amount'] - x['from_amount'], axis=1)
    if calc_cumnet:
        df['_cumnet'] = df['_net'].cumsum()
    return df


def calc_balance(df, start_amount):
    """Calculates balance for each month. Adds a balance column to df inplace."""
    if isinstance(start_amount, str):
        start_amount = _amount_to_float(start_amount)

    temp_df = _format_amount_cols(df)
    _calc_net(temp_df, calc_cumnet=True)
    df['balance'] = temp_df['_cumnet'] + start_amount  # Add cumulative net gain/loss to start_amount

    # Get aggregate balances in to and from cols
    to_group = df.groupby('to_date', sort=False).agg({'balance': 'sum'})
    from_group = df.groupby('from_date', sort=False).agg({'balance': 'sum'})
    balance = pd.concat([to_group, from_group]).drop_duplicates()

    return balance
