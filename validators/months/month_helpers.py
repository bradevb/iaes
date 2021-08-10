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
    temp_df.to_amount.fillna(value=0, inplace=True)
    temp_df.from_amount.fillna(value=0, inplace=True)
    return temp_df


def _calc_net(df, calc_cumnet=False):
    """Calculates net gain or loss for months, and adds results to new column inplace.
    If calc_cumnet is True, then also adds a cumulative net column."""
    df['_net'] = df.apply(lambda x: x['to_amount'] - x['from_amount'], axis=1)
    if calc_cumnet:
        df['_cumnet'] = df['_net'].cumsum()
    return df


def _group_multi_payments(frame):
    """Sums up all payments in each month."""
    # Put dates in both to and from dates for grouping
    df = frame.copy()
    df['from_date'] = df['from_date'].fillna(df['to_date'])
    df['to_date'] = df['to_date'].fillna(df['from_date'])

    funcs = {'to_date': 'first', 'to_amount': 'sum', 'from_date': 'first', 'from_amount': 'sum'}
    grouped = df.groupby(['to_date', 'from_date'], as_index=False, sort=False).agg(funcs)
    return grouped


def calc_balance(df, start_amount):
    """Calculates balance for each month. Return slice of date and balance for each month."""
    if isinstance(start_amount, str):
        start_amount = _amount_to_float(start_amount)

    temp_df = _format_amount_cols(df)
    temp_df = temp_df.dropna(how='all')
    temp_df = _group_multi_payments(temp_df)

    _calc_net(temp_df, calc_cumnet=True)
    temp_df['balance'] = temp_df['_cumnet'] + start_amount  # Add cumulative net gain/loss to start_amount

    return temp_df.loc[:, ('to_date', 'balance')]
