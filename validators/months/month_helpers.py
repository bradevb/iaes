from validators.val_helpers import amount_to_float


def _format_amount_cols(frame):
    """Returns a copy of frame, with transactions formatted as floats."""
    df = frame.copy()
    df['to_amount'] = df['to_amount'].apply(lambda x: amount_to_float(x))
    df['from_amount'] = df['from_amount'].apply(lambda x: amount_to_float(x))
    df = df.fillna({'to_amount': 0, 'from_amount': 0})
    return df


def _calc_net(frame):
    """Calculates net gain or loss for months, and returns a copy of frame with net columns."""
    df = frame.copy()
    df['_net'] = df.apply(lambda x: x['to_amount'] - x['from_amount'], axis=1)
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


def calc_balance(frame, start_amount):
    """Calculates balance for each month. Return slice of date and balance for each month."""
    if isinstance(start_amount, str):
        start_amount = amount_to_float(start_amount)

    df = frame.dropna(how='all')
    df = _format_amount_cols(df)
    df = _group_multi_payments(df)
    df = _calc_net(df)

    df['balance'] = df['_cumnet'] + start_amount  # Add cumulative net gain/loss to start_amount

    return df.loc[:, ('to_date', 'balance')]
