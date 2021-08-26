"""
Month validators

Just for clarification, a 'month' as defined here is actually an entire row of the Captiva form.

These validators check to make sure all rows have all the correct cells filled out (or that certain cells are *not*
filled out).
"""


def _get_col(dataframe, col_name, keep_none=True):
    col = list(dataframe[col_name])

    if keep_none:
        return col
    else:
        return [x for x in col if x is not None]


def _trim_trailing_rows(dataframe):
    ret_df = dataframe.copy(deep=True)
    while not any(ret_df.iloc[-1]):
        ret_df.drop(ret_df.tail(1).index, inplace=True)
    return ret_df


def ensure_to_cols_cells(dataframe):
    to_date_col = _get_col(dataframe, 'to_date')
    to_amount_col = _get_col(dataframe, 'to_amount')

    for to_col in zip(to_date_col, to_amount_col):
        if any(to_col) and not all(to_col):
            date, payment = to_col
            raise ValueError(f'Date {date} and amount {payment} must both be present, or not there.')


def ensure_from_cols_cells(dataframe):
    from_description_col = _get_col(dataframe, 'description')
    from_date_col = _get_col(dataframe, 'from_date')
    from_amount_col = _get_col(dataframe, 'from_amount')

    for from_col in zip(from_description_col, from_date_col, from_amount_col):
        if any(from_col) and not all(from_col):
            description, date, amount = from_col
            raise ValueError(f'Description {description}, date {date}, and amount {amount} must all be present.')


def ensure_no_blank_months(df):
    try:
        trimmed_df = _trim_trailing_rows(df)
    except IndexError:
        raise ValueError('There is a problem with the months. Please double check the form.')

    trimmed_df = trimmed_df.values

    for row, month in enumerate(trimmed_df):
        if not any(month):
            raise ValueError(f'Row {row + 1} is blank. There should not be any completely blank rows.')


MONTH_VALIDATORS = [
    ensure_to_cols_cells,
    ensure_from_cols_cells,
    ensure_no_blank_months,
]
