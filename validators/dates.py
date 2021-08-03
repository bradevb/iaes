import _date_validators as _dv


def _get_date_col(dataframe, to_or_from):
    acceptable_columns = ['to', 'from']
    if to_or_from not in acceptable_columns:
        raise ValueError(f'Invalid column argument "{to_or_from}", expected one of {*acceptable_columns,}')

    column = f'{to_or_from}_date'
    return list(dataframe[column])


def ensure_consecutive_dates(dataframe):
    dates = _get_date_col(dataframe, 'to')
    dates = [x for x in dates if x is not None]  # Remove all Nones from the dates list

    start = dates[0]
    end = dates[-1]
    if not _dv.check_year(start, end):
        return False

    for current_month, next_month in zip(dates, dates[1::]):
        if not _dv.check_consecutive_dates(current_month, next_month):
            return False

    return True


def ensure_same_date_cols(dataframe):
    to_date_col = _get_date_col(dataframe, 'to')
    from_date_col = _get_date_col(dataframe, 'from')

    prev_to_date = None
    for to_date, from_date in zip(to_date_col, from_date_col):
        if to_date is not None:
            prev_to_date = to_date
        if from_date is not None:
            prev_from_date = from_date
        else:
            continue

        if prev_to_date != prev_from_date:
            return False

    return True
