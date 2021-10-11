import validators.dates._date_helpers as _dv
from exceptions import ValidationError


def _get_date_col(dataframe, to_or_from, keep_none=True):
    acceptable_columns = ['to', 'from']
    if to_or_from not in acceptable_columns:
        raise ValueError(f'Invalid column argument "{to_or_from}", expected one of {*acceptable_columns,}')

    column = f'{to_or_from}_date'
    dates = list(dataframe[column])

    if keep_none:
        return dates
    else:
        return [x for x in dates if x is not None]


def ensure_correct_date_format(dataframe):
    to_date_col = _get_date_col(dataframe, 'to', keep_none=False)
    from_date_col = _get_date_col(dataframe, 'from', keep_none=False)

    for col in [to_date_col, from_date_col]:
        for date in col:
            if not _dv.check_date_format(date):
                raise ValidationError(f'Date {date} is not the correct format. It should be: MMYY.')


def ensure_total_timespan(dataframe):
    dates = _get_date_col(dataframe, 'to', False)
    start = dates[0]
    end = dates[-1]

    if not _dv.check_year(start, end):
        raise ValidationError('The first and last months are not one year apart.')


def ensure_no_to_date_duplicates(dataframe):
    to_date_col = _get_date_col(dataframe, 'to', False)

    checked = []
    for date in to_date_col:
        if date in checked:
            raise ValidationError(f'PAYMENT TO date {date} has a duplicate.')
        checked.append(date)


def ensure_consecutive_dates(dataframe):
    dates = _get_date_col(dataframe, 'to', keep_none=False)  # Remove all Nones from the dates list

    for current_month, next_month in zip(dates, dates[1::]):
        if not _dv.check_consecutive_dates(current_month, next_month):
            raise ValidationError(f'Months {current_month} and {next_month} are not consecutive.')


def ensure_same_date_cols(dataframe):
    to_date_col = _get_date_col(dataframe, 'to')
    from_date_col = _get_date_col(dataframe, 'from')

    prev_to_date = to_date_col[-1]
    for to_date, from_date in zip(to_date_col, from_date_col):
        if to_date is not None:
            prev_to_date = to_date
        if from_date is not None:
            prev_from_date = from_date
        else:
            continue

        if prev_to_date != prev_from_date:
            raise ValidationError(f'PAYMENT TO date {prev_to_date} is not equal to PAYMENT FROM date {prev_from_date}.')


DATE_VALIDATORS = [
    ensure_correct_date_format,
    ensure_no_to_date_duplicates,
    ensure_consecutive_dates,
    ensure_total_timespan,
    ensure_same_date_cols,
]
