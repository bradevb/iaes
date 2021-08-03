import _date_validators


def check_to_date_column(dates):
    start = dates[0]
    end = dates[-1]

    if not _date_validators.check_year(start, end):
        return False

    results = []
    for i in range(len(dates) - 1):
        current_month = dates[i]
        next_month = dates[i + 1]
        if current_month is None or next_month is None:
            continue

        results.append(_date_validators.check_consecutive_dates(current_month, next_month))

    if not all(results):
        return False
    return True


def check_from_date_column(to_date_col, from_date_col):
    prev_from_date = None
    for to_date, from_date in zip(to_date_col, from_date_col):
        if to_date is None:
            if from_date != prev_from_date:
                return False
            continue

        if to_date != from_date:
            return False

        prev_from_date = from_date

    return True
