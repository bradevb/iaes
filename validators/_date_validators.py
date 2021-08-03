from datetime import datetime
from dateutil import relativedelta


def check_date_format(date):
    if len(date) != 4:
        return False
    try:
        datetime.strptime(date, '%m%y')
        return True
    except ValueError:
        return False


def check_consecutive_dates(date_1, date_2):
    d1 = datetime.strptime(date_1, '%m%y')
    d2 = datetime.strptime(date_2, '%m%y')

    month_from_d1 = d1 + relativedelta.relativedelta(months=1)
    if month_from_d1 == d2:
        return True
    return False


def check_year(date_1, date_2):
    d1 = datetime.strptime(date_1, '%m%y')
    d2 = datetime.strptime(date_2, '%m%y')

    year_from_d1 = d1 + relativedelta.relativedelta(months=11)
    if year_from_d1 == d2:
        return True
    return False


def check_to_date_column(dates):
    start = dates[0]
    end = dates[-1]

    if not check_year(start, end):
        return False

    results = []
    checked = []
    for i in range(len(dates) - 1):
        current_month = dates[i]
        next_month = dates[i + 1]
        if current_month is None or next_month is None:
            continue

        results.append(check_consecutive_dates(current_month, next_month))

    if not all(results):
        print('THAR BE AN ERROR IN DIS ONE BRUV')
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
