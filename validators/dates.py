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
