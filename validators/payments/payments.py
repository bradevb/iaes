from exceptions import ValidationError


def _get_payment_col(dataframe, to_or_from, keep_none=True):
    acceptable_columns = ['to', 'from']
    if to_or_from not in acceptable_columns:
        raise ValueError(f'Invalid column argument "{to_or_from}", expected one of {*acceptable_columns,}')

    column = f'{to_or_from}_amount'
    dates = list(dataframe[column])

    if keep_none:
        return dates
    else:
        return [x for x in dates if x is not None]


def ensure_payments_format(dataframe):
    to_payment_col = _get_payment_col(dataframe, 'to', keep_none=False)
    from_payment_col = _get_payment_col(dataframe, 'from', keep_none=False)

    for col in [to_payment_col, from_payment_col]:
        for amount in col:
            try:
                int(amount)
            except ValueError:
                raise ValidationError(f'Invalid payment to/from amount: {amount}.')


def ensure_same_payment_to(dataframe):
    to_payment_col = _get_payment_col(dataframe, 'to', keep_none=False)

    starting_amount = to_payment_col[0]
    for amount in to_payment_col[1:]:  # Skip first element since it is starting amount and being compared

        if amount != starting_amount:
            raise ValidationError(f'Payment to amount {amount} is not equal to starting amount {starting_amount}.')


PAYMENT_VALIDATORS = [
    ensure_payments_format,
    ensure_same_payment_to,
]
