def amount_to_float(amount):
    """Inserts a decimal before the last two characters of amount and returns a float."""
    if amount is None:
        return None
    return float(f'{amount[:-2]}.{amount[-2:]}')
