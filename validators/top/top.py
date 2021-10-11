"""
Functions for validating only the top cells of Captiva's IAES form.

These validators can be run before parsing the bottom df, as they check the top cells against other top cells.
"""

from exceptions import ValidationError
from validators.val_helpers import amount_to_float


def ensure_total_amount(df):
    pi_amount = amount_to_float(df['pi_amount'])
    escrow_amount = amount_to_float(df['escrow_amount'])
    total_amount = amount_to_float(df['total_amount'])

    if round(pi_amount + escrow_amount, 2) != total_amount:
        raise ValidationError("Monthly PI and Escrow amounts don't add up to total monthly payment.")


def ensure_all_cells_present(df):
    cells = df.values
    if not all(cells) or len(cells) < 7:
        raise ValidationError('A cell in the top form is missing. They must all be present.')


TOP_VALIDATORS = [
    ensure_all_cells_present,
    ensure_total_amount,
]
