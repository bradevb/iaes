"""
This module contains functions to validate the top cells with the bottom df. Unlike the regular df validators,
these validators accept two arguments: a pandas Series of the top cells, and a DataFrame of the bottom df.

Because these validators require both the top cells and the bottom df, these validators should only be run AFTER
the entire bottom df has been parsed.
"""

from validators.months.month_helpers import calc_balance
from validators.val_helpers import amount_to_float as atf


def ensure_start_date_eq_bottom(top_df, bot_df):
    proj_start_date = top_df['proj_start_date']
    bot_start_date = bot_df['to_date'][0]
    if proj_start_date != bot_start_date:
        raise ValueError(f"Projected start date {proj_start_date} and table start date {bot_start_date} don't match.")


def ensure_min_date_eq_bottom(top_df, bot_df):
    start_amount = atf(top_df['beginning_bal'])
    proj_min_date, proj_min_bal = top_df['proj_min_date'], atf(top_df['proj_min_bal'])

    balance = calc_balance(bot_df, start_amount)
    min_idx = balance.balance.idxmin()
    bot_min_date, bot_min_bal = balance.iloc[min_idx]

    if proj_min_date != bot_min_date:
        raise ValueError(f"Projected min date {proj_min_date} and table min date {bot_min_date} don't match.")
    if proj_min_bal != bot_min_bal:
        raise ValueError(f"Projected min balance {proj_min_bal} and table min balance {bot_min_bal} don't match.")


def ensure_escrow_pay_eq_bottom(top_df, bot_df):
    escrow_amount = top_df['escrow_amount']
    to_amount = bot_df['to_amount']
    if to_amount.where(to_amount != escrow_amount).any():
        raise ValueError('A value in To Amount does not equal the expected monthly escrow amount.')


TOP_BOTTOM_VALIDATORS = [
    ensure_start_date_eq_bottom,
    ensure_min_date_eq_bottom,
    ensure_escrow_pay_eq_bottom,
]
