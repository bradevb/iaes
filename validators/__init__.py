"""
Validators to use when validating cells.

There are a few different types, and some of the modules are intended to be used
internally by more concrete implementations of validators that are specifically for
validating IAES cells. The internal modules are prefixed with an underscore.

The concrete implementations take an entire DataFrame of all cell data that is found.
They then extract only the information that they need to validate.
For example, a date validator receives the whole DataFrame, then it isolates the
DataFrame's to_date column, and finally performs all needed validations on said column.

These modules and their members are subject to change as development continues
and if a better method for validating is found.
"""

from validators.dates import DATE_VALIDATORS

v_list = [
    DATE_VALIDATORS,
]

VALIDATORS = []
list(map(VALIDATORS.extend, v_list))
