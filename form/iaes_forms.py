from dataclasses import dataclass, astuple, field

import pandas as pd


@dataclass
class Row:
    to_date: str = None
    to_amount: str = None
    description: str = None
    from_date: str = None
    from_amount: str = None

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class TopForm:
    """Top form data. If no args are passed when initialized, default dict keys are provided with None as values.
    The cells can be populated by either passing a pre-filled dict, or by calling append().

    PLEASE NOTE: this is likely to be reimplemented. I don't like the way that it is right now, but it works for now.
    """

    cells: dict = field(default_factory=lambda: {'proj_start_date': None,
                                                 'beginning_bal': None,
                                                 'proj_min_date': None,
                                                 'proj_min_bal': None,
                                                 'total_amount': None,
                                                 'pi_amount': None,
                                                 'escrow_amount': None})
    validators: list = None

    def __post_init__(self):
        self.df = pd.Series(self.cells)

    def append(self, cname, val):
        self.cells[cname] = val
        self.df = pd.Series(self.cells)

    def validate(self):
        if self.validators is None:
            return

        for v in self.validators:
            try:
                v(self.df)
            except ValueError as e:
                return e


@dataclass
class BottomForm:
    rows: list = field(default_factory=list)
    validators: list = None

    def __post_init__(self):
        self.df = pd.DataFrame(self.rows)

    def append(self, row):
        self.rows.append(row)
        self.df = pd.DataFrame(self.rows)

    def validate(self):
        if self.validators is None:
            return

        for v in self.validators:
            try:
                v(self.df)
            except ValueError as e:
                return e


@dataclass
class TopBottomForm:
    top_form: TopForm
    bot_form: BottomForm
    validators: list

    def validate(self):
        top_errs = self.top_form.validate()
        bot_errs = self.bot_form.validate()
        for errs in [top_errs, bot_errs]:  # Return top_errs first if any, then bot_errs
            if errs:
                return errs

        for v in self.validators:
            try:
                v(self.top_form.df, self.bot_form.df)
            except ValueError as e:
                return e
