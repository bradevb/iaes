from dataclasses import dataclass, astuple, field

import pandas as pd


@dataclass
class Cell:
    """Holds each cell's image, coords, row index, column name, bool for if the cell contains text, and text. Pass
    sort_by to change how the cells will be sorted (pass 0 to sort by x, 1 to sort by y)."""
    image: list = None
    coords: tuple = None
    row_idx: int = None
    col_name: str = None
    text: str = None
    sort_by: int = 1

    def __hash__(self):
        """Manually implement __hash__ for use in a set when grouping cells."""
        return hash(self.coords)

    def __lt__(self, other):
        return self.coords[self.sort_by] < other.coords[self.sort_by]


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

    def validate(self):
        if self.validators is None:
            return

        for v in self.validators:
            v(self.df)


@dataclass
class BottomForm:
    rows: list = field(default_factory=list)
    validators: list = None

    def __post_init__(self):
        self.df = pd.DataFrame(self.rows)

    def validate(self):
        if self.validators is None:
            return

        for v in self.validators:
            v(self.df)


@dataclass
class TopBottomForm:
    top_form: TopForm
    bot_form: BottomForm
    validators: list

    def validate(self):
        self.top_form.validate()
        self.bot_form.validate()

        if self.validators is None:
            return

        for v in self.validators:
            v(self.top_form.df, self.bot_form.df)
