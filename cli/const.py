# App and window names to use when getting screenshot from remote desktop
APP_NAME = 'Microsoft Remote Desktop'
WINDOW_NAME = 'DRPR-RDS-CAP2'

# Column names used when building the tables in main program
TOP_COL_NAMES = ['proj_start_date',
                 'beginning_bal',
                 'proj_min_date',
                 'proj_min_bal',
                 'total_amount',
                 'pi_amount',
                 'escrow_amount']
BOT_COL_NAMES = ['to_date', 'to_amount', 'description', 'from_date', 'from_amount']

# These are commonly seen descriptions. Each forms' descriptions are checked against these in the main program. The
# dictionary keys correspond to Halo's statuses. That way, when looping through the keys, one can easily just call
# spinner[key](description).
MONTH_STATUSES = {
    'succeed': [
        'MORTGAGE INSURANCE',
        'MTG INS',
        'HAZARD INSURANCE',
        'HAZ INS',
        'PROPERTY TAXES',
        'PROP TAXES',
        'COUNTY PROPERTY TAX',
        'COUNTY PROPERTY TAXES',
    ],
}
MONTH_STATUS_COLORS = {
    'fail': 'red',
    'warn': 'yellow',
    'succeed': 'green',
}

# For converting screenshots to the right color space
CSPACE_PATH = 'icc/IAES_COLOR_PROFILE.icc'

# HSV values used to replace/detect colors throughout the application
TEXT_COLOR_LOW = (0, 0, 0)
TEXT_COLOR_HIGH = (179, 255, 182)
ORANGE_LOW = (12, 190, 206)
ORANGE_HIGH = (179, 255, 255)
ORANGE_REFINED_LOW = (12, 209, 246)
ORANGE_REFINED_HIGH = (20, 255, 255)
RED_LOW = (4, 165, 255)
RED_HIGH = (11, 244, 255)
QUESTION_MARK_LOW = (111, 151, 178)
QUESTION_MARK_HIGH = (179, 172, 222)
SELECTION_LOW = (97, 158, 195)
SELECTION_HIGH = (112, 177, 255)
