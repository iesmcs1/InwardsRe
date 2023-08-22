import sys
import os
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

if os.getcwd().endswith('InwardsRe'):
    # Need to add parent folder to SYS PATH in order to import packages.
    sys.path.insert(0, os.path.abspath('./modules/market_risk'))
    sys.path.insert(0, os.path.abspath('./modules/market_risk/tests'))


from currency_risk import CurrencyData

if os.getcwd().endswith('InwardsRe'):
    TEST_DATA_PATH = os.path.join(os.getcwd(),
                                  'modules',
                                  'market_risk',
                                  'tests',
                                  'test_currency_risk')
else:
    TEST_DATA_PATH = '.'


@pytest.fixture
def currency_risk_data() -> CurrencyData:
    date = "30/06/2021"
    return CurrencyData(folder_path=TEST_DATA_PATH, report_date=date)


def test_import_data_balance_sheet(currency_risk_data):
    processed_bal_sheet = pd.read_csv(
        os.path.join(TEST_DATA_PATH, 'processed_RW_FX_BALANCE_SHEET.txt'),
        sep='\t'
    )

    assert currency_risk_data.data['BALANCE_SHEET'].equals(processed_bal_sheet)


def test_import_data_tp_position(currency_risk_data):

    processed_tp_pos = pd.read_csv(
        os.path.join(TEST_DATA_PATH,
                     'processed_IR_TP_POSITION_GRANULAR_CURRENCY.txt'),
        sep='\t'
    )

    assert currency_risk_data.data['TP_POSITION'].equals(processed_tp_pos)


def test_import_data_scr(currency_risk_data):
    processed_scr = pd.read_csv(
        os.path.join(TEST_DATA_PATH, 'processed_SCR.txt'),
        sep='\t',
        index_col=0
    )

    currency_risk_data.data['SCR'].to_csv('scr_test.txt')

    #round_scr_df = currency_risk_data.data['SCR'].round(4)
    for col in currency_risk_data.data['SCR'].columns:
        assert_series_equal(
            currency_risk_data.data['SCR'][col],
            processed_scr[col])


def test_get_scr_sum(currency_risk_data):
    assert currency_risk_data.get_scr_sum() == 30968629.230764803


def test_match_fc_computed_amount(currency_risk_data):
    assert currency_risk_data.FC_COMPUTED == 30968629.2308