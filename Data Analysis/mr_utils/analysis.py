"""
This module contains the functions used to analyze the data collected in the project.
"""

import logging
import typing
import warnings

import numpy as np
import pandas as pd
import robusta as rst
from numpy import typing as npt
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.rinterface_lib.embedded import RRuntimeError

from mr_utils import constants as cn

rpy2_logger.setLevel(logging.ERROR)

ANOVA_KWARG_KEYS = ('subject', 'between', 'within', 'dependent', 'data',
                    'which_models')


def main(full_df: pd.DataFrame, deltas: pd.DataFrame) -> typing.Tuple[dict, dict]:
    """
    Run several ANOVA models and t-tests on the data from the experiment.

    :param full_df: The full data frame.
    :param deltas: The deltas data frame.
    :return: A dictionary containing the results of the analysis.
    """

    results_dict = {
        't_tests': {'deltas': {'between-groups': {}, 'within-groups': {}}
                    },
        'anovas': {}
    }
    sequential_analysis_results = {}

    # For the inspection stage we can look at the following models:
    # 1. Inspection RT ~ Chromacity (288 potential trials).
    results_dict['anovas']['Inspection RT ~ Chromacity'] = run_anova(
        full_df, (), cn.COLUMN_NAME_RAW_INSPECTION_RT, cn.COLUMN_NAME_EXP_GROUP)
    # 2. Inspection RT ~ Chromacity X Grid Identity (96 potential trials).
    results_dict['anovas'][
        'Inspection RT ~ Chromacity X Grid Identity'] = run_anova(
        full_df, (cn.COLUMN_NAME_INSPECTION_GRID,),
        cn.COLUMN_NAME_RAW_INSPECTION_RT, cn.COLUMN_NAME_EXP_GROUP)
    # 3. Inspection RT ~ Chromacity X Grid Variation (36 potential trials).
    results_dict['anovas'][
        'Inspection RT ~ Chromacity X Grid Variation'] = run_anova(
        full_df, (cn.COLUMN_NAME_INSPECTION_GRID_VARIATION,),
        cn.COLUMN_NAME_RAW_INSPECTION_RT, cn.COLUMN_NAME_EXP_GROUP)

    # We can't include both grid variation (8 levels), as the number of trials
    # per participant will be very small (96 / 8 = 12), which does not allow
    # for a good estimate of the mean (i.e., n < 20).

    # 4. Inspection RT ~ Color Scheme (48 potential trials).
    # Color scheme (6 levels) can be tested only for the multichromatic group,
    # but not in combination with grid identity or variation due to low number
    # of observations.
    results_dict['anovas']['Inspection RT ~ Chromacity x Color Scheme (Multichrome Group)'] = run_anova(
        full_df.loc[full_df[cn.COLUMN_NAME_EXP_GROUP] == cn.REPRESENTATIVE_GROUP_LABELS[1]],
        (cn.COLUMN_NAME_INSPECTION_COLOR_SCHEME,),
        cn.COLUMN_NAME_RAW_INSPECTION_RT, None)

    # For the rotation stage we can look at the following models:
    #  1. Rotation RT ~ Chromacity X Absolute Rotation Size (144 potential trials)
    results_dict['anovas']['Rotation RT ~ Chromacity X Absolute Rotation Size'] = run_anova(
        full_df, (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,),
        cn.COLUMN_NAME_RAW_ROTATION_RT, cn.COLUMN_NAME_EXP_GROUP)
    #  2. Rotation RT ~ Chromacity X Absolute Rotation Size X Rotation Sign (72
    #   potential trials)
    results_dict['anovas']['Rotation RT ~ Chromacity X Absolute Rotation Size X Rotation Sign'] = run_anova(
        full_df,
        (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, cn.COLUMN_NAME_ROTATION_SIGN),
        cn.COLUMN_NAME_RAW_ROTATION_RT, cn.COLUMN_NAME_EXP_GROUP)
    #  3. Rotation RT ~ Chromacity X Absolute Rotation Size X Rotation Sign X Grid Identity (24
    #   potential trials)
    results_dict['anovas'][
        'Rotation RT ~ Chromacity X Absolute Rotation Size X Rotation Sign X Grid Identity'] = run_anova(
        full_df,
        (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, cn.COLUMN_NAME_ROTATION_SIGN,
         cn.COLUMN_NAME_INSPECTION_GRID),
        cn.COLUMN_NAME_RAW_ROTATION_RT, cn.COLUMN_NAME_EXP_GROUP)

    # For the comparison stage we cannot include both condition and group at
    # the same model, as the number of conditions is different for each group.
    # In terms of dependent variable, we have both response speed and accuracy.

    # For the comparison stage we can look at the following models (for both accuracy and response speed):
    for dv in [cn.COLUMN_NAME_RAW_COMPARISON_RT, cn.COLUMN_NAME_RAW_COMPARISON_ACCURACY]:
        pretty_dv_name = cn.VARIABLES_REPRESENTATIVE_NAMES[dv]

        #  1. DV ~ Chromacity (288 potential trials)
        results_dict['anovas'][f'{pretty_dv_name} ~ Chromacity'] = run_anova(
            full_df, (), dv, cn.COLUMN_NAME_EXP_GROUP)
        #  2. DV ~ Chromacity X Rotation Size (144 potential trials)
        results_dict['anovas'][f'{pretty_dv_name} ~ Absolute Rotation Size'
        ] = run_anova(full_df, (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,), dv,
                      None)
        #  3. DV ~ Rotation Size X Chromacity (144 for the multichrome group, 36
        #   for the monochrome group).
        results_dict['anovas'][f'{pretty_dv_name} ~ Chromacity X Absolute Rotation Size'
        ] = run_anova(full_df, (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,), dv,
                      cn.COLUMN_NAME_EXP_GROUP)
        #  4. DV ~ Rotation Size X Chromacity X Rotation Sign (72 potential trials).
        results_dict['anovas'][f'{pretty_dv_name} ~ Chromacity X Absolute Rotation Size X Rotation Sign'
        ] = run_anova(full_df, (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, cn.COLUMN_NAME_ROTATION_SIGN), dv,
                      cn.COLUMN_NAME_EXP_GROUP)

        for (name, group), label in zip(
                full_df.groupby(cn.COLUMN_NAME_EXP_GROUP),
                cn.REPRESENTATIVE_GROUP_LABELS):
            results_dict['anovas'][f'{pretty_dv_name} ~ Absolute Rotation Size X Condition ({label})'] = run_anova(
                group, (cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, cn.COLUMN_NAME_COMPARISON_TYPE), dv, None)

    # For the comparison stage we can look at the following models:
    results_dict['t_tests']['deltas']['between-groups'][
        'freq'] = rst.groupwise.T2Samples(
        data=deltas, paired=False, independent=cn.COLUMN_NAME_EXP_GROUP,
        subject=cn.COLUMN_NAME_UID,
        dependent=cn.COLUMN_NAME_ROTATION_DELTA, tail='x!=y'
    )

    results_dict['t_tests']['deltas']['between-groups'][
        'bayes'] = rst.groupwise.BayesT2Samples(
        data=deltas, paired=False, independent=cn.COLUMN_NAME_EXP_GROUP,
        subject=cn.COLUMN_NAME_UID,
        dependent=cn.COLUMN_NAME_ROTATION_DELTA, tail='x!=y'
    )

    # TODO - veryify against JASP
    results_dict['t_tests']['deltas']['within-groups'] = {
        n: {'freq': rst.groupwise.T1Sample(x=g['Delta'].values, y=0,
                                           tail='x>y'),

            'bayes': rst.groupwise.BayesT2Samples(
                x=g['Delta'].values, y=np.zeros(g.shape[0]), tail='x>y',
                paired=True)
            } for n, g in deltas.groupby(cn.COLUMN_NAME_EXP_GROUP)}

    group_labels = deltas[cn.COLUMN_NAME_EXP_GROUP].map(
        dict(zip(cn.REPRESENTATIVE_GROUP_LABELS, ['x', 'y'])))
    sequential_analysis_results['between_groups'] = {'values': run_sequence_of_bayes_t_tests(
        *deltas.groupby(
            cn.COLUMN_NAME_EXP_GROUP)[cn.COLUMN_NAME_ROTATION_DELTA].apply(
            lambda s: s.values).values,
        paired=False, hypothesis='x!=y',
        order=group_labels),
        'group_labels': group_labels}

    sequential_analysis_results['within_groups'] = {
        n: run_sequence_of_bayes_t_tests(
            x=g['Delta'].values, y=np.zeros(g.shape[0]), hypothesis='x>y',
            paired=True)
        for n, g in deltas.groupby(cn.COLUMN_NAME_EXP_GROUP)}

    return results_dict, sequential_analysis_results


def run_anova(df: pd.DataFrame, within_cols: tuple,
              dependent_column_name: str,
              between_col: typing.Union[None, str] = None) -> dict:
    factor_cols = list(within_cols) + ([between_col] if between_col is not None else [])

    # BayesAnova has issues with non-string factor variables, so we convert to string.
    df = df.assign(**dict(zip(factor_cols, df[factor_cols].astype(str).values.T)))

    # Aggregate as BayesAnova has a problem with missing values, so we aggregate beforehand.
    _df = df.groupby([cn.COLUMN_NAME_UID] + factor_cols)[dependent_column_name].mean().reset_index()

    # # Some rpy2 functions have a problem with a dot in the column names.
    # sanitized_dv_name = re.sub('\.', '', dependent_column_name)
    # _df = _df.rename(
    #     columns={dependent_column_name: sanitized_dv_name})

    n = _df[cn.COLUMN_NAME_UID].nunique()
    _df = _df.groupby([cn.COLUMN_NAME_UID] + factor_cols).filter(lambda x: x[dependent_column_name].notna().all())

    new_n = _df[cn.COLUMN_NAME_UID].nunique()
    if n != new_n:
        warnings.warn(f'{n - new_n} Participants removed from data due to missing observations on at least one '
                      f'sub-groups of the independent variable(s)')

    anova_args = dict(zip(ANOVA_KWARG_KEYS, (cn.COLUMN_NAME_UID, between_col, within_cols,
                                             dependent_column_name,
                                             _df,
                                             'bottom')))

    return {cn.TEST_KEYS_BAYES: rst.groupwise.BayesAnova(**anova_args),
            cn.TEST_KEYS_FREQ: rst.groupwise.Anova(**anova_args), }


def run_t_test(x: np.ndarray, y: np.ndarray, hypothesis: str, paired: bool, test_type: str, order=None) -> typing.Union[
    rst.groupwise.T2Samples, rst.groupwise.BayesT2Samples, np.ndarray]:
    return _select_test(test_type)(x, y, hypothesis, paired, order)


def _select_test(test_type: str) -> typing.Callable:
    if test_type == cn.TEST_KEYS_FREQ:
        return _run_freq_t_test
    if test_type == cn.TEST_KEYS_BAYES:
        return _run_bayes_t_test
    if test_type == cn.T_TEST_KEYS_SEQUENTIAL_BAYES:
        return run_sequence_of_bayes_t_tests


def _run_freq_t_test(x: npt.ArrayLike, y: npt.ArrayLike, hypothesis: str, paired: bool) -> rst.groupwise.T2Samples:
    return rst.groupwise.T2Samples(
        x=x, y=y, tail=hypothesis, paired=paired)


def _run_bayes_t_test(x: npt.ArrayLike, y: npt.ArrayLike, hypothesis: str,
                      paired: bool) -> rst.groupwise.BayesT2Samples:
    return rst.groupwise.BayesT2Samples(
        x=x, y=y, tail=hypothesis, paired=paired)


def run_sequence_of_bayes_t_tests(x: npt.ArrayLike, y: npt.ArrayLike, hypothesis: str, paired: bool,
                                  order: typing.Union[npt.ArrayLike, None] = None) -> npt.ArrayLike:
    """Run a sequence of Bayesian t-tests, where the model is fit after each new data point.

    Parameters
    ----------
    x : npt.ArrayLike
        The x values.
    y : npt.ArrayLike
        The y values.
    hypothesis : str
        The hypothesis to test ('x>y', 'x<y', 'x!=y').
    paired : bool
        Whether the data is paired or not.
    order : npt.ArrayLike, optional
        The order of the data points. If None, the order is assumed to be the order of the data points in the arrays.

    Returns
    -------
    npt.ArrayLike
        The Bayes factors calculated at each data point.
    """
    x = np.array(x)
    y = np.array(y)

    # To store the results
    sequential_bf_values = np.empty(x.size + (0 if paired else y.size))

    # Initialize the model, but don't fit it yet.
    m = rst.groupwise.BayesT2Samples(x=x, y=y, tail=hypothesis, paired=paired, fit=False)

    # Iterate over the data points, and fit the model after each new data point.
    for i in range(sequential_bf_values.size):

        if paired:
            current_x_values, current_y_values = x[: i + 1], y[: i + 1]

        else:
            current_values = {'x': 0, 'y': 0}
            current_values.update(**dict(zip(*np.unique((order[: i + 1]), return_counts=True))))
            current_x_values = x[: current_values['x']]
            current_y_values = y[: current_values['y']]

        # Reset the model to include only the current data points.
        m.reset(x=current_x_values, y=current_y_values, data=None, refit=False)

        try:
            m.fit()
            bf = m.report_table().iloc[0]['bf']
        # Handle exceptions occuring when there are not enough observations to fit the model.
        except RRuntimeError as e:
            if "not enough observations" in str(e):
                bf = np.nan
            else:
                raise e
        except ValueError as ve:
            if m.data['INDEPENDENT'].nunique() == 1:
                bf = np.nan  # One group is missing from an unpaired t-test
            else:
                raise ve

        # Otherwise, store the Bayes factor.
        sequential_bf_values[i] = bf

    return sequential_bf_values
