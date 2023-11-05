import typing
from copy import copy
from functools import partial

import numpy as np
import pandas as pd
from mr_utils import constants as cn
from mr_utils.general_utils import find_outlier_rows, remove_duplicate_series, calc_delta

DATA_PATH = 'Input Data/raw_data.csv'

PILOT_SUBJECTS = []


def export_task_files(aggregated_deltas_df: pd.DataFrame, task: pd.DataFrame,
                      aggregated_task: pd.DataFrame, pivot_aggregated_task: pd.DataFrame,
                      data_dir: str) -> None:
    aggregated_task.to_csv(f'{data_dir}/aggregated_task.csv')

    pivot_aggregated_task.to_csv(f'{data_dir}/pivot_aggregated_task.csv')

    aggregated_deltas_df.to_csv(f'{data_dir}/Deltas.csv')

    task.to_csv(f'{data_dir}/Clean.csv')


def main(data_dir, screening_set):
    raw_data = load_data()
    raw_task, vviq, debrief, demographics = split_data_to_stages(
        raw_data)

    demographics = preprocess_demographics(demographics)
    vviq = preprocess_vviq(vviq, demographics)
    debrief = preprocess_debrief(debrief, demographics)
    task, quantities_to_report = preprocess_mr_task(raw_task, demographics,
                                                    screening_set)

    aggregated_deltas_df = aggregate_for_delta_tests(task)

    aggregated_task = \
        task.groupby([cn.COLUMN_NAME_EXP_GROUP, cn.COLUMN_NAME_UID,
                      cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE]
                     )[cn.COLUMN_NAME_MEAN_ROTATION_RT].mean(

        ).reset_index()

    pivot_task = task.pivot_table(index=[cn.COLUMN_NAME_EXP_GROUP,
                                         cn.COLUMN_NAME_UID],
                                  columns=cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,
                                  values=cn.COLUMN_NAME_MEAN_ROTATION_RT,
                                  aggfunc='first').reset_index()

    pivot_task.columns = [''.join([i for i in str(col)]).strip() for col in pivot_task.columns.values]

    export_task_files(aggregated_deltas_df, task, aggregated_task, pivot_task,
                      data_dir)

    return (task, aggregated_task, aggregated_deltas_df,
            quantities_to_report, vviq, debrief, demographics)

    # Removing pilot participants


def load_data() -> pd.DataFrame:
    """
    Load the data, excluding pilot participants and participants which performed
     more than one session.

    For multi-session participants return only the first session.

    :return:
    df: pd.DataFrame containing the data from the experiment.
    """
    df = pd.read_csv(DATA_PATH)

    df = df.loc[~df[cn.COLUMN_NAME_UID].isin(PILOT_SUBJECTS)]

    # One participant completed the experiment twice,
    #  we need to remove the second run.
    df[cn.COLUMN_NAME_SESSION_TIMESTAMP] = pd.to_datetime(
        df[cn.COLUMN_NAME_SESSION_TIMESTAMP])
    df = df.sort_values([cn.COLUMN_NAME_SESSION_TIMESTAMP,
                         cn.COLUMN_NAME_UID, cn.COLUMN_NAME_TRIAL_NUM])
    df = remove_duplicate_series(df)

    return df


def split_data_to_stages(df: pd.DataFrame) -> typing.Tuple[pd.DataFrame]:
    # Remove instructions and columns related to instructions slides
    df = df.loc[:, (df.columns.difference(df.filter(like='inst').columns))]

    # Filter trial rows and task related columns
    mr_task = df.loc[df['init_trial_kb.keys'].notna(),
    df.columns.difference(df.filter(
        regex='inst|vviq|.keys|thisTrial|thisIndex').columns
                          )].dropna(axis=1, how='all')

    vviq = df.loc[
        ~pd.isna(df['vviq_item_slider.rt']),
        df.filter(like='vviq').columns.tolist() +
        cn.MULTI_COLUMN_NAMES_UID]

    debrief = df.loc[(df['open_ended_kb.keys'].notna()) | (
        df['talk_rather_than_imagine_q2_kb.keys'].notna()),
                     df.filter(
                         regex='talk_rather_than_imagine|open_ended_kb|enter_seq_resp').columns.tolist() +
                     cn.MULTI_COLUMN_NAMES_UID]

    demographics = df.loc[
        ~pd.isna(df['demog_loop.thisRepN']),
        df.filter(
            regex='^demog_que_(.*?)|color_blindness_q_kb').columns.tolist() +
        cn.MULTI_COLUMN_NAMES_UID]

    return mr_task, vviq, debrief, demographics


def preprocess_mr_task(mr_task: pd.DataFrame, demographics: pd.DataFrame,
                       screening_set: dict) -> typing.Tuple[
    pd.DataFrame, typing.Dict]:
    """Transform and clean the MR task data.

    Returns the clean task data, and a dictionary with quantities of removed
    data for generating a report of the preprocessing procedure.
    """

    removed_data = {}

    # We have both positive and negative rotations, but want to ignore
    # it for the time being.
    mr_task[cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE] = mr_task[
        cn.COLUMN_NAME_RAW_ROTATION].abs().values

    mr_task[cn.COLUMN_NAME_ROTATION_SIGN] = np.sign(
        mr_task[cn.COLUMN_NAME_RAW_ROTATION].values)

    mr_task[cn.RECAST_TO_INT_COLS] = mr_task[cn.RECAST_TO_INT_COLS].astype(
        int).values

    mr_task[cn.COLUMN_NAME_EXP_GROUP] = mr_task[cn.COLUMN_NAME_EXP_GROUP].map(
        cn.REPRESENTATIVE_GROUP_LABELS_DICT).values

    # Columns where the labeling would be nicer if the values begin at 1 and not
    #  at 0 (e.g., Index of color scheme).
    mr_task[
        list(set(cn.RECAST_TO_INT_COLS)
             - {cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, cn.COLUMN_NAME_ROTATION_SIGN})] = mr_task[
        set(cn.RECAST_TO_INT_COLS)
        - {cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, cn.COLUMN_NAME_ROTATION_SIGN}].add(1).values

    mr_task[cn.COLUMN_NAME_COMPARISON_TYPE] = mr_task[
        cn.COLUMN_NAME_COMPARISON_TYPE].map(cn.COMPARISON_TYPE_LABELS_DICT)

    # Change seconds to milliseconds on all columns that have relevant data.
    mr_task[mr_task.filter(like='rt').columns] *= 1000

    # There are three stages in a trial - Inspection, Rotation and Comparison.
    #  We sum the RTs to get a measure of the duration of each trial.
    mr_task[cn.COLUMN_NAME_TOTAL_ROUTINE_RT] = (
        mr_task[cn.MULTI_COLUMN_NAMES_TASK_STAGES_RT].sum(axis=1))

    # Remove RTs on trials where RT comes from an incorrect responses
    mr_task[cn.COLUMN_NAME_RAW_COMPARISON_ACCURACY] = mr_task[
        cn.COLUMN_NAME_COMPARISON_ACCURACY].eq(1)

    incorrect_responses_indices = mr_task[
        cn.COLUMN_NAME_COMPARISON_ACCURACY].ne(1)
    mr_task.loc[
        incorrect_responses_indices, cn.MULTI_COLUMN_NAMES_TASK_STAGES_RT] = np.nan

    # Record proportion of data removed due to errors
    removed_data[
        'incorrect_response_%'] = 100 * incorrect_responses_indices.mean()

    # Remove RTs on trials where any of the trial stages' RT > CUTOFF_SLOW_RT_RAW_UNITS seconds
    prolonged_responses = mr_task[cn.MULTI_COLUMN_NAMES_TASK_STAGES_RT].max(
        axis=1).gt(cn.CUTOFF_SLOW_RT_RAW_UNITS)
    mr_task.loc[
        prolonged_responses, cn.MULTI_COLUMN_NAMES_TASK_STAGES_RT] = np.nan
    # Record proportion of data removed due to extremely slow responses
    removed_data['prolonged_response_%'] = 100 * prolonged_responses.mean()

    # Remove RTs on trials where the total RT is above/below 99% of all other
    # trials (individually for each participant)
    remaining_outlier_trials = mr_task.groupby(cn.COLUMN_NAME_UID).apply(
        partial(find_outlier_rows,
                low_boundary=screening_set['CUTOFF_FAST_RT_CUMULATIVE_FREQ'],
                high_boundary=screening_set['CUTOFF_SLOW_RT_CUMULATIVE_FREQ']
                )).values

    mr_task.loc[
        remaining_outlier_trials, cn.MULTI_COLUMN_NAMES_TASK_STAGES_RT] = np.nan
    # Record proportion of remaining of RT in the percentiles <=1 and <=99
    removed_data[
        'remaininig_outlier_responses_%'] = 100 * remaining_outlier_trials.mean()

    # Calculate mean inspection RT for each participant.
    mr_task[cn.COLUMN_NAME_MEAN_INSPECTION_RT] = mr_task.groupby(
        [cn.COLUMN_NAME_UID])[cn.COLUMN_NAME_RAW_INSPECTION_RT].transform(
        'mean')

    # Calculate mean RT for each participant on rotation stage, for each
    # rotation level.
    mr_task[cn.COLUMN_NAME_MEAN_ROTATION_RT] = mr_task.groupby(
        [cn.COLUMN_NAME_UID, cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE]

    )[cn.COLUMN_NAME_RAW_ROTATION_RT].transform('mean')

    # Calculate mean accuracy and mean RT for each participant, for each level
    # of rotation, under each comparison type.
    mr_task[[cn.COLUMN_NAME_MEAN_COMPARISON_ACCURACY,
             cn.COLUMN_NAME_MEAN_COMPARISON_RT]] = mr_task.groupby(
        [cn.COLUMN_NAME_UID, cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,
         cn.COLUMN_NAME_COMPARISON_TYPE]
    )[[cn.COLUMN_NAME_COMPARISON_ACCURACY,
       cn.COLUMN_NAME_RAW_COMPARISON_RT]].transform(
        'mean')

    # Remove all RTs from the remaining
    mr_task.loc[
        (mr_task[cn.MULTI_COLUMN_NAMES_TASK_STAGES_RT].isna().any(axis=1)
         ) | incorrect_responses_indices,
        cn.COLUMN_NAME_TOTAL_ROUTINE_RT] = np.nan

    # Count the proportion of valid trials per participant
    mr_task[cn.COLUMN_NAME_VALID_TRIALS_PROPORTION] = (
        mr_task.groupby(
            cn.COLUMN_NAME_UID)[cn.COLUMN_NAME_TOTAL_ROUTINE_RT].transform(
            lambda s: (~s.isna()).mean()))

    poor_performance_participants = mr_task.loc[
        mr_task[cn.COLUMN_NAME_VALID_TRIALS_PROPORTION].le(
            screening_set['MINIMAL_VALID_TRIALS_PROPORTION']
        ), cn.COLUMN_NAME_UID].unique()

    removed_data['poor_performance_participants_%'] = 100 * (
            poor_performance_participants.size / mr_task[
        cn.COLUMN_NAME_UID].nunique())

    removed_data[
        'poor_performance_participants_N'] = poor_performance_participants.size

    mr_task = mr_task.loc[~mr_task[cn.COLUMN_NAME_UID].isin(
        poor_performance_participants)]

    # Finally, sort by date of session
    mr_task = mr_task.sort_values(cn.COLUMN_NAME_SESSION_TIMESTAMP)

    removed_data['color_blind_participants'] = (
            demographics['color_blindness'] == "None").sum()

    removed_data['total_data_removed_%'] = 100 * mr_task[
        cn.COLUMN_NAME_TOTAL_ROUTINE_RT].isna().mean()

    return drop_color_blind_participants(mr_task, demographics), removed_data


def aggregate_for_delta_tests(mr_task: pd.DataFrame):
    return mr_task.groupby(
        cn.MULTI_COLUMN_NAMES_UID).apply(
        partial(calc_delta, grouper=cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE,
                vals_column=cn.COLUMN_NAME_RAW_ROTATION_RT, agg_func='mean')
    ).reset_index(
    ).rename(columns={180: cn.COLUMN_NAME_ROTATION_DELTA}
             )


def preprocess_vviq(df: pd.DataFrame, demographics: pd.DataFrame
                    ) -> pd.DataFrame:
    return drop_color_blind_participants(df.groupby(cn.MULTI_COLUMN_NAMES_UID)[
                                             'vviq_item_slider.response'].mean().reset_index(),
                                         demographics)


def preprocess_demographics(df):
    color_blindness = df['color_blindness_q_kb.keys'].map(
        dict(zip(range(1, 9),
                 ("None",
                  "Protanopia (Can't see red)",
                  "Protanomaly (Sees some shades of red)",
                  "Deuteranopia (Can't see green)",
                  "Deuteranomaly (Sees some shades of green)",
                  "Tritanopia (Can't see blue)",
                  "Tritanomaly (Sees some shades of blue)",
                  "Achromatopsia (Can't see most colors)")))
    ).astype('category')

    sex = df['demog_que_sex_slider.response'].map(
        dict(zip(range(1, 6),
                 ["Male", "Female", "Trans", "Non-Binary", "Rather not say"]))
    ).astype('category')

    age = df['demog_que_age_slider.response'].copy()

    return df[[cn.COLUMN_NAME_UID, cn.COLUMN_NAME_SESSION_TIMESTAMP,
               cn.COLUMN_NAME_EXP_GROUP]].assign(
        sex=sex, age=age, color_blindness=color_blindness)


def preprocess_debrief(df: pd.DataFrame, demographics: pd.DataFrame):
    talk_rather_than_imagine = df['talk_rather_than_imagine_q2_kb.keys'].map(
        dict(zip(range(1, 7),
                 [f'{n}%' for n in range(0, 120, 20)]))
    ).astype('category').dropna().values

    monochrome_ques = [
        "Did you have any specific strategy to rotate the squares in your mind's "
        "eye?",
        "What do you think was the goal of the experiment?",
        "Do you have any questions or general comments on the experiment?"
    ]

    df.loc[~df[cn.COLUMN_NAME_EXP_GROUP], 'open_ended_que'] = np.tile(
        monochrome_ques,
        df.loc[~df[cn.COLUMN_NAME_EXP_GROUP]].shape[0] // len(monochrome_ques)
    )

    multichrome_ques = copy(monochrome_ques)
    multichrome_ques.insert(1,
                            "Have you memorized the order of colors verbally?")
    multichrome_ques.insert(2,
                            "Have you named the colors in order to remember their order?")

    df.loc[df[cn.COLUMN_NAME_EXP_GROUP], 'open_ended_que'] = np.tile(
        multichrome_ques,
        df.loc[df[cn.COLUMN_NAME_EXP_GROUP]].shape[0] // len(multichrome_ques)
    )

    df = df.pivot(index=[cn.COLUMN_NAME_UID, cn.COLUMN_NAME_SESSION_TIMESTAMP
        , cn.COLUMN_NAME_EXP_GROUP],
                  values='enter_seq_resp',
                  columns='open_ended_que').reset_index()

    df = df.assign(talk_rather_than_imagine=talk_rather_than_imagine)

    return drop_color_blind_participants(df, demographics)


def drop_color_blind_participants(df, demographics):
    if cn.SCREEN_COLORBLIND_PARTICIPANTS:
        return df.loc[~df[cn.COLUMN_NAME_UID].isin()(
            demographics.loc[demographics['color_blindness'] != 'None',
            cn.COLUMN_NAME_UID].unique())]
    return df

# TODO all of the following should be put in a own general utilities module
################################################################################
###############################GENERAL UTILITY FUNCTIONS########################
################################################################################
# TODO This function should take the formats, and should fail if nans are found
# Also this seems to be redundant as the dates on the pooled data are
# already date-time.


# TODO This function should take a series not a dataframe


# TODO This function should be generalized


# TODO Finish docstring. Add a base-level option, if there are more than two
#  levels.
