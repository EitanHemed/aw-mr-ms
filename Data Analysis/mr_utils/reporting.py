import re

import numpy as np

from mr_utils import constants as cn

def main(path, demographics, screening_dict, results_dict, screening_set):

    with open(f'{path}/report.txt', 'w', encoding='utf-8') as f:

        print("Participants:", file=f)
        print(gen_demographics_report(demographics), file=f)
        print("\n", file=f)

        print("Data Screening:", file=f)
        print(gen_screening_report(screening_dict, screening_set), file=f)
        print("\n", file=f)

        print("Results:", file=f)

        print(f"ANOVA models", file=f)
        for k, v in results_dict['anovas'].items():
            print(f"{k}", file=f)
            print(gen_anova_report(results_dict['anovas'][k]), file=f)
            print("\n", file=f)
            test_data = v['freq'].data.copy()
            aggregated_test_data = test_data.groupby(v['freq'].independent)
            subgroup_sizes = aggregated_test_data.size()

            descriptives = prettify_descriptives(
                aggregated_test_data[v['freq'].dependent])

            print(descriptives.to_markdown(
                tablefmt="grid", index=False),
                  file=f)
            print("\n", file=f)

        # Between groups Deltas t-test
        directionality = (results_dict['t_tests']['deltas']
                                       ['between-groups']['freq'].tail)
        print(f"RT Deltas Between groups Welch T-test ({directionality})",
              file=f)
        print(gen_ttest_report(results_dict['t_tests']['deltas']
                                       ['between-groups']), file=f)
        print("\n", file=f)

        # Within groups Deltas t-test
        print(f"One-sample test of Deltas VS. mu = 0: ", file=f)
        for k, v in results_dict['t_tests']['deltas']['within-groups'].items():
            directionality = v['freq'].tail
            print(f"Group: {k}, tail {directionality}:", file=f)
            print(gen_ttest_report(v), file=f)
            print("\n", file=f)

def gen_demographics_report(demographics):
    color_blindness_exclusion_stub = (
        ' (excluding data from participants which registered to the experiment, '
        'but reported during the demographics questionnaire '
        'that they suffer from color-blindness)'
        if cn.SCREEN_COLORBLIND_PARTICIPANTS else '')
    return (
        f"We collected data from {demographics.shape[0]} participants"
        f'{color_blindness_exclusion_stub}. '
        f"Participants' ages were {demographics['age'].min():.0f}"
        f"-{demographics['age'].max():.0f} (M = {demographics['age'].mean():.2f}, "
        f"SD = {np.std(demographics['age']):.2f}), "
        f"{(demographics['sex'] == 'Female').sum():.0f} "
        f"({(demographics['sex'] == 'Female').mul(100).mean():.2f}%) "
        'identified as female. '
        'Participants were recruited using Prolific. Data was collected '
        'remotely using Pavlovia. Participants received £7.5 (GBP) as an '
        'hourly rate, in addition to an optional bonus (see below). '
    )

def gen_screening_report(screening_dict, screening_set):
    return (
        "Note that data from a participant or a specific trial can be "
        "invalid due to more than one reason. We removed task trials where "
        "responses on the Comparison stage were incorrect "
        f"({screening_dict['incorrect_response_%']:.2f}% of all responses), "
        f"trials where the response time on any of the three stages was longer "
        f"than {(cn.CUTOFF_SLOW_RT_RAW_UNITS / 1000) :.0f} "
        f"Seconds ({screening_dict['prolonged_response_%']:.2f}%). "
        "Then, we removed trials where the total response time was slower than "
        f"{(100 * screening_set['CUTOFF_SLOW_RT_CUMULATIVE_FREQ']):.2f}% or faster than "
        f"{(100 * screening_set['CUTOFF_FAST_RT_CUMULATIVE_FREQ']):.2f}% of the trials, "
        "calculated individually for each participant "
        f"({screening_dict['remaininig_outlier_responses_%']:.2f}% of all "
        f"data). Next we removed the data of "
        f"{screening_dict['poor_performance_participants_N']:.0f} "
        f"({screening_dict['poor_performance_participants_%']:.2f}%) "
        f"participants which had less than "
        f"{int(100 * screening_set['MINIMAL_VALID_TRIALS_PROPORTION'])}% of valid responses, "
        f"either due to slow responding, or low accuracy. In total, we removed "
        f"{screening_dict['total_data_removed_%']:.2f}% of the data, either in "
        f"single trials or whole participants."
    )

def gen_anova_report(anovas_dict):
    bayes_anova_text = anovas_dict[cn.TEST_KEYS_BAYES].report_text()

    freq_anova_text = anovas_dict[cn.TEST_KEYS_FREQ].report_text()
    freq_anova_text = prettify_robusta_frqeuentist_anova_str(freq_anova_text)

    bayes_anova_terms = prettify_robusta_bayesian_anova_str(bayes_anova_text)

    anova_terms = []
    for fa, fb in zip(freq_anova_text.split(']'),
                      bayes_anova_terms):
        anova_terms.append(f'{fa}, {fb}]')

    complete_anova_report = ''.join(anova_terms)
    return complete_anova_report

def prettify_robusta_bayesian_anova_str(s: str):
    terms = re.findall(r"[^[]*\[([^]]*)\]", s)
    terms = [re.sub('Error = 0.001', 'Error <= 0.001', term) for term in terms]
    return terms


def prettify_robusta_frqeuentist_anova_str(s: str) -> str:
    """Pretify the output of robusta's frquentist ANOVA text report.

    First modify according to reporting standards (APA). Then update the names
    of factor variables and interactions.
    """

    # P-values are always between 0 and 1
    s = s.replace('p = 0.', 'p = .')
    # Very small p-values should be replaced with 'smaller than'
    s = s.replace('p = .000', 'p < .001')
    # Small partial Eta-squared values should be replace with 'smaller than'
    s = s.replace('Partial Eta-Sq. = 0.00', 'Partial Eta-Sq. < .01')
    # Small F values should be replaced with 'smaller than'
    s = s.replace('F = 0.00', 'F < 0.01')
    # Interaction terms should be replaced with 'Interaction of Z X Y'
    s = s.replace(':', ' X ')

    # Variable names of factors should be replaced with prettier versions
    for raw_variable_name, processed_variable_name in cn.VARIABLES_REPRESENTATIVE_NAMES.items():
        s = s.replace(raw_variable_name, processed_variable_name)

    return s

def gen_ttest_report(tests_dict: dict):

    freq = tests_dict['freq']
    bayes = tests_dict['bayes']

    freq_text = freq.report_text(effect_size=True)
    bayes_results = bayes.report_table().T.to_dict()[0]

    bayes_text = (f"BF1:0 = {bayes_results['bf']:.4f} " 
                 f", Error = {bayes_results['error']:.2f}")
    bayes_text = bayes_text.replace("Error = 0.00", "Error <= 0.001%")

    test_text = f"{freq_text}, {bayes_text}"

    return test_text

def gen_rotation_report_table(df):
    df.groupby([cn.COLUMN_NAME_UID])['Delta'].agg(
        ['mean', np.std]).reset_index()


def prettify_descriptives(aggregated_test_data):
    grouper_cols = aggregated_test_data.grouper.names

    desc = aggregated_test_data.apply(
        lambda s: f'{s.mean():.2f} ({np.std(s):.2f})').reset_index()

    if cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE in grouper_cols:
        desc = desc.sort_values(cn.COLUMN_NAME_ABSOLUTE_ROTATION_SIZE, ascending=False)

    desc = desc.rename(columns=cn.VARIABLES_REPRESENTATIVE_NAMES)

    return desc





