import hashlib
import json
import os

from mr_utils import preprocessing, reporting, analysis, plot_data, plot_illustration


def main():
    screening_parameters_dicts = load_screening_sets()

    for k, v in screening_parameters_dicts.items():
        figures_dir, texts_dir, data_dir = prep_output_dirs(k, v)

        task, aggregated_task, \
            deltas, quantities_to_report, vviq, debrief, demographics = (
            preprocessing.main(data_dir, v))

        results_dict, sequential_bf_results = analysis.main(task, deltas)

        reporting.main(texts_dir, demographics, quantities_to_report,
                       results_dict,
                       v)

        plot_data.main(figures_dir, aggregated_task, deltas, sequential_bf_results, task)

        plot_illustration.main('Output/illustration')


def prep_output_dirs(param_set_name, param_set):
    """
    Prepares output directories for the given screening set and saves the
    set of screening parameters inside the directory.

    Parameters
    ----------
    param_set_name: str
        The name of the set of screening parameters. Doesn't have to be unique.
    param_set: dictionary
        A dictionary containing the parameters for the processing of the data.

    Returns
    -------
    texts_dir, figures_dir, data_dir: str
        The paths to the output directories for the given screening parameters
        set.
    """

    screening_set_json = json.dumps(
        param_set, sort_keys=True).encode('utf-8')
    hashed_screening_set = hashlib.md5(screening_set_json).hexdigest()

    output_dir = f'Output/Analyses/{hashed_screening_set}'
    output_subdirs = [f'{output_dir}/{d}'
                      for d in ['Figures', 'Texts', 'Data']]
    for _dir in output_subdirs:
        os.makedirs(_dir, exist_ok=True)

    with open(f'{output_dir}/Screening parameters - {param_set_name}.json', 'w',
              encoding='utf-8') as f:
        json.dump(param_set, f, ensure_ascii=False)

    texts_dir, figures_dir, data_dir = output_subdirs
    return texts_dir, figures_dir, data_dir


def load_screening_sets():
    with open('screening_param_sets.json') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    main()
