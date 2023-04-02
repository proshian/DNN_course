from typing import List
import matplotlib.pyplot as plt


def get_phases_and_metric_names_from_hirtory_list(history_list: List[dict[str, List[float]]]) -> tuple[List[str], List[str]]:
    history1 = history_list[0]
    phases = list(history1.keys())
    metric_names = list(history1[phases[0]].keys())
    return phases, metric_names


def get_epoches_num_from_history(history: dict[str, List[float]], phases: List[str], metric_names: List[str]) -> int:
    return len(history[phases[0]][metric_names[0]])


def plot_history(history_list: List[dict[str, List[float]]],
                 history_names: List[str] = None,
                 omit_first_epoch: bool = False,
                 force_legend: bool = False,
                 img_name: str = None) -> None:
    """
    Plots histories on same plot

    history_list is a list of histories. A history is a dict with phase
    names as keys and a an inner dict as values. The inner dict has metric
    names as keys and list of metric values as values.
    Here is an example of a history:
        {
            'train': {
                'accuracy': [0.1, 0.2, 0.3, ...],
                'loss': [0.1, 0.2, 0.3, ...],
                'f1_score': [0.1, 0.2, 0.3, ...],
            },
            'test': {
                'accuracy': [0.1, 0.2, 0.3, ...],
                'loss': [0.1, 0.2, 0.3, ...],
                'f1_score': [0.1, 0.2, 0.3, ...],
            }
        }
    
    The resulting plot is a grid of suplots with (phases number) rows
    and (metric names number) columns. Each subplot contains metric values for all histories.

    The legend is present if len(history_list) > 1 or force_legend is True. If history_names is not None,
    it is used as legend labels. Otherwise, history_list indexes are used as legend labels.

    Args:
        history_list (List[dict[str, List[float]]]): list of histories.
        history_names (List[str], optional):
            list of history names. Defaults to None.
        omit_first_epoch (bool, optional):
            if True, first epoch will be omitted. Defaults to False.
        force_legend (bool, optional): if True, legend will be
            present even if len(history_list) == 1. Defaults to False.
        img_name (str, optional): if not None, the plot will be
            saved to img_name. Defaults to None.
    """

    assert len(history_list) > 0, "history_list is empty"

    phases, metric_names = get_phases_and_metric_names_from_hirtory_list(history_list)

    if history_names is None:
        history_list_indexes = range(len(history_list))
        history_names = history_list_indexes
    else:    
        assert len(history_list) == len(history_names), "len(history_list) != len(history_names)"

    max_epochs_num = max([get_epoches_num_from_history(history, phases, metric_names) for history in history_list])
    
    fig, axs = plt.subplots(len(phases), len(metric_names))

    fig.set_figheight(10)
    fig.set_figwidth(20)

    start = 1 if omit_first_epoch else 0

    for phase_index, phase in enumerate(phases):
        for metric_index, metric_name in enumerate(metric_names):
            ax = axs[phase_index][metric_index]
            ax.set_title(f"{phase} {metric_name}")
            ax.set_xticks(range(max_epochs_num - start))
            ax.set_xticklabels(range(start + 1, max_epochs_num + 1))

            for history, name in zip(history_list, history_names):
                ax.plot(history[phase][metric_name][start:], label = name)
            
            if force_legend or len(history_list) > 1:
                ax.legend()
    
    if img_name is not None:
        plt.savefig(img_name)
    plt.show()