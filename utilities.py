import sklearn
from colorama import Fore,Back,Style

# Get the evaluation metrics of a model given the true labels and predicted
# labels.
#
# Micro takes all TPs, FPs, etc. in the entire model and then solves the metric
# Macro solves the metric for each class, and averages the results
def get_eval_metrics_percent(true_y, predicted_y, average='macro'):
    model_accuracy = sklearn.metrics.accuracy_score(true_y, predicted_y)
    model_precision = sklearn.metrics.precision_score(true_y, predicted_y, average=average)
    model_recall = sklearn.metrics.recall_score(true_y, predicted_y, average=average)
    metrics = {
        'accuracy': model_accuracy*100.0,
        'precision': model_precision*100.0,
        'recall': model_recall*100.0,
    }

    return metrics

# Print evaluation metrics to stdout
def print_metrics(metrics):
    print(Fore.GREEN + Style.BRIGHT + "[+] Statistics: acc=%.2f%%, prec=%.2f%%, rec=%.2f%%" % (metrics['accuracy'], metrics['precision'], metrics['recall']))

    # New for reproducibility: store to file as well
    with open("/tmp/tmp_result", "w") as f:
        f.write("%.2f%% & %.2f%% & %.2f%%\n" % (metrics['accuracy'], metrics['precision'], metrics['recall']))
