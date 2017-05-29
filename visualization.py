# Visualizations for debugging and Tensorboard
import matplotlib
import socket
if socket.gethostname() != 'arch':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf
from matplotlib.patches import Circle
from matplotlib.patches import Patch

# Keep colors consistent
class_colors = [None, '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#000000', '#80ff80', '#b0bc32', '#d65111', '#615562', '#ef8bd4', '#83bc8c', '#726800', '#40d93e', '#54692c', '#6fd4f1', '#e2d978', '#ff8000', '#1dcceb', '#7a58f7', '#1aaa91', '#ba60b0', '#76191f']
class_labels = [None, 'LoRa 1 ', 'LoRa 2 ', 'LoRa 3 ', 'LoRa 4 ', 'LoRa 5 ', 'LoRa 6 ', 'LoRa 7 ', 'LoRa 8 ', 'LoRa 9 ', 'LoRa 10', 'LoRa 11', 'LoRa 12', 'LoRa 13', 'LoRa 14', 'LoRa 15', 'LoRa 16', 'Lora 17', 'LoRa 18', 'LoRa 19', 'LoRa 20', 'LoRa 21', 'LoRa 22', 'LoRa 23', 'LoRa 24']

def dbg_plot(y, title=''):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    ax.plot(np.arange(len(y)), y)
    ax.set_xlim([0, len(y)])
    ax.set_xlabel("samples")
    plt.show()

def dbg_plot_complex(y, title=''):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    ax.plot(np.arange(len(y)), np.real(y), "b", np.arange(len(y)), np.imag(y), "g")
    ax.set_xlim([0, len(y)])
    ax.set_xlabel("samples")
    plt.show()

# Convert matplotlib plot to tensorboard image
def _plt_to_tf(plot, tag):
    # Write to PNG buffer
    buf = io.BytesIO()
    plot.savefig(buf, format='png')
    plot.savefig("/tmp/tf_" + tag + ".pdf", format='pdf')
    buf.seek(0)

    # Add to TensorBoard summary
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0) # Add the batch dimension
    return tf.summary.image(tag, image, 1)

# See https://stackoverflow.com/questions/38543850/tensorflow-how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots
def plot_values(values, instances_mapping, height=800, width=800, tag="", title="", label=None, backdrop=None):
    # Configure figure
    dpi = 96
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.title(title)
    plot_color = 'gray'

    if not label is None: # Show plot in color of the label according to class_colors
        title += " (LoRa " + str(instances_mapping.map_to_lora_id(label)) + ")"
        plt.title(title)
        plot_color = class_colors[instances_mapping.map_to_lora_id(label)]

    # Plot main values
    if backdrop is None:
        xvalues = range(0, len(values))
        values_normed = (values - values.min(0)) / values.ptp(0)
        plt.plot(xvalues, values_normed, plot_color, alpha=0.7)

    # Plot weights backdrop?
    else:
        num_classes = backdrop.shape[1]

        props = dict(alpha=0.5, edgecolors='none')
        for i in range(0, num_classes):
            class_backdrop = backdrop[0:,i]
            class_backdrop_normed = (class_backdrop - class_backdrop.min(0)) / class_backdrop.ptp(0)
            xvalues = range(0, len(class_backdrop_normed))

            # Get correct color for backdrop
            color = class_colors[instances_mapping.map_to_lora_id(i)]

            plt.scatter(xvalues, class_backdrop_normed, c=color, **props)

    # Organize plot tightly
    plt.tight_layout()

    # Fix axis ranges
    ax = plt.gca()
    ax.set_xlim([0, len(xvalues)])

    return _plt_to_tf(plt, tag)

def plot_kernels(kernels, kernel_size, height, width, tag="", title=""):
    dpi = 96
    cols = 2  # TODO: Make user definable
    rows = len(kernels)/cols  # Should be round number
    plt.title(title)
    fig, axes = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(width/dpi, height/dpi), dpi=dpi)

    kernel_idx = 0
    for axis_rows in axes:
        for axis_col in axis_rows:
            kernel = kernels[kernel_idx]
            # Line plot
            #axis_col.plot(range(0, len(kernel)), kernel)
            #axis_col.set_xlim([0, len(kernel)])

            # Image
            kernel_image = kernel.reshape((1, len(kernel)))
            axis_col.imshow(kernel_image, extent=(0, width, 0, 64), interpolation='nearest', cmap=plt.get_cmap('Blues'))
            kernel_idx += 1

    plt.tight_layout()

    return _plt_to_tf(plt, tag)


def plot_weights(weights, real_labels, predictions, expected_values, thresholds, instances_mapping, height=600, width=800, tag="", title="", xlabel="Class A", ylabel="Class B", metrics=None, equal_aspect=False, tf=True):
    # Configure figure TODO duplicate code fix me
    dpi = 96
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.title(title)

    # Plot output weights
    num_points = len(weights)
    num_real_labels = len(real_labels)
    num_predictions = len(predictions)

    if num_points != num_real_labels != predictions:
        print("[-] Number of points != number of real_labels. That's not good. plot_output_weights exiting.")
        exit(1)

    if weights.shape[1] != 2:
        print("[-] Can't plot other-than 2D data, continuing without plot.")
        return

    # Draw weights
    real_props = dict(alpha=0.50, edgecolors='none')
    predicted_props = dict(alpha=1.00, facecolors='none')
    adversary_props = dict(alpha=0.50, facecolors='r', marker="x")
    for i in range(0, num_points):
        point = weights[i]
        real_lora_id = real_labels[i]
        predicted_lora_id = predictions[i]

        real_point_color = class_colors[real_lora_id]
        plt.scatter(point[0], point[1], c=real_point_color, **real_props)

        if predicted_lora_id == -1:
            plt.scatter(point[0], point[1], **adversary_props)
        else:
            predicted_point_color = class_colors[predicted_lora_id]
            plt.scatter(point[0], point[1], edgecolors=predicted_point_color, **predicted_props)

    # Draw expected values
    # TODO: temp disabled until I figure out what to do with this
    """
    for i in range(0, len(expected_values)):
        circle_x = expected_values[i][0]
        circle_y = expected_values[i][1]
        circle = Circle((circle_x, circle_y), thresholds[i], edgecolor=class_colors[instances_mapping.map_to_lora_id(i)], facecolor='none', linewidth=2, alpha=0.5)
        plt.gca().add_patch(circle)
    """

    # Draw legend
    patches = []
    tmp = []
    for rlabel in sorted(real_labels):
        lora_id = rlabel
        real_color = class_colors[lora_id]
        real_label = class_labels[lora_id]
        if not (real_label in tmp):
            patches.append(Patch(color=real_color, label=real_label))
            tmp.append(real_label)
    plt.legend(loc='upper right', ncol=4, fancybox=True, shadow=True, fontsize=8, handles=patches)

    # Draw metrics on the pdf
    if not metrics is None:
        ax = plt.gca()
        metrics_text = 'accuracy: %.2f%%\nprecision: %.2f%%\nrecall: %.2f%%' % (metrics['accuracy'], metrics['precision'], metrics['recall'])
        ax.text(0.01, 0.80, metrics_text,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)

    # Set labels
    #plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Fix axis aspect ratio
    if equal_aspect:
        plt.axes().set_aspect('equal', 'datalim')

    if tf:
        return _plt_to_tf(plt, tag)
    else:
        plt.show()
