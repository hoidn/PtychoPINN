import matplotlib.pyplot as plt
from ipywidgets import interactive

def ishow_imgs(*arrs_list, styles = None, labels = None,
              log = False, height = '550px',
              nested_label_callback = None):
    """
    Plot a series of curves interactively.
    """
    plt.rcParams["figure.figsize"]=(12, 9)
    #labels = [label1, label2]
    if labels is None:
        labels = [''] * len(arrs_list)
    def f(i):
        for j, patterns in enumerate(arrs_list):
            if styles is not None:
                extra_args = (styles[j],)
            else:
                extra_args = ()
            try:
                for k in range(len(patterns[i])):
                    len(patterns[i][k]) # TODO hack
                    if nested_label_callback is not None:
                        label = nested_label_callback(patterns[i], k)
                    else:
                        label = k
                    plt.imshow(patterns[i][k], *extra_args, label = label)
            except: # TODO except what?
                if j < 2:
                    plt.imshow(patterns[i], label = labels[j])
                else:
                    plt.imshow(patterns[i], *extra_args)

    interactive_plot = interactive(f, i=(0, len(arrs_list[0]) - 1), step = 1)
    output = interactive_plot.children[-1]
    output.layout.height = height
    return interactive_plot
