import matplotlib.pyplot as plt

def plot_template(title = "",
                 xlabel = "X",
                 ylabel = "Y",
                 equal_axis = True,
                 grid = True,
                 legend = False,
                 save = False,):
    # Labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Options
    if equal_axis:
        plt.axis('equal')
    if grid:
        plt.grid()
    if legend:
        plt.legend(shadow = True)
    if save:
        plt.savefig(title + ".png", dpi = 200)
        plt.clf()
    else:
        plt.show()