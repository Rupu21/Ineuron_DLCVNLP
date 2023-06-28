import matplotlib.pyplot as plt


def save_plot(df, plotPath):
    df.plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(plotPath)
