import matplotlib.pyplot as plt


def pixel2inch(pixel):
    return pixel / 96


def generate_img_fig(img, fig=None, ax=None):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(pixel2inch(img.shape[1]), pixel2inch(img.shape[0])))

    # Visualize image
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticsk([])
    fig.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    return fig, ax
