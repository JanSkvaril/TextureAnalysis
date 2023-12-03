import matplotlib.pyplot as plt

def ImShow(image, value = None):
    plt.figure(figsize=(5,5))
    plt.imshow(image, cmap='gray')
    if value is not None:
        plt.title(f'Value: {value}')
    plt.axis('off')
    plt.show()


def ImShowArray(images, values=None):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
    for i in range(n):
        axs[i].imshow(images[i], cmap='gray')
        if values is not None:
            axs[i].set_title(f'Value: {values[i]}')
        axs[i].axis('off')
    plt.show()


def ImShowArrayWithHistograms(images, histograms=None):
    n = len(images)
    fig, axs = plt.subplots(2, n, figsize=(5*n, 10))
    for i in range(n):
        axs[0, i].imshow(images[i], cmap='gray')
        if histograms is not None:
            axs[1, i].bar(range(len(histograms[i])), histograms[i])
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.show()

def ImShowArrayWithImages(images, images2=None):
    n = len(images)
    fig, axs = plt.subplots(2, n, figsize=(5*n, 10))
    for i in range(n):
        axs[0, i].imshow(images[i], cmap='gray')
        if images2 is not None:
            axs[1, i].imshow(images2[i], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.show()

def ImShowDistribution(images, values, amount):
    n =  len(images) // amount
    ImShowArray(images[::n], values[::n])