import numpy as np
import matplotlib.pyplot as plt


def show_video(video, frames=5):
    i = 0
    fig, axs = plt.subplots(1, frames, figsize=(20, 20))

    for k in np.linspace(0, 64, frames).astype(np.uint8):
        if k >= video.shape[0]:
            k = video.shape[0] - 1
        video_k = np.uint8(((video[k] + 1) * 127.5))
        axs[i].imshow(np.uint8(video_k))
        axs[i].axis('off')
        i += 1


def plot_signal(signal, label):
    fig, axs = plt.subplots(1, 1, figsize=(15, 4))
    signal = np.reshape(signal, (-1,))
    axs.plot(np.linspace(0, signal.shape[0]*0.015,signal.shape[0]), signal, color="red")
    axs.set_title("Action: {}".format(label), fontsize=14)
    axs.set_xlabel('Time (seconds)', fontsize=14)
    axs.set_ylabel('Pulse', fontsize=14)
    axs.set_ylim(top=1.05, bottom=-1.05)
    axs.grid(True)
    plt.show()
