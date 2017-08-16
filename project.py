import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, rescale, iradon, iradon_sart

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def get_random_xy(size=20, keep_threshole=0.99, n_project=180):
    image = np.random.rand(size, size)
    for x in range(size):
        for y in range(size):
            rx = x - size/2
            ry = y - size/2
            r = size/2 -1
            if (rx*rx + ry*ry) > r*r:
                image[x][y] = 0.
            elif image[x][y] > keep_threshole:
                image[x][y] = 1.
            else:
                image[x][y] = 0.

    theta = np.linspace(0., 180., n_project, endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

    fig.tight_layout()
    plt.show()

    reconstruction_fbp = iradon_sart(sinogram, theta=theta)
    error = reconstruction_fbp - image
    print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                   sharex=True, sharey=True,
                                   subplot_kw={'adjustable': 'box-forced'})
    ax1.set_title("Reconstruction\nFiltered back projection")
    ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nFiltered back projection")
    ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    '''

    return iradon_sart(sinogram, theta=theta), image

def get_batch(size=1024):
    X = []
    Y = []
    for _ in range(size):
        recon, image = get_random_xy()
        X.append(recon.flatten())
        Y.append(image.flatten())

    return np.array(X), np.array(Y)

def get_model():
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()
    img_prep.add_samplewise_stdnorm()

    net = tflearn.input_data(shape=[None, 400], data_preprocessing=img_prep)
    net = tflearn.fully_connected(net, 512, activation='relu')
    net = tflearn.fully_connected(net, 512, activation='relu')
    net = tflearn.fully_connected(net, 400, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', loss='binary_crossentropy')
    model = tflearn.DNN(net)

    return net, model

if __name__ == '__main__':
    net, model = get_model()
    train_x, train_y = get_batch(4096)

    model.fit(train_x, train_y, n_epoch=10000, show_metric=True)
