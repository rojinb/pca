import sys
import argparse
import pylab as plt

import utils
from pca import RobustPCA

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True,
                        choices=['1.2', '1.3', '2.1', '3', '3.1', '3.2'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '2.1':
        X = utils.load_dataset('highway')['X'].astype(float)/255
        n, d = X.shape
        print(X.shape)
        h, w = 64, 64  # height and width of each image

        # the two variables below are parameters for the foreground/background extraction method
        # you should just leave these two as default.

        k = 5  # number of PCs
        threshold = 0.04  # a threshold for separating foreground from background

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat = model.expand(Z)

        # save 10 frames for illustration purposes
        for i in range(10):
            plt.subplot(1, 3, 1)
            plt.title('Original')
            plt.imshow(X[i].reshape(h, w).T, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title('Reconstructed')
            plt.imshow(Xhat[i].reshape(h, w).T, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title('Thresholded Difference')
            plt.imshow(1.0*(abs(X[i] - Xhat[i]) < threshold).reshape(h, w).T, cmap='gray')
            utils.savefig('q2_highway_{:03d}.jpg'.format(i))

 