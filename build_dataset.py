import numpy as np
import cv2
import glob
import h5py
import matplotlib.pyplot as plt


def dot_h5_make(dot_h_path, img_path):
    # dot_h_path: save location for the .h file
    # img_dir_path: images path for build the dataset

    # read addresses and labels from the 'train' folder
    addrs = glob.glob(img_path)
    labels = [0 if 'bellpepper' in addr else 1 if 'bokchoy' in addr else 2 if 'gherkins' in addr else 3 for addr in
              addrs]  # 0 = bellpepper, 1 = bokchoy, 2 = , 3 = , 4 =

    train_addrs = addrs[0:int(1 * len(addrs))]
    train_y = labels[0:int(1 * len(labels))]

    # crate h5 file with tables
    # data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
    # check the order of data and chose proper data shape to save images
    train_shape = (len(train_addrs), 227, 227, 3)
    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(dot_h_path, mode='w')
    hdf5_file.create_dataset("train_x", train_shape, np.uint8)
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.uint8)
    hdf5_file.create_dataset("train_y", (len(train_addrs),), np.uint8)
    hdf5_file["train_y"][...] = train_y

    # Load images and save them
    # a numpy array to save the mean of the images
    mean = np.zeros(train_shape[1:], np.float32)
    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print
            'Train data: {}/{}'.format(i, len(train_addrs))
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add any image pre-processing here
        # if the data order is Theano, axis orders should change
        """if data_order == 'th':
            img = np.rollaxis(img, 2)"""
        # save the image and calculate the mean so far
        hdf5_file["train_x"][i, ...] = img[None]
        mean += img / float(len(train_y))

    # save the mean and close the hdf5 file
    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    print("dataset created successfully!")


def load_dataset(dot_h_path):
    dataset = h5py.File(dot_h_path, 'r')
    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    print(train_y[3])
    img = train_x[3]
    plt.imshow(img)
    plt.show()
