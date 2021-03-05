from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Cropping2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 150
        self.img_cols = 150
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False    # False !!!!

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #self.combined.compile(loss=['mse', 'binary_crossentropy'],
        #    loss_weights=[0.999, 0.001],
        #    optimizer=optimizer)

    def build_generator(self):
        
        first_layer = (7, 7, 256)

        model = Sequential()

        model.add(Dense(first_layer[0] * first_layer[1] * first_layer[2], activation="relu", input_dim=self.latent_dim))
        model.add(Reshape(first_layer))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3)) # padding="same"
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3)) # padding="same"
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3)) # padding="same"
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3)) # padding="same"
        #model.add(UpSampling2D())
        #model.add(Conv2D(128, kernel_size=3)) # padding="same"
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=5)) # padding="same"
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=5)) # padding="same"
        model.add(Activation("tanh"))
        model.add(Cropping2D(cropping=((3, 3), (3, 3))))

        print("GENERATOR")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_data(self, dataset_path):
        print("Loading dataset from %s..." % dataset_path)
        img_list = os.listdir(dataset_path)
        data = np.zeros((len(img_list), self.img_rows, self.img_cols, self.channels))
        i = 0
        for img_name in img_list:
            #data[i,:,:,:] = cv2.resize(cv2.imread(os.path.join(dataset_path, img_name), cv2.IMREAD_GRAYSCALE), (self.img_rows, self.img_cols), interpolation = cv2.INTER_AREA )[:,:,np.newaxis]
            #data[i,:,:,:] = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(dataset_path, img_name)), (self.img_rows, self.img_cols), interpolation = cv2.INTER_AREA ), cv2.COLOR_BGR2RGB)
            data[i,:,:,:] = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, img_name)), cv2.COLOR_BGR2RGB)
            i += 1
        print("Done.")
        print("Dataset size in memory: %d bytes." % (data.nbytes))
        return data

    def train(self, epochs, batch_size=128, save_interval=50):
        
        dataset_folder = "C:\\Users\\thijs\\Documents\\Studie\\IIa\\Deep Learning\\deep_learning_2_dataset\\dataset\\"
        img_list = os.listdir(dataset_folder)
        n_samples = len(img_list)
        imgs = np.zeros( (batch_size, self.img_rows, self.img_cols, self.channels) )

        '''
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        '''

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, n_samples, batch_size)
            for k, i in enumerate(idx):
                #imgs[k,:,:,:] = cv2.imread(os.path.join(dataset_folder, "%05d.jpg" % i))
                imgs[k,:,:,:] = cv2.cvtColor(cv2.imread(os.path.join(dataset_folder, "%05d.jpg" % i)), cv2.COLOR_BGR2RGB)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%4d [Decoder -- loss: %10.7f, acc.: %6.2f%%] [Generator -- loss: %11.8f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(30,30))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%04d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=5)
