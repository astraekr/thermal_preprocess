import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile


class PreProcess:
    """
        Pre-process object for organising and preparing thermal image sets
        I made this into a class partly as an exercise (I'm rusty) but also
        as I thought it made sense, as a pre-processing instance, with a set
        parent/working directory could be played with in the interactive
        interpreter.

        :param working_dir: the top directory of the data-set being used
        :type working_dir: str
        :param `*args`: the variable arguments are used for ...
        :param `**kwargs`: the keyword arguments are used for ...
        :ivar parent_folder: this is where we store workingDir
        :type parent_folder: str
    """

    def __init__(self, working_dir):
        self.parent_folder = working_dir
        self.folder_list = self.get_list_of_folders('*rotated90')

    def get_list_of_folders(self, end_of_folder_name):
        """Looks in the top directory and returns a list of folders

        :param end_of_folder_name: string used to determine folders to return
        :type end_of_folder_name: str
        :returns: folder_list list of the folders
        :rtype: list
        """
        folder_list = [os.path.basename(f) for f in glob.glob(os.path.join(self.parent_folder, end_of_folder_name))]
        folder_list.sort()
        return folder_list

    @staticmethod
    def get_photo_list(folder_name, extension='*.png'):
        """Gets a list of photos of a given type from a given folder

        :param folder_name: Folder to list photos from
        :param extension: The file type that should be listed, eg '*.png'
        :type folder_name: str
        :type extension: str
        :returns: List of photos
        :rtype: list
        """
        photo_list = [os.path.basename(f) for f in glob.glob(os.path.join(folder_name, extension))]
        photo_list.sort()
        return photo_list

    def stitch_images(self, photo_list):
        """Stitches images from 8 folders into a 4*2 grid

        :param photo_list:
        :type photo_list: list
        :return: name of the stitched folder
        :rtype: str
        """
        stitched_folder_name = self.parent_folder + '/stitched'
        print("Stitching images in:")
        print(self.folder_list)
        print("Storing in: " + str(stitched_folder_name))

        # check if the folder exists
        if os.path.exists(stitched_folder_name):
            print("Folder exists, have you already done this stitching??")
            exit()
        else:
            print("Making dir " + str(stitched_folder_name) + " for stitching")
            os.mkdir(stitched_folder_name)

        # get photo sizes
        size_photo = cv2.imread(self.parent_folder + '/' + self.folder_list[0] +
                                '/' + photo_list[0], cv2.IMREAD_ANYDEPTH)
        photo_height, photo_width = np.shape(size_photo)
        stitched_height = photo_height * 2
        stitched_width = photo_width * 4

        for photo in photo_list:
            stitched_photo = np.full((stitched_height, stitched_width), 0)

            for i, folder in enumerate(self.folder_list):
                print i
                print folder
                print(self.parent_folder + '/' + folder + photo)

                stitched_photo[(int((float(i) / 4.0)) * photo_height):(int(((float(i) / 4.0) + 1)) * photo_height),
                               (int(i % 4) * photo_width):((int((i % 4) + 1)) * photo_width)] \
                    = cv2.imread(self.parent_folder + '/' + folder + '/' + photo, cv2.IMREAD_ANYDEPTH)

            stitched_photo = stitched_photo.astype(np.uint16)
            cv2.imwrite(stitched_folder_name + '/' + photo, stitched_photo, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return stitched_folder_name

    def rotate_image_set(self, folder_name, num_rotations=3):
        """Rotates images in a folder

        :param folder_name:
        :param num_rotations:
        :type folder_name:
        :type num_rotations:
        :return:
        """
        print("Rotating images in folder: "+str(folder_name))
        photo_list = self.get_photo_list(folder_name, '*.png')
        destination_folder = folder_name+"_rotated90"

        if os.path.exists(destination_folder):
            print("Folder exists, have you already done this rotation??")
            exit()
        else:
            print("Making dir "+str(destination_folder)+" for rotation")
            os.mkdir(destination_folder)

        for i, name in enumerate(photo_list):
            print(name)
            file_name = folder_name + '/' + name
            destination_filename = destination_folder + '/' + name[:-4] + '.png'
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            cv2.imwrite(destination_filename, np.rot90(image, num_rotations), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    @staticmethod
    def normalise_image(file_name, destination):
        """Reads image, normalises it and saves it withe new name

        :param file_name: full path to the file to normalise
        :param destination: full path to destination folder
        :type file_name: str
        """
        normalised_image_name = destination + '/' + file_name[:-4] + 'normed.png'
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        new_image = image.astype(np.uint8)
        new_image = cv2.equalizeHist(new_image)
        cv2.imwrite(normalised_image_name, new_image)

    @staticmethod
    def normalise_image_16bit(file_name, max_possible_intensity=65535):
        """Normalise single image, retaining 16bit depth

        :param file_name: full path to the file to normalise
        :param max_possible_intensity: the maximum intensity for the output image
        :type file_name: str
        :type max_possible_intensity: int
        """
        normalised_image_name = file_name[:-4] + 'normed16bit.png'
        image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)

        minimum_intensity = np.amin(image)
        maximum_intensity = np.amax(image)
        factor = float(max_possible_intensity) / float(maximum_intensity - minimum_intensity)

        subtracted_image = np.subtract(image, np.full(np.shape(image), minimum_intensity))
        subtracted_scaled_image = (subtracted_image * factor)
        normalised_image = subtracted_scaled_image.astype(np.uint16)
        cv2.imwrite(normalised_image_name, normalised_image)

    def normalise_images(self, folder_name, max_possible_intensity=65535):
        """ Normalise a set of images to the bounds of the full set

        :param folder_name: name of the folder to be normalised
        :param max_possible_intensity: the maximum intensity for the output image
        :type folder_name: str
        :type max_possible_intensity: int
        """
        normalised_folder_name = folder_name + '_normalised'

        # check if the folder exists
        if os.path.exists(normalised_folder_name):
            print("Folder exists, have you already done this normalisation??")
            exit()
        else:
            print("Making dir "+str(normalised_folder_name)+" for normalisation")
            os.mkdir(normalised_folder_name)

        minimum_intensity, maximum_intensity = self.get_min_max_values(folder_name)
        intensity_scaling_factor = float(max_possible_intensity) / float(maximum_intensity - minimum_intensity)

        print("Writing to folder +" + str(normalised_folder_name))
        photo_list = self.get_photo_list(folder_name, '*.png')
        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            normalised_image_name = normalised_folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            subtracted_image = np.subtract(image, np.full(np.shape(image), minimum_intensity))
            scaled_image = (subtracted_image * intensity_scaling_factor)
            normalised_image = scaled_image.astype(np.uint16)
            cv2.imwrite(normalised_image_name, normalised_image)

    def get_min_max_values(self, folder_name):
        """get minimum and maximum pixel intensities for a folder

        :param folder_name: the folder in which to find the minimum and maximum
        :return:  minimum and maximum pixel intensity values for the folder
        :rtype int, int
        """
        minimim_pixel_values = []
        maximum_pixel_values = []
        print("Getting min/max vals in folder " + str(folder_name))
        photo_list = self.get_photo_list(folder_name)
        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

            minimim_pixel_values.append(np.amin(image))
            maximum_pixel_values.append(np.amax(image))

        folder_minimum_pixel_value = np.amin(minimim_pixel_values)
        folder_maximum_pixel_value = np.amin(maximum_pixel_values)

        print("This min is: " + str(folder_minimum_pixel_value) + "  This max is: " + str(folder_maximum_pixel_value))

        return folder_minimum_pixel_value, folder_maximum_pixel_value

    def normalise_per_image(self, folder_name):
        """

        :param folder_name: folder to have images normalised in
        :type folder_name: str
        """
        normalised_folder_name = folder_name + '_normalisedPerImage'

        # check if the folder exists
        if os.path.exists(normalised_folder_name):
            print("Folder exists, have you already done this normalisation??")
            exit()
        else:
            print("Making dir "+str(normalised_folder_name)+" for normalisation")
            os.mkdir(normalised_folder_name)

        print("Writing to folder +" + str(normalised_folder_name))
        photo_list = self.get_photo_list(folder_name)
        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            self.normalise_image(file_name, normalised_folder_name)

    def mask_images(self, folder_name, mask_image_name):
        """ Masks images. Needs mask where background = 0. Writes to new folder.

        :param folder_name: folder to be masked
        :param mask_image_name: full path to mask image
        :type folder_name: str
        :type mask_image_name: str
        """

        photo_list = self.get_photo_list(folder_name)
        masked_folder_name = folder_name + '_masked'

        # check if the folder exists
        if os.path.exists(masked_folder_name):
            print("Folder exists, have you already done this masking??")
            exit()
        else:
            print("Making dir " + str(masked_folder_name) + " for masking")
            os.mkdir(masked_folder_name)

        full_mask_image = cv2.imread(mask_image_name, cv2.IMREAD_ANYDEPTH)

        for i, image_name in enumerate(photo_list):
            print("0," + str(i))
            print (folder_name + '/' + image_name)
            img = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
            masked_image = img

            size = img.shape
            for rowPixel in range(0, size[0]):
                for columnPixel in range(0, size[1]):
                    if full_mask_image[rowPixel, columnPixel] != 0:
                        masked_image[rowPixel, columnPixel] = img[rowPixel, columnPixel]

                    else:
                        masked_image[rowPixel, columnPixel] = 0

            cv2.imwrite(masked_folder_name + '/' + image_name, masked_image.astype(np.uint16))

    @staticmethod
    def invert_image(folder_name, image_name, bit_depth=65535):
        """inverts an image, by subtracting it from bit depth (16 bit by default)

        :param folder_name: folder containing target image
        :param image_name: image to be inverted
        :param bit_depth: resolution of image
        :type folder_name: str
        :type image_name: str
        """
        image = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
        inverted_image = np.subtract(np.full(np.shape(image), bit_depth), image)
        cv2.imwrite(folder_name + '/' + image_name[:-4]+'inverted.png', inverted_image.astype(np.uint16))

    @staticmethod
    def get_histogram(folder_name, image_name, save_location):
        """Uses pyplot hist method to get and save the histogram for an image

        :param folder_name: folder image is contained in
        :param image_name: image to histogram of
        :param save_location: where to save the histogram to
        :type folder_name: str
        :type image_name: str
        :type save_location: str
        """
        print("Getting histogram for:" + str(folder_name) + '/' + str(image_name))
        image = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
        plt.hist(image.ravel(), 256, [0, 65535])
        plt.savefig(save_location + 'histogram.eps', format='eps')
        plt.show()

    @staticmethod
    def create_binary_masks(image_path):
        """ inverts mask and ensures it is binary

        :param image_path: full path to the mask image
        :return:
        """
        mask = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        size = mask.shape
        for row_pixel in range(0, size[0]):
            for column_pixel in range(0, size[1]):
                if mask[row_pixel, column_pixel] == 0:
                    mask[row_pixel, column_pixel] = 65535

                else:
                    mask[row_pixel, column_pixel] = 0

        cv2.imwrite(image_path[:-4]+'_binary.png', mask)

    def colourise_image(self, folder_name):
        """ adjusts images from greyscale to opencv jet

        :param folder_name: folder of images to be adjusted
        :type folder_name: str
        """
        colourised_folder_name = folder_name + '_colourised'

        # check if the folder exists
        if os.path.exists(colourised_folder_name):
            print("Folder exists, have you already done this colourisation??")
            exit()
        else:
            print("Making dir "+str(colourised_folder_name)+" for colourisation")
            os.mkdir(colourised_folder_name)

        print("Writing to folder +" + str(colourised_folder_name))
        photo_list = self.get_photo_list(folder_name)
        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            colourised_image_name = colourised_folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            image_8bit = image.astype(np.uint8)
            colour_image = cv2.applyColorMap(image_8bit, cv2.COLORMAP_JET)
            cv2.imwrite(colourised_image_name, colour_image)

    def subsample_imageset(self, source_folder_name, destination_folder_name, sample_step=4):
        """Copies 1 in every 'sample_step' images to a new folder. Generally for display purposes.

        :param source_folder_name: folder for images to be sub-sampled
        :param destination_folder_name: destination folder
        :param sample_step: size of step between images for sampling
        """
        photo_list = self.get_photo_list(source_folder_name)
        for i in range(0, len(photo_list), sample_step):
            copyfile(source_folder_name + '/' + photo_list[i], destination_folder_name + '/' + photo_list[i])
