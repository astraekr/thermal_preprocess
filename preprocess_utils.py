import os
import glob
import numpy as np
import cv2
from shutil import copyfile
import pandas as pd
import datetime
import scipy.signal as sgnl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as dates


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
        self.full_folder_list = self.get_parent_subfolders()
        self.folder_basename = os.path.basename(os.path.normpath(working_dir))
        # TODO add photo_list to this - is basically used everywhere
        # TODO add photo sizes to this
        # TODO change the path handling to be platform-agnostic
        print('Working in: ' + working_dir)
        print('Folders are: ')
        for i in range(0, len(self.full_folder_list)):
            print(self.full_folder_list[i])

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

    def get_parent_subfolders(self):
        """Prints all directories in the chosen 'working/parent dir', just because

        :return: subfolders
        """
        return [x[0] for x in os.walk(self.parent_folder)]

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

    def stitch_images(self):
        """Stitches images from 8 folders into a 4*2 grid

        :return: name of the stitched folder
        :rtype: str
        """
        stitched_folder_name = self.parent_folder + '/stitched'
        print("Stitching images in:")
        print(self.folder_list)
        print("Storing in: " + str(stitched_folder_name))

        try:
            print("Making dir " + str(stitched_folder_name) + " for stitching")
            os.mkdir(stitched_folder_name)
        except OSError:
            print("Folder exists, have you already done this stitching??")
            return

        photo_list = self.get_photo_list(self.parent_folder + '/' + self.folder_list[0])
        # get photo sizes
        print(self.parent_folder + '/' + self.folder_list[0] + '/' + photo_list[0])
        size_photo = cv2.imread(self.parent_folder + '/' + self.folder_list[0] +
                                '/' + photo_list[0], cv2.IMREAD_ANYDEPTH)
        photo_height, photo_width = np.shape(size_photo)
        stitched_height = photo_height * 2
        stitched_width = photo_width * 4

        for photo in photo_list:
            stitched_photo = np.full((stitched_height, stitched_width), 0)

            for i, folder in enumerate(self.folder_list):
                print(i)
                print(folder)
                print(self.parent_folder + '/' + folder + photo)

                stitched_photo[(int((float(i) / 4.0)) * photo_height):(int(((float(i) / 4.0) + 1)) * photo_height),
                               (int(i % 4) * photo_width):((int((i % 4) + 1)) * photo_width)] \
                    = cv2.imread(self.parent_folder + '/' + folder + '/' + photo, cv2.IMREAD_ANYDEPTH)

            stitched_photo = stitched_photo.astype(np.uint16)
            cv2.imwrite(stitched_folder_name + '/' + photo, stitched_photo, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return stitched_folder_name

    def unstitch_image(self, file_path):
        """splits a stitched image back into the 8 constituent parts

        :param file_path: full path to the stitched image to be unstitched
        :return:
        """
        new_height = 320
        new_width = 256
        stitched_image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        stitched_image_one = stitched_image[0:new_height, :new_width]
        stitched_image_two = stitched_image[0:new_height, new_width:new_width * 2]
        stitched_image_three = stitched_image[0:new_height, new_width * 2:new_width * 3]
        stitched_image_four = stitched_image[0:new_height, new_width * 3:new_width * 4]

        stitched_image_five = stitched_image[new_height:new_height * 2, :new_width]
        stitched_image_six = stitched_image[new_height:new_height * 2, new_width:new_width * 2]
        stitched_image_seven = stitched_image[new_height:new_height * 2, new_width * 2:new_width * 3]
        stitched_image_eight = stitched_image[new_height:new_height * 2, new_width * 3:new_width * 4]

        cv2.imwrite(file_path[:-4]+ 'one.png', stitched_image_one)
        cv2.imwrite(file_path[:-4] + 'two.png', stitched_image_two)
        cv2.imwrite(file_path[:-4] + 'three.png', stitched_image_three)
        cv2.imwrite(file_path[:-4] + 'four.png', stitched_image_four)

        cv2.imwrite(file_path[:-4] + 'five.png', stitched_image_five)
        cv2.imwrite(file_path[:-4] + 'six.png', stitched_image_six)
        cv2.imwrite(file_path[:-4] + 'seven.png', stitched_image_seven)
        cv2.imwrite(file_path[:-4] + 'eight.png', stitched_image_eight)

        print('DONE')

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

        try:
            print("Making dir " + str(destination_folder) + " for rotation")
            os.mkdir(destination_folder)
        except OSError:
            print("Folder exists, have you already done this rotation??")
            return

        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            destination_filename = destination_folder + '/' + name[:-4] + '.png'
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            cv2.imwrite(destination_filename, np.rot90(image, num_rotations), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def rotate_image(self, file_name, num_rotations=3):
        """rotates an individual image and pops it in the parent folder

        :param file_name: full path to file name
        :return:
        """
        image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        destination_filename = self.parent_folder + os.path.basename(file_name)
        cv2.imwrite(destination_filename, np.rot90(image, num_rotations), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    @staticmethod
    def normalise_image(file_name, source_folder_name, destination_folder_name):
        """Reads image, normalises it and saves it withe new name

        :param file_name: full path to the file to normalise
        :param destination: full path to destination folder
        :type file_name: str
        """
        normalised_image_name = destination_folder_name + '/' + file_name
        image = cv2.imread(source_folder_name + '/' + file_name, cv2.IMREAD_GRAYSCALE)
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

    def normalise_images(self, folder_name, max_possible_intensity=65535.0):
        """ Normalise a set of images to the bounds of the full set


        :param folder_name: full path name of the folder to be normalised
        :param max_possible_intensity: the maximum intensity for the output image
        :type folder_name: str
        :type max_possible_intensity: int
        """
        normalised_folder_name = folder_name + '_normalised'

        try:
            print("Making dir " + str(normalised_folder_name) + " for normalisation")
            os.mkdir(normalised_folder_name)
        except OSError:
            print("Folder exists, have you already done this normalisation??")
            return

        minimum_intensity, maximum_intensity = self.get_min_max_values(folder_name)
        intensity_scaling_factor = max_possible_intensity / float(maximum_intensity - minimum_intensity)
        print("Factor = " + str(intensity_scaling_factor))

        print("Writing to folder: " + str(normalised_folder_name))
        photo_list = self.get_photo_list(folder_name, '*.png')
        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            normalised_image_name = normalised_folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            subtracted_image = np.maximum(np.subtract(image.astype(np.int32), np.full(np.shape(image),
                                                      minimum_intensity.astype(np.int32))),
                                          np.zeros(np.shape(image)))

            normalised_image = (subtracted_image * intensity_scaling_factor)
            cv2.imwrite(normalised_image_name, normalised_image.astype(np.uint16))

    def get_min_max_values(self, folder_name, non_zero=True):
        """get minimum and maximum pixel intensities for a folder

        :param folder_name: the folder in which to find the minimum and maximum
        :type folder_name: str
        :param non_zero: if true, the method will return minimum non-zero pixel, useful for masked images
        :type non_zero: bool
        :return:  minimum and maximum pixel intensity values for the folder
        :rtype int, int
        """
        minimum_pixel_values = []
        maximum_pixel_values = []
        print("Getting min/max vals in folder " + str(folder_name))
        photo_list = self.get_photo_list(folder_name)
        for i, name in enumerate(photo_list):
            file_name = folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            if non_zero:
                image = image.ravel()
                image.sort()
                non_zero_pixels = image[np.nonzero(image)]
                minimum_pixel_values.append(non_zero_pixels[0])

            else:
                minimum_pixel_values.append(np.amin(image))

            maximum_pixel_values.append(np.amax(image))

        folder_minimum_pixel_value = np.amin(minimum_pixel_values)
        folder_maximum_pixel_value = np.amax(maximum_pixel_values)

        print("The min is: " + str(folder_minimum_pixel_value) + "  The max is: " + str(folder_maximum_pixel_value))
        return folder_minimum_pixel_value, folder_maximum_pixel_value

    def normalise_per_image(self, folder_name):
        """

        :param folder_name: folder to have images normalised in
        :type folder_name: str
        """
        normalised_folder_name = folder_name + '_normalisedPerImage'

        try:
            print("Making dir " + str(normalised_folder_name) + " for normalisation")
            os.mkdir(normalised_folder_name)
        except OSError:
            print("Folder exists, have you already done this normalisation??")
            return

        print("Writing to folder +" + str(normalised_folder_name))
        photo_list = self.get_photo_list(folder_name)
        for i, name in enumerate(photo_list):
            file_name = name
            self.normalise_image(file_name, folder_name, normalised_folder_name)

    def mask_images(self, folder_name, mask_image_name):
        """ Masks images. Needs mask where background = 0. Writes to new folder.
            This method is very slow, but had trouble using opencv bitwise AND

        :param folder_name: folder to be masked
        :param mask_image_name: full path to mask image
        :type folder_name: str
        :type mask_image_name: str
        """

        photo_list = self.get_photo_list(folder_name)
        masked_folder_name = folder_name + '_background'

        try:
            print("Making dir " + str(masked_folder_name) + " for masking")
            os.mkdir(masked_folder_name)
        except OSError:
            print("Folder exists, have you already done this masking??")
            return

        full_mask_image = cv2.imread(mask_image_name, cv2.IMREAD_ANYDEPTH)

        for i, image_name in enumerate(photo_list):
            print(i)
            print (folder_name + image_name)
            img = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
            masked_image = img

            size = img.shape
            for row_pixel in range(0, size[0]):
                for column_pixel in range(0, size[1]):
                    if full_mask_image[row_pixel, column_pixel] != 0:
                        masked_image[row_pixel, column_pixel] = img[row_pixel, column_pixel]

                    else:
                        masked_image[row_pixel, column_pixel] = 0

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
        # plt.show()

    def get_histograms(self, folder_name):
        """A more automated version of the above method, for mass histogram generation

        :param folder_name: folder image is contained in
        :type folder_name: str

        """
        histograms_folder_name = folder_name + '_histograms'

        try:
            print("Making dir " + str(histograms_folder_name) + " for histograms")
            os.mkdir(histograms_folder_name)
        except OSError:
            print("Folder exists, have you already created these/this??")
            return

        print("Writing to folder: " + str(histograms_folder_name))
        photo_list = self.get_photo_list(folder_name, '*.png')
        for name in photo_list:
            image = cv2.imread(folder_name + '/' + name, cv2.IMREAD_ANYDEPTH)
            plt.hist(image.ravel(), 256, [0, 65535])
            plt.savefig(histograms_folder_name + '/' + name + 'histogram.eps', format='eps')
            plt.clf()
            # plt.show()

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

        try:
            print("Making dir " + str(colourised_folder_name) + " for colourisation")
            os.mkdir(colourised_folder_name)
        except OSError:
            print("Folder exists, have you already done this colourisation??")
            return

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

    def unmask_images(self, folder_name, background_image_name, mask_image_name):
        """For presentation. Uses background imagery from a full image rather than a flat black mask.
        Results aren't particularly nice to look at unfortunately.

        :param folder_name: path to folder to be unmasked
        :param background_image_name: path to image to be used as constant background for full image set
        :param mask_image_name: path to the image originally used to mask the dataset
        :type folder_name: str
        :type background_image_name: str
        :type mask_image_name: str
        """
        # TODO add functionality to unmask the correct background for each image

        photo_list = self.get_photo_list(folder_name)
        unmasked_folder_name = folder_name + '_unmasked'

        try:
            print("Making dir " + str(unmasked_folder_name) + " for unmasking")
            os.mkdir(unmasked_folder_name)
        except OSError:
            print("Folder exists, have you already done this unmasking??")
            return

        full_unmask_image = cv2.imread(background_image_name, cv2.IMREAD_ANYDEPTH)
        full_mask_image = cv2.imread(mask_image_name, cv2.IMREAD_ANYDEPTH)

        for i, image_name in enumerate(photo_list):
            print("0," + str(i))
            print (folder_name + '/' + image_name)
            img = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
            unmasked_image = img

            size = img.shape
            for rowPixel in range(0, size[0]):
                for columnPixel in range(0, size[1]):
                    if full_mask_image[rowPixel, columnPixel] != 0:
                        unmasked_image[rowPixel, columnPixel] = img[rowPixel, columnPixel]

                    elif full_mask_image[rowPixel, columnPixel] == 0:
                        unmasked_image[rowPixel, columnPixel] = full_unmask_image[rowPixel, columnPixel]

            cv2.imwrite(unmasked_folder_name + '/' + image_name, unmasked_image.astype(np.uint16))

    def save_imageset_histogram(self, folder_name, save_location):
        """ This doesn't work for data-sets like the one I'm using

        :param folder_name:
        :param save_location:
        :type folder_name: str
        :type save_location: str
        """
        all_images = []
        photo_list = self.get_photo_list(folder_name)
        for name in photo_list:
            image = cv2.imread(folder_name + '/' + name, cv2.IMREAD_ANYDEPTH)
            all_images.append(image.ravel())

        plt.hist(all_images, 256, [0, 65535])
        plt.savefig(save_location + 'histogram.eps', format='eps')
        # plt.show()

    # def plot_pixel_timeseries(self, folder_name, (x_index, y_index)):
    def plot_pixel_timeseries(self, folder_name, indices):
        """Gets series of pixel intensities and plots it then saves it

        :param folder_name: folder to extract series from
        :param (x_index, y_index) index of the pixel to plot
        :type folder_name str
        :type (x_index, y_index) tuple of ints, or tuple of lists of ints
        :return:
        """
        # TODO, swap x and y axes in the parameters
        # single pixel to plot
        (x_index, y_index) = indices
        if type(x_index) == int:
            print('Plotting ' + str(x_index) + ' , ' + str(y_index))
            ts = self.get_pixel_timeseries(folder_name, indices)
            indices = self.get_indices_from_filenames(folder_name)
            index_dates = dates.date2num(indices)
            fig, ax = plt.subplots()
            ax.plot_date(index_dates[500:600], ts[500:600], xdate=True, linestyle='solid', marker='None',
                         label=str(x_index) + ' , ' + str(y_index))
            ax.legend()
            ax.grid(b=True, which='major', color='#666666', linestyle='-')

            # Show the minor grid lines with very faint and almost transparent grey lines
            ax.minorticks_on()
            ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            fig.set_figwidth(40)
            fig.savefig(self.parent_folder + 'analysis/timeseries_TEST2_' + str(x_index) + '_' + str(y_index) + '.png')
            fig.savefig(self.parent_folder + 'analysis/timeseries_TEST2_' + str(x_index) + '_' + str(y_index) + '.svg')
            fig.clf()

        # multiple pixels to plot
        else:
            fig, ax = plt.subplots()
            for i in range(0, len(x_index)):
                print('Plotting ' + str(x_index[i]) + ' , ' + str(y_index[i]))
                ts = self.get_pixel_timeseries(folder_name, (x_index[i], y_index[i]))
                indices = self.get_indices_from_filenames(folder_name)
                index_dates = dates.date2num(indices)

                ax.plot_date(index_dates[500:600], ts[500:600], xdate=True, linestyle='solid', marker='None',
                             label=str(x_index[i]) + ' , ' + str(y_index[i]))

            ax.legend()
            ax.grid(b=True, which='major', color='#666666', linestyle='-')

            # Show the minor grid lines with very faint and almost transparent grey lines
            ax.minorticks_on()
            ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            fig.set_figwidth(40)
            fig.savefig(
                self.parent_folder + 'analysis/timeseries_TEST_' + str(x_index) + '_' + str(y_index) + '.png')
            fig.savefig(
                self.parent_folder + 'analysis/timeseries_TEST_' + str(x_index) + '_' + str(y_index) + '.svg')
            fig.clf()

    def get_pixel_timeseries(self, folder_name, indices):
        (y_index, x_index) = indices
        intensities = []
        photo_list = self.get_photo_list(folder_name)
        for image_name in photo_list:
            image = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
            intensities.append(image[y_index, x_index])

        print(intensities)
        return intensities

    def get_average_around_pixel(self, folder_name, indices):
        (y_index, x_index) = indices
        # (tl_row, tl_column) = top_left
        # (br_row, br_column) = bottom_right
        photo_list = self.get_photo_list(folder_name)
        averages = np.zeros((len(photo_list)))
        for i, name in enumerate(photo_list):
            photo = cv2.imread(folder_name + '/' + name, cv2.IMREAD_ANYDEPTH)
            averages[i] = np.mean(photo[y_index-1:y_index+2, x_index-1:x_index+2])

        return averages

    def plot_intensity_lines(self, folder_name, columns, froms, tos, destination):
        """The nice embarrassing thing about using github is when you start resorting to code like this

        :param folder_name:
        :param columns:
        :param froms:
        :param tos:
        :param destination: path from self.parent_folder, where output graphs to be stored
        :return:
        """
        # TODO improve the get min max to only get the min and max in the columns - how without 2 long for loops?
        photo_list = self.get_photo_list(folder_name)
        minimum, maximum = self.get_min_max_intensity_lines(folder_name, columns, froms, tos, non_zero=True)
        print("Will set minimum y to " + str(minimum - (0.1 * minimum)) +
              " (min - (0.1 * min)) and maximum y to " + str(maximum - (0.1 * maximum)))
        for name in photo_list:
            image = cv2.imread(folder_name + '/' + name, cv2.IMREAD_ANYDEPTH)
            line1 = image[froms[0]:tos[0], columns[0]]
            line2 = image[froms[1]:tos[1], columns[1]]
            line3 = image[froms[2]:tos[2], columns[2]]
            line4 = image[froms[3]:tos[3], columns[3]]
            line5 = image[froms[4]:tos[4], columns[4]]
            line6 = image[froms[5]:tos[5], columns[5]]
            line7 = image[froms[6]:tos[6], columns[6]]
            line8 = image[froms[7]:tos[7], columns[7]]
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.plot(line1, label='one')
            ax1.plot(line2, label='two')
            ax1.plot(line3, label='three')
            ax1.plot(line4, label='four')
            ax2.plot(line5, label='five')
            ax2.plot(line6, label='six')
            ax2.plot(line7, label='seven')
            ax2.plot(line8, label='eight')
            ax1.set_ylim(minimum - (0.1 * minimum), maximum + (0.1 * maximum))
            ax2.set_ylim(minimum - (0.1 * minimum), maximum + (0.1 * maximum))
            ax1.legend()
            ax2.legend()
            fig.set_figwidth(20)
            fig.savefig(self.parent_folder + destination + '/' + str(name) + 'intensitylineplots.png')
            fig.clf()
            plt.close(fig)

    def get_min_max_intensity_lines(self, folder_name, columns, froms, tos, non_zero=True):
        """ Used to get the maximums and minimums within the intensity lines, for plotting

        :param folder_name:
        :param columns:
        :param froms:
        :param tos:
        :param non_zero:
        :return:
        """
        minimum_pixel_values = []
        maximum_pixel_values = []
        photo_list = self.get_photo_list(folder_name)

        for i, name in enumerate(photo_list):
            pixels = []
            file_name = folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            pixels.extend(image[froms[0]:tos[0], columns[0]])
            pixels.extend(image[froms[1]:tos[1], columns[1]])
            pixels.extend(image[froms[2]:tos[2], columns[2]])
            pixels.extend(image[froms[3]:tos[3], columns[3]])
            pixels.extend(image[froms[4]:tos[4], columns[4]])
            pixels.extend(image[froms[5]:tos[5], columns[5]])
            pixels.extend(image[froms[6]:tos[6], columns[6]])

            if non_zero:
                pixels.sort()
                # need to make the 'pixels' variable into a np.ndarray i think
                new_pixels = np.full((len(pixels), 1), 1)
                for k, j in enumerate(pixels):
                    new_pixels[k, :] = j

                # this is a really fiddly and stupid way to get around how bad this is implemented for lists
                non_zero_pixels = new_pixels[np.nonzero(new_pixels)]
                minimum_pixel_values.append(non_zero_pixels[0])

            else:
                minimum_pixel_values.append(np.amin(pixels))

            maximum_pixel_values.append(np.amax(pixels))

        folder_minimum_pixel_value = np.amin(minimum_pixel_values)
        folder_maximum_pixel_value = np.amax(maximum_pixel_values)

        print("The min is: " + str(folder_minimum_pixel_value) + "  The max is: " + str(folder_maximum_pixel_value))
        return folder_minimum_pixel_value, folder_maximum_pixel_value

    def convert_8bit_to_16bit(self, folder_name):
        sixteen_bit_folder_name = folder_name + '_16b'

        try:
            print("Making dir " + str(sixteen_bit_folder_name) + " for masking")
            os.mkdir(sixteen_bit_folder_name)
        except OSError:
            print("Folder exists, have you already done this masking??")
            return

        photo_list = self.get_photo_list(folder_name)
        size_photo = cv2.imread(self.parent_folder + '/' + self.folder_list[0] +
                                '/' + photo_list[0], cv2.IMREAD_ANYDEPTH)
        photo_height, photo_width = np.shape(size_photo)

        for image_name in photo_list:
            print (folder_name + '/' + image_name)
            img = cv2.imread(folder_name + '/' + image_name, cv2.IMREAD_ANYDEPTH)
            scale = np.full([photo_height, photo_width], 256)

            sixteen_bit_image = np.multiply(img, scale)
            """
            sixteen_bit_image = img
            
            size = img.shape
            for row_pixel in range(0, size[0]):
                for column_pixel in range(0, size[1]):
                        sixteen_bit_image[row_pixel, column_pixel] = img[row_pixel, column_pixel] * 256
            """
            cv2.imwrite(sixteen_bit_folder_name + '/' + image_name, sixteen_bit_image.astype(np.uint16))

    def get_image_gradient(self, folder_name, output_folder):
        photo_list = self.get_photo_list(folder_name[0])
        image = cv2.imread(folder_name[0] + photo_list[0], cv2.IMREAD_ANYDEPTH)
        derivative = np.gradient(image)
        d1 = derivative[0]
        d2 = derivative[1]
        print(derivative)
        print(np.shape(d1))
        print(np.shape(d2))
        print(output_folder + photo_list[0][:-4] + 'd1.png')
        cv2.imwrite(output_folder + photo_list[0][:-4] + 'd1.png', d1)
        cv2.imwrite(output_folder + photo_list[0][:-4] + 'd2.png', d2)

    def get_plot_fft_pixel_timeseries(self, folder_name, indices, file_string):
        (x_index, y_index) = indices
        ts = self.get_pixel_timeseries(folder_name, (x_index, y_index))
        self.plot_fft_pixel_timeseries(folder_name, ts, str(x_index)+ '_' + str(y_index) + file_string)

    def plot_fft_pixel_timeseries(self, folder_name, time_series, file_string):
        """ Plots spectrum of a pixel intensity time series

        :param folder_name:
        :param file_string: to be added to the file name
        :return:
        """
        n = len(time_series)
        frequency = self.get_sampling_frequency(folder_name)
        d = 1.0 / frequency         # 'sample spacing'
        fig, ax = plt.subplots()
        sample_freqs = np.fft.rfftfreq(n, d)
        fourier = np.fft.rfft(time_series*np.hanning(len(time_series)))

        ax.plot(sample_freqs[1:], fourier.real[1:(len(time_series)/2)+1], label='real')
        ax.legend()
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax.set_xlabel('Frequency: Hz')

        fig.set_figwidth(30)
        fig.savefig(
            self.parent_folder + 'analysis/timeseries_fourier_'
            + file_string + '.png')
        fig.savefig(
            self.parent_folder + 'analysis/timeseries_fourier_'
            + file_string + '.svg')
        fig.clf()

    def get_indices_from_filenames(self, folder_name):
        #TODO some way of interpreting the date string and using that to define how to process it
        indices = []
        photo_list = self.get_photo_list(folder_name)
        for name in photo_list:
            indices.append(datetime.datetime.strptime(name[:-4], '%Y-%m-%d %H:%M:%S'))

        return indices

    def apply_bandpass_filter_timeseries(self, folder_name, indices, start_stop_freq, stop_stop_freq):
        """Gets timeseries of a pixel, plots it, applies bandpass filter, plots it

        :param folder_name:
        :param (x_index, y_index)
        :param start_stop_freq
        :param stop_stop_freq
        :return:
        """
        (x_index, y_index) = indices
        photo_list = self.get_photo_list(folder_name)

        ts = self.get_pixel_timeseries(folder_name, (x_index, y_index))
        self.plot_fft_pixel_timeseries(folder_name, ts, str(x_index) + '_' +  str(y_index) + 'pre_butterworth')
        n = len(ts)
        frequency = self.get_sampling_frequency(folder_name)
        d = 1.0 / frequency         # 'sample spacing'
        fig, ax = plt.subplots()
        sample_freqs = np.fft.rfftfreq(n, d)
        fourier = np.fft.rfft(ts)
        print(sample_freqs)
        nyquist = frequency / 2.0

        start_stop_band = start_stop_freq / nyquist
        stop_stop_band = stop_stop_freq / nyquist

        print(start_stop_band)
        print(stop_stop_band)

        sos = sgnl.butter(2, Wn=[start_stop_band, stop_stop_band], btype='bandstop', output='sos')
        filtered = sgnl.sosfilt(sos, ts)
        self.plot_fft_pixel_timeseries(folder_name, filtered, str(x_index) + '_' + str(y_index) + 'post_butterworth')
        fig, ax = plt.subplots()
        indices = self.get_indices_from_filenames(folder_name)
        index_dates = dates.date2num(indices)
        ax.plot_date(index_dates, ts, xdate=True, linestyle='solid', marker='None',
                     label=str(x_index) + ' , ' + str(y_index))
        ax.plot_date(index_dates, filtered, xdate=True, linestyle='solid', marker='None',
                     label=str(x_index) + ' , ' + str(y_index) + ' filtered')

        ax.legend()
        ax.grid(b=True, which='major', color='#666666', linestyle='-')

        # Show the minor grid lines with very faint and almost transparent grey lines
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig.set_figwidth(40)
        fig.savefig(self.parent_folder + 'analysis/timeseries_filtered_' + str(x_index) + '_' + str(y_index) + '.png')
        fig.savefig(self.parent_folder + 'analysis/timeseries_filtered_' + str(x_index) + '_' + str(y_index) + '.svg')
        fig.clf()

    def get_sampling_frequency(self, folder_name):
        """

        :param folder_name:
        :return:
        """
        indices = self.get_indices_from_filenames(folder_name)
        n = len(indices)
        time_diff = (indices[-1] - indices[0]).total_seconds()
        period = time_diff / n
        frequency = 1.0 / period
        return frequency

    def resample(self, folder_name):
        # get images
        # unravel images
        # get indices for the columns
        # save columns as .csv files (or groups of columns?)
        # clear memory ??
        # start looping through the columns (or groups of columns?)
        #       resample
        #       save as new .csv
        # start looping through the new columns
        #       save as new big .csv
        # start looping through as chunks in big .csv
        #       un-un-ravel rows back into 'images'
        #       save 'images' as images
        actual_pic_width = 1024
        actual_pic_height = 640
        block_size = 8
        resampled_folder_name = folder_name + '_resampled'
        temp_folder_name = self.parent_folder + 'temp'

        try:
            print("Making dir " + str(resampled_folder_name) + " for resampling")
            os.mkdir(resampled_folder_name)
        except OSError:
            print("Folder exists, have you already done this resampling??")
            return

        try:
            print("Making dir " + str(temp_folder_name) + " for resampling")
            os.mkdir(temp_folder_name)
        except OSError:
            print("Folder temp exists, have you already done this resampling??")
            return


        print("Writing to folder +" + str(resampled_folder_name))
        source_folder_name = folder_name
        photo_list = self.get_photo_list(folder_name)
        indices = self.get_indices_from_filenames(folder_name)
        size_photo = cv2.imread(folder_name + '/' + photo_list[0], cv2.IMREAD_ANYDEPTH)
        photo_height, photo_width = np.shape(size_photo)
        all_images = np.zeros((len(photo_list), photo_height * photo_width))

        for i, name in enumerate(photo_list):
            print("1, "+ str(i))
            file_name = folder_name + '/' + name
            image = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            all_images[i, :] = image.ravel()

        # save data in groups of columns (defined by block_size)
        # in .npy files
        for i in range(0, photo_height * photo_width, block_size):
            print("2, " + str(i) + " of " + str(photo_height * photo_width))
            data_for_csv = all_images[:, i:(i + 8)]
            np.save(self.parent_folder + "temp/" + str(i), data_for_csv)
            # D:\Laptop

        for i in range(0, photo_height * photo_width, block_size):
            print("3, " + str(i) + " of " + str(photo_height * photo_width))
            little_block = np.load(self.parent_folder + "temp/" + str(i) + ".npy")
            little_block_df = pd.DataFrame(little_block, index=indices)
            offset = None
            res = little_block_df.resample('1s', loffset=offset).asfreq()
            upsampled = res.interpolate(method='time')
            downsampled = upsampled.resample('1T').mean()  # likely need to change the resampling method here
            # downsampled = upsampled.resample('70s').mean() # alt version with closer to average period
            # at this stage could do a mean squared error calculation to see what sampling process gets the
            # closest results?
            downsampled_npy = downsampled.to_numpy()
            np.save(self.parent_folder + "temp/" +str(i) + 'downsampled', downsampled_npy)

        resampled_indices = downsampled.index
        resampled_all_images = np.zeros((len(resampled_indices), photo_height * photo_width))

        for i in range(0, photo_height * photo_width, block_size):
            print("4, " + str(i) + " of " + str(photo_height * photo_width))
            resampled_all_images[:, i:(i + 8)] = np.load(self.parent_folder + "/temp/" + str(i) + 'downsampled' + ".npy")

        #print("Building Pandas DataFrame")
        #resampled_dataset = pd.DataFrame(resampled_all_images, index=resampled_indices)
        #print("Writing .csv file")
        #resampled_dataset.to_csv(self.parent_folder + "resampled.csv")

        for i in range(0, len(resampled_indices)):
            print("5, " + str(i) + " of " + str(len(resampled_indices)))
            image = np.reshape(resampled_all_images[i,:], (photo_height, photo_width))
            cv2.imwrite(resampled_folder_name + '/' + str(resampled_indices[i])+'.png', image.astype(np.uint16))

        print("DONE")

    def parser(self, x):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    def get_average_pixel_value(self, folder_name):

        # get image list
        average_pixel_values = []
        photo_list = self.get_photo_list(folder_name)

        for name in photo_list:
            image = cv2.imread(folder_name + '/' + name, cv2.IMREAD_ANYDEPTH)
            average_pixel_values.append(np.mean(image))

        return average_pixel_values

    def get_average_nonzero_pixel_value(self, folder_name):
        # get image list
        average_pixel_values = []
        photo_list = self.get_photo_list(folder_name)

        for name in photo_list:
            image = cv2.imread(folder_name + '/' + name, cv2.IMREAD_ANYDEPTH)
            image = image.ravel()
            image.sort()
            non_zero_pixels = image[np.nonzero(image)]
            average_pixel_values.append(np.mean(non_zero_pixels))

        return average_pixel_values


    def get_and_plot_average_pixel_value(self, folder_name):

        average_pixel_values = self.get_average_pixel_value(folder_name)
        # plot
        fig, ax = plt.subplots()
        indices = self.get_indices_from_filenames(folder_name)
        index_dates = dates.date2num(indices)
        ax.plot_date(index_dates, average_pixel_values, xdate=True, linestyle='solid', marker='None')
        ax.grid(b=True, which='major', color='#666666', linestyle='-')
        # Show the minor grid lines with very faint and almost transparent grey lines
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig.set_figwidth(40)
        fig.savefig(
            self.parent_folder + 'analysis/timeseries_average_pixels_.png')
        fig.clf()
        # save
        return average_pixel_values, index_dates

    def test(self):
        print("A test github pycharm commit method")