# Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)
# TODO: tell this to Brent: https://stackoverflow.com/a/25644503/1490584
import dataclasses
import shutil
from copy import copy
from datetime import datetime
from shutil import copy2
from types import SimpleNamespace
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
import cv2
from PIL import Image
import doxapy
from requests import head
from scipy import ndimage
from locate import this_dir
from pathlib import Path

import csv

tau = 2 * np.pi

# Used within main execution
def mkdir_and_delete_content(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for f in path.iterdir():
            f.unlink()
    else:
        path.mkdir()

    return path

# Used within function: get_indexes_of_a_circle_starting_from_behind
# Used within class: Robot
def roundi(x):
    return np.round(x).astype(int)

# Used within main execution
def binarize_image(filepath_from, filepath_to, upscale=1.0, **parameters):
    if "k" not in parameters:
        parameters["k"] = 0.2
    if "window" not in parameters:
        parameters["window"] = 75

    def read_image(file):
        return np.array(Image.open(file).convert("L"))

    # Read our target image and setup an output image buffer
    grayscale_image = read_image(filepath_from)
    if upscale != 1:
        grayscale_image = cv2.resize(grayscale_image, (0, 0), fx=upscale, fy=upscale)

    binary_image = np.empty(grayscale_image.shape, grayscale_image.dtype)

    # Pick an algorithm from the DoxaPy library and convert the image to binary
    sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    sauvola.initialize(grayscale_image)
    sauvola.to_binary(binary_image, parameters)
    plt.imsave(filepath_to, binary_image, cmap="gray")

# Used within function: get_indexes_of_all_pixels_forming_a_line
def get_indexes_of_all_pixels_within_any_three_points(p1, p2, p3):
    """
    Get the indexes of the pixels within a trianlge of three points
    :param im: image
    :param p1: point 1
    :param p2: point 2
    :param p3: point 3
    :return: indexes
    """
    # Create a mask of the image
    # Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)
    im_width = (
        np.max([p1[0], p2[0], p3[0]]) + 1 - (w_min := np.min([p1[0], p2[0], p3[0]]))
    )
    im_height = (
        np.max([p1[1], p2[1], p3[1]]) + 1 - (h_min := np.min([p1[1], p2[1], p3[1]]))
    )
    mask = np.zeros((im_height, im_width), dtype=np.uint8)

    # Create the triangle
    p1, p2, p3 = [np.array([p[0] - w_min, p[1] - h_min]) for p in [p1, p2, p3]]
    triangle = np.array([p1, p2, p3], dtype=np.int32)
    # Fill the triangle with 1
    cv2.fillConvexPoly(mask, triangle, 1)
    # Get the indexes of the pixels within the triangle
    indexes = tuple(i + x for (i, x) in zip(np.where(mask == 1), [h_min, w_min]))
    return indexes

# Used within function: draw_slime_trail
def get_indexes_of_all_pixels_within_any_four_points(p1, p2, p3, p4):
    """
    Get the indexes of the pixels within a quadrilateral of four points
    :param im: image
    :param p1: point 1
    :param p2: point 2
    :param p3: point 3
    :param p4: point 4
    :return: indexes
    """
    # Create a mask of the image
    # Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)

    im_width = (
        np.max([p1[0], p2[0], p3[0], p4[0]])
        + 1
        - (w_min := np.min([p1[0], p2[0], p3[0], p4[0]]))
    )
    im_height = (
        np.max([p1[1], p2[1], p3[1], p4[1]])
        + 1
        - (h_min := np.min([p1[1], p2[1], p3[1], p4[1]]))
    )
    mask = np.zeros((im_height, im_width), dtype=np.uint8)

    p1, p2, p3, p4 = [np.array([p[0] - w_min, p[1] - h_min]) for p in [p1, p2, p3, p4]]

    # Create the triangle
    cv2.fillConvexPoly(mask, np.array([p1, p2, p3], dtype=np.int32), 1)
    cv2.fillConvexPoly(mask, np.array([p1, p3, p4], dtype=np.int32), 1)
    cv2.fillConvexPoly(mask, np.array([p1, p4, p2], dtype=np.int32), 1)

    # Get the indexes of the pixels within the triangle
    indexes = tuple(i + x for (i, x) in zip(np.where(mask == 1), [h_min, w_min]))
    return indexes

# NOT USED
def get_indexes_of_a_circle(midpoint, radius, number_of_points=100, starting_angle=0):
    """
    Get the points of a circle
    :param midpoint: midpoint
    :param radius: radius
    :param number_of_points: number of points
    :param starting_angle: starting angle
    :return: points
    """
    starting_angle = starting_angle - tau * 0.25
    points = []
    for i in range(number_of_points):
        angle = starting_angle + i * tau / number_of_points
        points.append(
            (
                midpoint[0] + radius * np.cos(angle),
                midpoint[1] + radius * np.sin(angle),
            )
        )
    return (x := np.array(points))[:, 1], x[:, 0]

# Used in function(s): draw_on_image, get_best_angle_from_image (3 times)
def get_indexes_of_a_circle_starting_from_behind(
    midpoint, radius, starting_angle=0, number_of_points=359
):
    """
    Get the points of a circle
    :param midpoint: midpoint
    :param radius: radius
    :param number_of_points: number of points
    :param starting_angle: starting angle
    :return: points
    """
    starting_angle = starting_angle - tau * 0.25 + tau * 0.5
    points = []
    for i in range(number_of_points):
        angle = starting_angle + i * tau / number_of_points
        points.append(
            (
                midpoint[0] + radius * np.cos(angle),
                midpoint[1] + radius * np.sin(angle),
            )
        )
    return (x := roundi(np.array(points)))[:, 1], x[:, 0]

# NOT USED
def get_indexes_of_pixel_circle(midpoint, radius):
    """
    Get the indexes of the pixels forming a circle
    :param midpoint: midpoint of the circle
    :param radius: radius of the circle
    :return: indexes
    """
    # Create a mask of the image
    mask = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    # Create the circle
    cv2.circle(mask, (radius, radius), radius, 1, 1)
    # Get the indexes of the pixels within the circle
    indexes = tuple(
        m_idx + i - radius for (m_idx, i) in zip(np.where(mask == 1), midpoint[::-1])
    )

    # Hectic complex way to keep the circle going from left to right
    dividor = indexes[1] > midpoint[1]
    idxes_rows = [
        (i, j[0], j[1]) for i, j in zip(dividor, np.c_[indexes[0], indexes[1]])
    ]
    idxes_rows = np.array(sorted(idxes_rows))
    split_mask = idxes_rows[:, 0] == 1
    idxes_rows = np.r_[idxes_rows[split_mask], idxes_rows[~split_mask][::-1]]

    return (idxes_rows[:, 1], idxes_rows[:, 2])

# Used in function: draw_on_image
def get_indexes_of_all_pixels_forming_a_line(p1, p2):
    """
    Get the indexes of the pixels forming a line
    :param p1: point 1
    :param p2: point 2
    :return: indexes
    """
    # Get the indexes of the pixels within the triangle
    indexes = get_indexes_of_all_pixels_within_any_three_points(p1, p1, p2)
    return indexes

# NOT USED
def rotate_line_formed_by_two_points(p1, p2, angle):
    """
    Rotate a line formed by two points
    :param p1: point 1
    :param p2: point 2
    :param angle: angle to rotate
    :return: rotated line
    """
    # Get the indexes of the pixels within the triangle
    midpoint = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    p1 = np.array(p1)
    p2 = np.array(p2)
    p1 = p1 - midpoint
    p2 = p2 - midpoint
    p1 = np.array(
        [
            p1[0] * np.cos(angle) - p1[1] * np.sin(angle),
            p1[0] * np.sin(angle) + p1[1] * np.cos(angle),
        ]
    )
    p2 = np.array(
        [
            p2[0] * np.cos(angle) - p2[1] * np.sin(angle),
            p2[0] * np.sin(angle) + p2[1] * np.cos(angle),
        ]
    )
    p1 = p1 + midpoint
    p2 = p2 + midpoint
    return p1, p2

# NOT USED
def get_normal_vector_of_two_points(p_left, p_right):
    """
    Get the normal vector of a line formed by two points
    :param p_left: point 1
    :param p_right: point 2
    :return: normal vector in the direction a left and right sided creature would walk
    """
    # Get the indexes of the pixels within the triangle
    p1_, p2_ = np.array(p_right), np.array(p_left)
    normal_vector = np.array([p1_[1] - p2_[1], p2_[0] - p1_[0]])
    return normal_vector / np.linalg.norm(normal_vector)

# Used within class: Robot
def rotate_normal_vector(p, angle):
    """
    Rotate a normal vector
    :param p: normal vector
    :param angle: angle to rotate
    :return: rotated normal vector
    """
    # Get the indexes of the pixels within the triangle
    p = np.array(p)
    p = np.array(
        [
            p[0] * np.cos(angle) - p[1] * np.sin(angle),
            p[0] * np.sin(angle) + p[1] * np.cos(angle),
        ]
    )
    return p

# Used within main execution
def add_black_border_around_image(im, border_size):
    """
    Add a black border around an image
    :param im: image
    :param border_size: size of the border
    :return: image with border
    """
    # Add a border around the image
    im = np.pad(im, border_size, mode="constant", constant_values=0)
    return im

# Used in function: get_best_angle_from_image
def blur_1d(vec, sigma=3):
    """
    Add a 1d blur to a vector
    :return: blurred vector
    """
    return ndimage.gaussian_filter1d(vec, sigma)

# Used in main function
@dataclasses.dataclass
class Robot:
    """
    Robot class
    """

    position: Tuple[float, float] = (0, 0)
    angle: float = 0
    radius: int = 100
    step_size: int = 50
    slime_radius: int = 50

    def __post_init__(self):
        if self.angle - tau > 0:
            self.angle -= tau

    def get_left_right_points(self):
        """
        Get the left and right points
        :return: left and right points
        """

        # convert angle to a unit vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        )
        unit_vector = unit_vector * self.radius

        # rotate unit vector by 90 degrees left and then by 90 degrees right
        p_left = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, -tau * 0.25))
        )
        p_right = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, +tau * 0.25))
        )

        return p_left, p_right

    def get_left_right_slime_points(self):
        """
        Get the left and right points
        :return: left and right points
        """

        # convert angle to a unit vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        )
        unit_vector = unit_vector * self.slime_radius

        # rotate unit vector by 90 degrees left and then by 90 degrees right
        p_left = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, -tau * 0.25))
        )
        p_right = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, +tau * 0.25))
        )

        return p_left, p_right

    def walk(self, turn_angle):
        """
        Walk the robot
        :param turn_angle: angle to walk
        """
        # rotate the robot
        self.angle += turn_angle
        while(self.angle - tau > 0):
            self.angle -= tau

        # convert angle to a unit vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        ) * self.step_size
        self.position = roundi(np.array(self.position) + np.array(unit_vector))

    def draw_on_image(self, im):
        draw_0_on_img(
            im, get_indexes_of_a_circle_starting_from_behind, self.position, self.radius
        )
        l, r = self.get_left_right_points()

        draw_0_on_img(im, get_indexes_of_all_pixels_forming_a_line, l, r)

        # get normal vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        )
        draw_0_on_img(im, get_indexes_of_all_pixels_forming_a_line, self.position, roundi(np.array(self.position) + unit_vector*self.radius/4))


    def draw_slime_trail(self, im, r_prev, severity):
        l, r = self.get_left_right_slime_points()
        l_p, r_p = r_prev.get_left_right_slime_points()
        idxes = get_indexes_of_all_pixels_within_any_four_points(l, r, l_p, r_p)
        im[idxes] = np.clip(im[idxes] + severity, 0, 1)

    def get_best_angle_from_image(self, im, plot_dir=None):
        # as far as the bot looks
        idxes = get_indexes_of_a_circle_starting_from_behind(
            self.position, self.radius, self.angle
        )
        vec1 = im[idxes]

        # as far as the bot walks
        idxes = get_indexes_of_a_circle_starting_from_behind(
            self.position, self.step_size, self.angle
        )
        vec2 = im[idxes]

        # mask out indermediate objects (ideally it should include all pixels, aint nobody got time for that)
        mask_obsticle = np.zeros_like(vec1, dtype="int")
        for mult in [1, 0.8, 0.5, 0.3, 0.1]:
            idxes = get_indexes_of_a_circle_starting_from_behind(
                self.position, self.step_size * mult, self.angle
            )
            v = im[idxes]
            mask_obsticle += v == 0
        mask_obsticle = mask_obsticle.astype("bool")

        # add a bias so that the bot want's to look in the forward direction
        bias = np.r_[np.linspace(0.85, 1, 180), np.linspace(1, 0.85, 180)[1:]]

        # calculate a score for each angle degree
        vec = vec1 * vec2 * bias
        mask = vec <= 0

        # blur the score so that the bot doesn't want to walk exacly next to an obsticle/ slime trail
        vec = blur_1d(vec)
        for m in [mask_obsticle, mask]:
            vec[m] = 0

        # basically just a hook to be able to see what's going on
        if plot_dir is not None:
            x = plt.linspace(-180, 179, 359)
            plt.plot(x, vec)
            #plt.show()

        # finally, get the best angle to walk to next
        argm = np.argmax(vec)
        stop = mask_obsticle[argm]
        angle = ((argm - 180) / 360) * tau

        if plot_dir is not None:
            plt.plot([argm-180, argm-180], [0, 1], "k")
            path = Path(plot_dir, get_current_datetime_as_string() + ".png")
            # print(path)
            plt.savefig(path, dpi=50)
            plt.close()
            shutil.copy2(path, Path(plot_dir, "__latest__.png"))

        return angle, stop

# Used in function: draw_on_image
def draw_0_on_img(im, f, *args, **kwargs):
    indexes = f(*args, **kwargs)
    im[indexes] = 0

# Used in function: imshow_gray_as_purple, get_best_angle_from_image
def get_current_datetime_as_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

# Used in main execution
def as_degree(angle):
    return angle * 360 / tau


# Note that you should edit this further in order to export images to use in your presentation
# Used in main execution
def imshow_gray_as_purple(im, dir_location, *args, **kwargs):
    im = np.stack([im, im, im], axis=-1)

    mask0 = im == 0
    mask1 = im == 1

    # TODO: ask Brent use some kind of interpolation to get from light purple to dark purple
    #  https://stackoverflow.com/questions/73314297

    # purple = 1,0,1
    im[:, :, 1] = 0.7

    im[np.where(mask0)] = 0
    im[np.where(mask1)] = 1

    # TODO: ask Brent to export the image as a png wit a high resolution
    #  also, be able to overwrite a certain directory with the images, so that cou can choose a directory to save the images to
    #  should probably be a argument to this function then
    #plt.imshow(im, *args, **kwargs)
    #plt.savefig(Path(dir_location, get_current_datetime_as_string()+".png"), dpi=300, bbox_inches="tight")
    path = Path(dir_location, get_current_datetime_as_string()+".png")
    plt.imsave(path, im)

    copy2(path, Path(path.parent, "__latest__.png"))

    # TODO: uncomment this to show the plot on your screen
    #plt.show()
    #plt.clf()

"""
if 1:
    A = (plt.imread(Path(this_dir(), "binimage.png"))[:, :, 0]).astype("float")
    A = add_black_border_around_image(A, 120)

    # TODO: since the parameters are here, why not ask Brent put the rest of the code into a function?
    p = SimpleNamespace(
        start_position=(6000, 280),
        start_angle=tau * 0.1,
        radius=40,
        stepsize=20,
        slime_radius=20,
        slime_severity=0.04,
    )

    r = Robot(p.start_position, 448*tau/360, p.radius, p.stepsize, p.slime_radius); r.draw_on_image(A); print("->", as_degree(r.angle))
    r2 = copy(r); r2.walk(140*tau/360); r2.draw_on_image(A); print("->", r2.angle)

    r3 = Robot((5900, 240), 228*tau/360, p.radius, p.stepsize, p.slime_radius); r3.draw_on_image(A); print("->", as_degree(r3.angle))
    r3.walk(0); r3.draw_on_image(A); print("->", as_degree(r3.angle))

    plt.imsave(Path(this_dir(), "dump.png"), np.stack([A, A, A], axis=-1))
"""
def main_execution(xvalue, yvalue, tau_input, run, stop_range, draw_steps, rad, step_length, radius_slime, severity_slime, draw_image, from_image_location, to_image_location, plot_folder_name1, example_folder_name1, plot_folder_name2, example_folder_name2):
    n = run

    print('Starting on Location ' + from_image_location + ' @ x-coordinate: ' + str(xvalue) + ', y-coordinate: ' + str(yvalue) + '.')

    if n == 1:

        binarize_image(
            Path(this_dir(), from_image_location),
            Path(this_dir(), to_image_location),
            upscale=4,
            k=0.22,
            window=75,
        )
        A = (plt.imread(Path(this_dir(), to_image_location))[:, :, 0]).astype("float")
        A_slime = A

        # TODO: since the parameters are here, why not ask Brent put the rest of the code into a function?
        p = SimpleNamespace(
            start_position=(xvalue, yvalue),
            start_angle=tau_input * 0.1,
            radius=rad, #p = 100
            stepsize=step_length, #p = 20
            slime_radius=radius_slime, #p = 20
            slime_severity=severity_slime, #p = 0.1
        )
        plotdir = mkdir_and_delete_content(Path(this_dir(), plot_folder_name1, example_folder_name1))
        angledir = mkdir_and_delete_content(Path(this_dir(), plot_folder_name2, example_folder_name2))

        # TODO: maybe make border radius * 1.1 or something, but that gets convoluted with the start position!
        A = add_black_border_around_image(A, 120)
        A_slime = add_black_border_around_image(A_slime, 120)

        # Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)
        r = Robot(p.start_position, p.start_angle, p.radius, p.stepsize, p.slime_radius)
        r.draw_on_image(A)

        r_prev = copy(r)
        angle = 0
        for i in range(stop_range):
            if i % draw_steps == 0:
                if draw_image == True:
                    imshow_gray_as_purple(A_slime, plotdir)
                else:
                    continue
            try:
                r.walk(angle)
                r.draw_on_image(A)
                r.draw_slime_trail(A_slime, r_prev, -p.slime_severity)
                angle, stop = r.get_best_angle_from_image(
                    A_slime,
                    #angledir
                )
                print(f"{int(r.angle*360/tau_input)}\t{int(angle*360/tau_input)}")
                if stop:
                    break
                r_prev = copy(r)

                # Convert A and A_slime to dataframe

            # walked out of the image
            except IndexError:
                break

        #r_prev.get_best_angle_from_image(A_slime, True)
        #r.get_best_angle_from_image(A_slime, True)

        imshow_gray_as_purple(A_slime*A, plotdir)

        print('Ending on Location ' + from_image_location + '.')
    else: 
        print("Did not execute")

xvalues = [550,550,2000]
yvalues = [1150,3000,800]
range_value = [90000,90000,90000]
go_no_go = [1,1,1]
steps = [1,1,1]
v_radius = [100,100,100]
v_step_length = [20,20,20]
v_radius_slime = [20,20,20]
v_severity_slime = [0.1,0.1,0.1]
v_draw_image = [True,True,True]

image_from_list = ['Meerandal_Vinyard_V1.png','Meerandal_Vinyard_V2.png','Meerandal_Vinyard_V3.png']
image_to_list = ['binimage.png','binimage_v2.png','binimage_v3.png']
folder_plots1 = ['plots','plots2','plots3']
folder_example1 = ['example','example2','example3']
angle_plots1 = ['plots','plots2','plots3']
angle_examples = ['example_angle','example_angle2','example_angle3']

print("The maximum number of images available are " + str(len(image_from_list)))
set_image_no = int(input("Start at Image number (e.g. 1): "))

image_no = 0

if set_image_no > len(image_from_list) or set_image_no == None:
    print("The number you selected is not in range. The maximum number of images available are " + str(len(image_from_list)))
    set_image_no = int(input("Start at Image number (e.g. 1): "))
else: 
    image_no = set_image_no - 1

image_no = set_image_no - 1

while image_no <= 2:
    main_execution(
        xvalues[image_no],
        yvalues[image_no],
        tau, 
        go_no_go[image_no], 
        range_value[image_no], 
        steps[image_no], 
        v_radius[image_no], 
        v_step_length[image_no], 
        v_radius_slime[image_no], 
        v_severity_slime[image_no], 
        v_draw_image[image_no], 
        image_from_list[image_no], 
        image_to_list[image_no], 
        folder_plots1[image_no], 
        folder_example1[image_no], 
        angle_plots1[image_no], 
        angle_examples[image_no]
    )
    image_no += 1

# binarize_image(
#     Path(this_dir(), "apex.png"),
#     Path(this_dir(), "apex_binimage.png"),
#     upscale=4,
#     k=0.22,
#     window=75,
# )

# def store_bin_image_in_csv(image_name, csv_to_store):

#     file = Path(this_dir(), image_name)
#     test = np.array(Image.open(file).convert("L"))

#     total_rows = len(test)
#     # print(total_rows)
#     total_columns = len(test[0])
#     # print(total_columns)

#     # print(len(test[1]))

#     f = open(Path(this_dir(), csv_to_store), 'w', newline='')
#     writer = csv.writer(f)

#     i = 0
#     while i <= total_rows:
#         try:
#             n = 0
#             string = []
#             while n <= total_columns:
#                 try:
#                     string.append(str(test[i][n]))
#                     n += 1
#                 except:
#                     break
#             writer.writerow(string)
#             # print(string)
#             # print('row ' + str(i) + ' completed')
#             i += 1
#         except:
#             break

#     f.close()

# v_image_name = "sample_image.png"
# v_csv_to_store = "binary_image_3.csv"

# store_bin_image_in_csv(v_image_name,v_csv_to_store)