# !/usr/bin/python3
'''
Rover Reconstruction Group
M2020 Utility Script
Author: Ashwin Nedungadi and Tushar Jayesh Barot
Maintainer: ashwin.nedungadi@tu-dortmund.de and tushar.barot@tu-dortmund.de
Licence: The MIT License
Status: Production
'''

import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, shutil
from pathlib import Path
import cv2
import pds4_tools
import transforms3d
import math


def normalize(input):
    """ Given an image numpy array, normailzes the image to Uint8.
    :param np array
    :returns normalized np array """

    # Double check if downsampling is happening here, plt only accepts Uint8 and not Uint16 cv2.CV_16U
    normalized_image = cv2.normalize(input, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Manual Way of Normalizing
    # max_val = input.max()
    # min_val = 0
    # k = 255/max_val
    # normalized_image = input*k

    # Another way of normalizing
    # norm = (input - np.min(input)) / (np.max(input) - np.min(input))

    return normalized_image


def get_imgs(input_directory):
    """ A function that filters local .IMG and .png files from local directory, ignores thumbnails based on img_size
    :param absolute filepath of directory
    :returns list of .IMG or .png files in directory"""

    # read all files into a list
    os.chdir(input_directory)
    print(f"Local directory of .IMG files: {os.getcwd()}")

    # filter .img files only
    img_list = []
    for file in os.listdir(path=input_directory):
        if file.endswith(".IMG") or file.endswith(".png"):
            img_list.append(file)

    # files above ~100KB only
    imgs = []
    for file in img_list:
        img_size = os.path.getsize(os.path.join(input_directory, file))
        if img_size > 1000000:
            imgs.append(file)
    return imgs


def read_img(file):
    """ reads .IMG file and returns the rgb array
    :param .IMG filename
    :returns rbg np array"""
    img_bands = []
    band_dim = []
    image_array = []

    image = gdal.Open(os.path.join(os.getcwd(), file))
    # print(f'image: {file}')
    for band in range(image.RasterCount):
        band += 1

        np_band = np.array(image.GetRasterBand(band).ReadAsArray())
        img_bands.append(np_band)

    img_x = img_bands[0].shape[0]
    img_y = img_bands[0].shape[1]
    dim = len(img_bands)

    if dim == 1:
        # imgs = np.zeros((img_x,img_y), 'uint8')
        image_array.append(img_bands[0])
        band_dim.append(dim)

    else:
        img_rgb = np.zeros((img_x, img_y, dim), 'uint16')
        # Combining all color channels into a single array
        for i in range(dim):
            img_rgb[:, :, i] = img_bands[i]

        image_array.append(img_rgb)
        band_dim.append(dim)
        return image_array[0], band_dim[0]

    return image_array[0], band_dim[0]


def view(input_image):
    """ views an .IMG or .png file when called with the filename as argument.
    :param .IMG or .png file
    :returns None"""

    if input_image.endswith('.IMG'):

        image, bands = read_img(input_image)
        image = normalize(image)

        if bands == 1:
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111)

            c = ax.imshow(image, cmap='gray')

            plt.title('M2020 BW Image', fontweight="bold")
            plt.show()
        else:
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111)

            c = ax.imshow(image)

            plt.title('M2020 RGB Image', fontweight="bold")
            plt.show()

    elif input_image.endswith('.png'):
        # To implement png viewing part here
        png_img = mpimg.imread(input_image)
        png_plot = plt.imshow(png_img)
        plt.show()
    else:
        print("Error: File format not recognized. Must be .IMG or .png.")


def convert_to_png(input_directory, output_directory):
    """ saves the input .img file as a .png, if a directory is given, iterates through all .img files in directory and saves as .png in the optional directory given.
    :param input direcotry, output directory as absolute filepath
    :returns None"""
    file_count = 0
    # Is it a directory or single file?
    if os.path.isdir(input_directory):
        images = get_imgs(input_directory)

        for file in images:
            file_count += 1
            img_file = os.path.join(input_directory, file)
            image_array, bands = read_img(img_file)
            normalized_image = normalize(image_array)
            if bands > 1:
                plt.imsave(os.path.join(output_directory, Path(file).stem + '.png'), normalized_image, format='png')
            else:
                plt.imsave(os.path.join(output_directory, Path(file).stem + '.png'), image_array, format='png',
                           cmap='gray')

    else:
        if input_directory.endswith(".IMG"):
            image_array, bands = read_img(input_directory)
            normalized_image = normalize(image_array)
            if bands > 1:
                plt.imsave(os.path.join(output_directory, Path(input_directory).stem + '.png'), normalized_image,
                           format='png')
            else:
                plt.imsave(os.path.join(output_directory, Path(input_directory).stem + '.png'), image_array,
                           format='png', cmap='gray')

    if file_count:
        print(file_count, "Files were converted")
    else:
        print("Image converted successfully")


def get_filetags(image_name):
    """ Given an image, returns relevant information from the filename such as which stereo pair, which engineering camera and what level of processing has been done.
     :param image filename as absolute path
     :returns strings"""

    stereo_tag = ""
    camera_tag = ""
    image_processing_code = ""
    # to remove the absolute path details for furthur processing -Added by Tushar Jayesh Barot
    image_name = Path(image_name).name
    try:
        image_name = image_name.split('_')
        if image_name[0][0].upper() == "N":
            camera_tag = "Navigation Camera"
        elif image_name[0][0].upper() == "Z":
            camera_tag = "Mast Camera"
        elif image_name[0][0].upper() == "F":
            camera_tag = "Front Camera"
        elif image_name[0][0].upper() == "R":
            camera_tag = "Rear Camera"
        else:
            camera_tag = "N/A"

        if image_name[0][1].upper() == "L":
            stereo_tag = "Left"
        elif image_name[0][1].upper() == "R":
            stereo_tag = "Right"
        else:
            stereo_tag = "N/A"

        image_processing_code = image_name[3][3:6]

        return camera_tag, stereo_tag, image_processing_code
    except IndexError:
        print("Input path not of expected type. Please double check input.")


def sort_stereo(input_directory):
    """ Given a directory, sorts all stereo images into new directories for left and right cameras. Creates a copy, does not move original files.
    :param input directory as absolute file path
    :returns None"""

    images = get_imgs(input_directory)
    left_dir_name = "Camera_Left"
    right_dir_name = "Camera_Right"
    print("Making new directories for Left and Right stereo pairs...")
    left_path = os.path.join(input_directory, left_dir_name)
    right_path = os.path.join(input_directory, right_dir_name)
    os.mkdir(left_path)
    os.mkdir(right_path)

    for im in images:
        # Get the stem name as we need both IMG and xml too
        im_stem = Path(im).stem
        print('Working on:\t\t' + im)
        camera, stereo_pair, image_code = get_filetags(im)
        # print(camera, stereo_pair, image_code)

        existing_imgpath = os.path.join(input_directory, im)
        existing_xmlpath = os.path.join(input_directory, im_stem + ".xml")

        try:
            if stereo_pair.lower() == "left":

                shutil.copy(existing_imgpath, left_path)
                shutil.copy(existing_xmlpath, left_path)

            elif stereo_pair.lower() == "right":

                shutil.copy(existing_imgpath, right_path)
                shutil.copy(existing_xmlpath, right_path)
            else:
                print(im, "File was not moved as it's not of expected type.")
        except FileNotFoundError:
            print("Could not move", existing_xmlpath, im, "File may not exist.")


def find_stereo(image_path):
    """ Given a left or right camera image, returns the filename of the other pair. If no matching stereo pair is found, returns None.
    :param Image filename as absolute path
    :returns stereo pair as string"""
    # To Do: Depending on tests, must dynamically search for stereo pair in directory and not just edit filename for other pair.

    image_name = image_path.split("/")[-1]
    print("Given File:", image_name)
    camera, stereo_pair, image_code = get_filetags(image_name)

    if stereo_pair.lower() == "left":
        im = image_name.split('_')

        right_pair = im[0].replace("L", "R")

        im[0] = right_pair
        new_file = "_".join(im)
        print("Stereo Pair:", new_file)
        return new_file

    elif stereo_pair.lower() == "right":
        im = image_name.split('_')
        left_pair = im[0].replace("R", "L")

        im[0] = left_pair
        new_file = "_".join(im)
        print("Stereo Pair:", new_file)
        return new_file
    else:
        print("File name not of expected type.")
        return None


def clean_poses(pose_array):
    """ Helper function for get_poses(), cleans and repackages a dictionary input."""
    # vector origin
    clean_poses = dict()
    for key, val in pose_array[0].items():
        # print(key, val)
        clean_poses[key.strip("geom:")] = val

    # Quaternion
    for key, val in pose_array[1].items():
        clean_poses[key.strip("geom:")] = val

    return clean_poses


def get_poses(input_xml, frame):
    """ Given an xml file and frame as string, returns a dictionary with all relevant pose information
    :param input xml filename, one str tag from list:
        Nav: ["rover", "site", "rsm", "arm"]
        Mast: ["rover", "rsm", "arm_turret", "arm_pixl"]
    :returns 8 element dictionary containing position, quaternion & rotation direction
    note: requires pds4_tools package"""

    # Read structure using pds4_tools
    structures = pds4_tools.read(input_xml)
    # all poses are under the Coordinate_Space_Definition Tag
    parameters = structures.label.findall('.//geom:Coordinate_Space_Definition')

    # Depending on which frame is required, the index is assigned based on xml order
    cam_name = get_filetags(input_xml)[0]
    pose_index = []
    if 'nav' in cam_name.lower():
        pose_index = ["rover", "site", "rsm", "arm"]
    elif 'mast' in cam_name.lower():
        pose_index = ["rover", "rsm", "arm_turret", "arm_pixl"]
    try:
        index = pose_index.index(frame)
    except:
        print(f'\n{frame} does not exist in {input_xml}')
        return None

    xml_poses = parameters[index].to_dict()
    # list to store found poses
    pose = list()

    # Nested Dictionary Processing
    for val in xml_poses.values():
        for k1, v1 in val.items():
            # extract frame id from "local_identifier" xml tag
            if k1 == "local_identifier":
                frame_id = (v1[0].split('_')[0]).lower()

                if frame_id == "rover":
                    for k2, v2 in val.items():
                        if k2 == "geom:Vector_Origin_Offset":
                            pose.append(v2)
                        elif k2 == "geom:Quaternion_Plus_Direction":
                            pose.append(v2)
                    return clean_poses(pose)

                elif frame_id == "site":
                    for k2, v2 in val.items():
                        if k2 == "geom:Vector_Origin_Offset":
                            pose.append(v2)
                        elif k2 == "geom:Quaternion_Plus_Direction":
                            pose.append(v2)
                    return clean_poses(pose)

                elif frame_id == "rsm":
                    for k2, v2 in val.items():
                        if k2 == "geom:Vector_Origin_Offset":
                            pose.append(v2)
                        elif k2 == "geom:Quaternion_Plus_Direction":
                            pose.append(v2)
                    return clean_poses(pose)

                elif frame_id == "arm":
                    for k2, v2 in val.items():
                        if k2 == "geom:Vector_Origin_Offset":
                            pose.append(v2)
                        elif k2 == "geom:Quaternion_Plus_Direction":
                            pose.append(v2)
                    return clean_poses(pose)


def get_cahvore(input_xml):
    """ Given an xml file, returns a dictionary with all relevant CAHVORE information
     :param xml filename as absolute path
     :returns 7 element dictionary with keys corresponding to C-A-H-V-O-R-E
     note: requires pds4_tools package
     Mastcam: CAHVOR model
     Navcam: CAHVORE model

     """

    # Read structure and CAHVORE Label
    structures = pds4_tools.read(input_xml)

    cam_name, *temp = get_filetags(Path(input_xml).name)

    if 'Mast' in cam_name:
        parameters = structures.label.findall('.//geom:CAHVOR_Model')
    elif 'Nav' in cam_name:
        parameters = structures.label.findall('.//geom:CAHVORE_Model')

    # Setup dictionary for CAHVORE parameters
    camera_parameters = dict()

    # Extract Information from XML
    # Vector_Center
    x_position_unit = parameters[0].find('.//geom:x_position').text
    y_position_unit = parameters[0].find('.//geom:y_position').text
    z_position_unit = parameters[0].find('.//geom:z_position').text
    camera_parameters["Vector_Center"] = [x_position_unit, y_position_unit, z_position_unit]
    # Vector_Axis
    VA_x_unit = parameters[0].findall('.//geom:x_unit')[0].text
    VA_y_unit = parameters[0].findall('.//geom:y_unit')[0].text
    VA_z_unit = parameters[0].findall('.//geom:z_unit')[0].text
    camera_parameters["Vector_Axis"] = [VA_x_unit, VA_y_unit, VA_z_unit]
    # Vector Horizontal
    VH_x_pixel = parameters[0].findall('.//geom:x_pixel')[0].text
    VH_y_pixel = parameters[0].findall('.//geom:y_pixel')[0].text
    VH_z_pixel = parameters[0].findall('.//geom:z_pixel')[0].text
    camera_parameters["Vector_Horizontal"] = [VH_x_pixel, VH_y_pixel, VH_z_pixel]
    # Vector Vertical
    VV_x_pixel = parameters[0].findall('.//geom:x_pixel')[1].text
    VV_y_pixel = parameters[0].findall('.//geom:y_pixel')[1].text
    VV_z_pixel = parameters[0].findall('.//geom:z_pixel')[1].text
    camera_parameters["Vector_Vertical"] = [VV_x_pixel, VV_y_pixel, VV_z_pixel]
    # Vector Optical
    VO_x_unit = parameters[0].findall('.//geom:x_unit')[1].text
    VO_y_unit = parameters[0].findall('.//geom:y_unit')[1].text
    VO_z_unit = parameters[0].findall('.//geom:z_unit')[1].text
    camera_parameters["Vector_Optical"] = [VO_x_unit, VO_y_unit, VO_z_unit]
    # Radial Terms
    RT_C0 = parameters[0].findall('.//geom:c0')[0].text
    RT_C1 = parameters[0].findall('.//geom:c1')[0].text
    RT_C2 = parameters[0].findall('.//geom:c2')[0].text
    camera_parameters["Radial_Terms"] = [RT_C0, RT_C1, RT_C2]
    if 'Nav' in cam_name:
        # Entrance Terms
        ET_C0 = parameters[0].findall('.//geom:c0')[1].text
        ET_C1 = parameters[0].findall('.//geom:c1')[1].text
        ET_C2 = parameters[0].findall('.//geom:c2')[1].text
        camera_parameters["Entrance_Terms"] = [ET_C0, ET_C1, ET_C2]

    return camera_parameters


def debayer(array):
    """ Given a normalized numpy image array, returns the debayered numpy image array.
    :param numpy array
    :returns numpy array"""

    # Import here as this is the only function that uses the script. Make sure to have malvar_debayer in working directory
    from malvar_debayer import demosaicing_CFA_Bayer_Malvar2004
    # Normalize locally
    norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    # Debayer
    debayered = demosaicing_CFA_Bayer_Malvar2004(norm)

    return debayered


def get_cahv_ref(xml_file: str):
    """
    Given an xml as string, return a dictionary of pose components:
    3D translation vector and quaternion

    Example:
    input: 'ZL0_0096_0675441669_163EDR_N0040136ZCAM08054_110085J03.xml'
    output: {'x': '0.805029', 'y': '0.559415', 'z': '-1.91903', 'qcos': '0.979816', 'qsin1': '-0.000630205', 'qsin2': '-0.00411854', 'qsin3': '-0.199856'}

    Maintainer: Tushar Jayesh Barot
    """

    structures = pds4_tools.read(xml_file)
    parameters = structures.label.findall('.//geom:Camera_Model_Parameters')

    cahv_ref_quat = parameters[0].find('.//geom:Quaternion_Model_Transform').to_dict()
    cahv_ref_trvec = parameters[0].find('.//geom:Vector_Model_Transform').to_dict()

    cahv_ref = [cahv_ref_trvec, cahv_ref_quat]

    pose = list()
    for pose_component in cahv_ref:
        for i in pose_component.values():
            pose.append(i)
    return clean_poses(pose)


def get_pose_rsm_wrt_site(xml: str):
    '''
    returns the homogeneous matrix for the transformation of the rsm head wrt the site frame

    input: xml file of the image
    output: homogeneous matrix

    Concept:
    Navcam pose is derived from CAHV* model, RSM head frame, Rover_nav frame and the Site frame,
    whereas Mastcam has the exception of not using the site frame as the given pose is wrt SITE_INDEX_1 directly

    Maintainer: Tushar Jayesh Barot
    '''

    camera_name = ''
    poses = {}
    if 'nav' in get_filetags(xml)[0].lower():
        camera_name = 'nav'
        for frame in ["rover", "site", "rsm"]:
            poses[frame] = get_poses(xml, frame)
    elif 'mast' in get_filetags(xml)[0].lower():
        camera_name = 'mast'
        for frame in ["rover", "rsm"]:
            poses[frame] = get_poses(xml, frame)

    if camera_name == 'nav':  # only navcam has site frame
        trv_site = np.array([poses['site']['x_position'], poses['site']['y_position'], poses['site']['z_position']],
                            dtype=float)
        q_site = np.array(
            [poses['site']['qcos'], poses['site']['qsin1'], poses['site']['qsin2'], poses['site']['qsin3']],
            dtype=float)
        rmat_site = transforms3d.quaternions.quat2mat(q_site)
        tf_site = transforms3d.affines.compose(trv_site, rmat_site, np.ones(3))

    trv_rover = np.array([poses['rover']['x_position'], poses['rover']['y_position'], poses['rover']['z_position']],
                         dtype=float)
    trv_rsm = np.array([poses['rsm']['x_position'], poses['rsm']['y_position'], poses['rsm']['z_position']],
                       dtype=float)

    # quat2mat for all frames
    q_rover = np.array(
        [poses['rover']['qcos'], poses['rover']['qsin1'], poses['rover']['qsin2'], poses['rover']['qsin3']],
        dtype=float)
    rmat_rover = transforms3d.quaternions.quat2mat(q_rover)

    q_rsm = np.array([poses['rsm']['qcos'], poses['rsm']['qsin1'], poses['rsm']['qsin2'], poses['rsm']['qsin3']],
                     dtype=float)
    rmat_rsm = transforms3d.quaternions.quat2mat(q_rsm)

    # homogeneous matrices
    tf_rover = transforms3d.affines.compose(trv_rover, rmat_rover, np.ones(3))
    tf_rsm = transforms3d.affines.compose(trv_rsm, rmat_rsm, np.ones(3))

    if camera_name == 'nav':
        tf_rover_wrt_site = np.dot(tf_site, tf_rover)
    elif camera_name == 'mast':
        tf_rover_wrt_site = tf_rover
    tf_rsm_wrt_site = np.dot(tf_rover_wrt_site, tf_rsm)
    # print(tf_rsm_wrt_site)

    return (tf_rsm_wrt_site)


def get_pose_cahv(cam_model: str):
    '''
    returns the homogeneous matrix for the transformation of the camera wrt the site frame

    input: photogrammetric model derived from CAHV* model
    output: dict{'euler':[], 'trvec':[]}

    Maintainer: Tushar Jayesh Barot
    '''

    # translation vector
    trv_cahv = np.array([cam_model['Xc'], cam_model['Yc'], cam_model['Zc']], dtype=float)
    # orientation
    w_rad = math.radians(cam_model['w'])
    phi_rad = math.radians(cam_model['phi'])
    k_rad = math.radians(cam_model['k'])
    rmat_cahv = transforms3d.taitbryan.euler2mat(k_rad, phi_rad, w_rad)

    tf_cahv = transforms3d.affines.compose(trv_cahv, rmat_cahv, np.ones(3))

    return tf_cahv


if __name__ == '__main__':

    convert_to_png("C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/sol255/Sol54", "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/sol255/Sol54")