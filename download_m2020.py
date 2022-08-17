#!/usr/bin/python3
'''
Rover Reconstruction Group
M2020 Download Script
Author: Ashwin Nedungadi & Tushar Barot
Maintainer: ashwin.nedungadi@tu-dortmund.de
Licence: The MIT License
Last Updated: 05/12/2022
Status: Production
'''
import sys
import wget
from bs4 import BeautifulSoup
import requests
import os

def bar_progress(current, total, width=80):
    """ A function that shows the progress in bytes of the current download """

    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

def isDirectory(url):
    """ A function that checks if the url is a directory
        :parameter url(str): the url to the directory where the images are
        :returns bool"""

    if(url.endswith('/')):
        return True
    else:
        return False

def fetch_data(url, output_directory):
    """ The main function which checks and downloads every file in the url recursively after checking if it's the
    relevant file type. A file counter is also incremented after each successful download and is printed at the end of
    the download process.
    #link['href'] shows every header in the url, useful for checking which files are there
    #(isDirectory(link['href'])) gives boolean output as to if it's a directory True or file False

    :parameter
    url(str): the url to the directory where the images are
    output_directory: the ouput_directory on local host where the files will be downloaded
    :returns
    does not return anything
    """
    page = requests.get(url).content
    soup = BeautifulSoup(page, 'html.parser')

    maybe_directories = soup.findAll('a', href=True)

    img_count = 0

    for link in maybe_directories:
        # check if files end with .IMG and .xml
        if (link['href'].endswith('.png') or link['href'].endswith('.xml') or link['href'].endswith('.IMG')):
            # workaround to avoid downloading the same file twice
            if img_count % 2 == 0:
                img_count += 1
                pass
            else:
                if os.path.exists(os.path.join(output_directory, link['href'])):
                    print(f"Skipping as file exists: {link['href']}")
                    img_count += 1
                    continue


                # If it passes the above checks now safe to download
                print("Downloading File: " + link['href'])
                # Counter for the files in directory
                img_count += 1
                # Download everything that ends with .IMG and .xml
                wget.download(url + link['href'], output_directory, bar=bar_progress)
                print("-------->Download Successful!")


    # Do a tally of how many files got downloaded
    print("Number of Image & xml files downloaded:", int(img_count)/2)


def download_pds(start_sol, end_sol, output_directory, pds_code = "browse", cam = "ncam"):
    """ function that prepares url to download from the relevant pds archive and camera.

    :parameter
    start_sol: the starting sol (int)
    end_sol: end sol (int)
    output_directory: The absolute directory path on local host where the files will be downloaded (str)
    pds_code: The PDS archive code. can be either "browse" (retreives .PNG) or "data" (retreives .IMG) (str)
    cam: The camera to download from. "ncam", "zcam" etc. (str)
    :returns
    Does not return anything
    """

    # Assign bundle
    if cam == "ncam":
        bundle = "mars2020_navcam_ops_raw"
    elif cam == "zcam":
        bundle = "mars2020_mastcamz_ops_raw"
    elif cam == "fcam" or cam == "rcam":
        bundle = "mars2020_hazcam_ops_raw"
    else:
        print("cam code not recognized, will fetch data for ncam.")
        bundle = "mars2020_navcam_ops_raw"

    if start_sol == end_sol:
        sol = str("{:05d}".format(start_sol))
        url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/' + bundle + '/' + pds_code + '/sol/' + sol + '/ids/edr/' + cam + '/'

        new_path = os.path.join(output_directory, sol)
        os.makedirs(new_path, exist_ok=True)
        print("Directory '% s' created" % sol)

        # Begin data download
        fetch_data(url, new_path)

    else:
        start = int(start_sol)
        end = int(end_sol)

        print("Data from SOLS", start, "-", end, "for", cam, "Will be downloaded.")
        for s in range(start, end+1):
            sol = str("{:05d}".format(s))
            url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/' + bundle + '/' + pds_code + '/sol/' + sol + '/ids/edr/' + cam + '/'

            # Make a new directory for each sol
            directory = sol
            new_path = os.path.join(output_directory, directory)
            os.makedirs(new_path, exist_ok=True)
            print("Directory '% s' created" % directory)

            # Begin data download
            fetch_data(url, new_path)


def download_filenames(text_file, output_directory):
    """ Given a .txt file, searches the PDS server after extracting relevant information from the filename.
    Warning: Filenames cannot have the prefix "MarsPerseverence" and must start with either "Z" or "N" as standard PDS format.
    :parameter
    text file: .txt file with all the names of the images to be downloaded
    Output Directory: Path of the output directory
    :returns
    does not return anything.
    """
     files_to_get = list()

    file = open(text_file, 'r')
    filenames = file.readlines()
    img_count = 0
    pull_count = 0
    # Strips the newline character
    for line in filenames:
        img_count += 1
        files_to_get.append(line.strip())

    # Hard Coded because it's reverse searching the .IMG from .png
    pds_code = "data"
    bundle = "mars2020_mastcamz_ops_raw"

    for f in files_to_get:
        temp_name = f.split("_")
        cam = temp_name[0][0].lower() + "cam"
        t_sol = temp_name[1]
        t_sol = int(t_sol)

        # Format sol for PDS
        sol = str("{:05d}".format(t_sol))

        # Assign bundle based on PDS code
        if cam.lower() == "ncam":
            bundle = "mars2020_navcam_ops_raw"
        elif cam.lower() == "zcam":
            bundle = "mars2020_mastcamz_ops_raw"

        # Replace extensions creating filenames to download
        base = os.path.splitext(f)[0]

        # Delete the last part of the file/PDS code
        temp_base = base.split("_")[:-1]
        new_base = "_".join(temp_base)

        # Assemble url to pull data from
        url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/' + bundle + '/' + pds_code + '/sol/' + sol + '/ids/edr/' + cam + '/'

        page = requests.get(url).content
        soup = BeautifulSoup(page, 'html.parser')

        maybe_directories = soup.findAll('a', href=True)

        for link in maybe_directories:
            # check if files end with .IMG and .xml
            if (link['href'].endswith('.IMG') or link['href'].endswith('.xml')):
                # take out the last 10 char or last code after _ and compare
                original_filename = (link['href']).split('.')[0]
                server_filename = original_filename.split("_")[:-1]
                server_filename = "_".join(server_filename)
                #print(original_filename, server_filename)
                #print(new_base, server_filename)

                # if the file isn't already in dir
                if not os.path.exists(os.path.join(output_directory, link['href'])):
                    # If the name of the file on PDS matches the name in filename.txt
                    if server_filename == new_base:
                        pull_count += 1
                        print(base, "was found with name-->", link['href'])

                        print("Downloading File: " + link['href'])
                        # Download everything that ends with .IMG and .xml
                        wget.download(url + link['href'], output_directory, bar=bar_progress)
                        print("-------->Download Successful!")

    print(img_count, "Files were provided, ", pull_count/2, "Files were successfully downloaded from the server.")


if __name__ == '__main__':

    # Set where the files have to be downloaded here
    output_directory = 'C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/download_test/'
    
    # Pay attention to which camera data of m2020 you are downloading in the url below: here it's the Navigation Camera
    # Nav, Mast, Front Haz, Rear Haz
    cams = ["ncam", "zcam", "fcam", "rcam"]

    # browse - png and minimal .xml files
    # data - IMG and complete .xml files with camera parameters (int and ext)
    pds_data_codes = ["browse", "data"]
    start_sol = 1
    end_sol = 3
    
    #download_pds(start_sol, end_sol, output_directory, pds_data_codes[1], cams[1])
    #download_pds(start_sol, end_sol, output_directory, pds_data_codes[0], cams[0])
