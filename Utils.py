"""
Helper functions for the project fractures
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import os
import shutil
import cv2
import json

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/Code/Datasets/Scaphoid/'

rawCR_dir = home_dir + 'Raw_Scaphoid_CR/'
test_loc_folder = home_dir + 'Test/'
cleanCR_folder = home_dir + 'Cleaned_CR_Single/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

def process_raw():

    """
    This function takes the 9883 raw DICOMs and removes non image files, then saves the DICOMs
    as filename: Accno_BodyPart/Side_View
    """

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'Cleaned_CR/'
    filenames = sdl.retreive_filelist('*', True, rawCR_dir)
    filenames2 = sdl.retreive_filelist('**', True, rawCR_dir)
    filenames += filenames2

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0
    saved = []

    for file in filenames:

        # Load the Dicom
        try:
            img, _, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information. Only body part fail will fail us
        view, laterality, part, accno = 0, 0, 0, 0

        # Laterality: L, R or UK
        try:laterality = header['tags'].ImageLaterality.upper()
        except:
            try:laterality = header['tags'].Laterality.upper()
            except:
                try:
                    if 'LEFT' in header['tags'].StudyDescription.upper():laterality = 'L'
                    else: laterality = 'R'
                except:
                    try:
                        if 'LEFT' in header['tags'].SeriesDescription.upper():laterality = 'L'
                        else:laterality = 'R'
                    except: laterality = 'UKLAT'

        # Accession number
        try:
            dir = os.path.dirname(file)
            accno = dir.split('/')[-2]
        except:
            try: accno = header['tags'].StudyID
            except:
                try: accno = header['tags'].AccessionNumber
                except Exception as e:
                    print('Header error: %s' % e)
                    continue

        # View 'LAT, PA, OBL, TAN. Maybe use SeriesNumber 1 = PA, 2 = LAT if 2 views etc
        try: view = header['tags'].ViewPosition.upper()
        except:
            try: view = 'V' + str(header['tags'].SeriesNumber)
            except: view = 'UKVIEW'

        if not view:
            try: view = 'V' + header['tags'].SeriesNumber
            except: view = 'UKVIEW'

        # PART: WRIST, HAND
        try:
            if 'WRIST' in header['tags'].StudyDescription.upper():
                part = 'WRIST'
            elif 'HAND' in header['tags'].StudyDescription.upper():
                part = 'HAND'
            elif 'WRIST' in header['tags'].SeriesDescription.upper():
                part = 'WRIST'
            elif 'HAND' in header['tags'].SeriesDescription.upper():
                part = 'HAND'
            else:
                part = 'UKPART'
        except:
            try:
                part = header['tags'].BodyPartExamined.upper()
            except Exception as e:
                print('Header error: %s' % e)
                continue

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        """

        # Sometimes lateraity is off
        if laterality != 'L' and laterality != 'R':
            if 'RIGHT' in header['tags'].StudyDescription.upper(): laterality = 'R'
            elif 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'

        # Get instance number from number in this folder with these accnos
        savedir = (save_path + accno + '/')
        try: copy = len(sdl.retreive_filelist('dcm', True, savedir)) + 1
        except: copy = 1

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view + '-' + str(copy)

        # Skip non wrists or hands
        if 'WRIST' not in part and 'HAND' not in part:
            print ('Skipping: ', dst_File)
            continue

        # Filename
        savefile = savedir + dst_File + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)
        #savefile = save_path + dst_File + '.dcm'

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        #if index % 10 == 0 and index > 1:
        print('Saving pt %s of %s to dest: %s' % (index, len(filenames), dst_File))

        # Increment counters
        index += 1
        del img

    print('Done with %s images saved' % index)


def raw_to_PNG():

    """
    This function takes the 9883 raw DICOMs and removes non image files, then saves the DICOMs as PNGs
    of filename: Accno_BodyPart/Side_View. The PNG format is used because it supports 16 bit grayscale
    """

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'PNG_Files/'
    filenames = sdl.retreive_filelist('**', True, rawCR_dir)
    shuffle(filenames)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    for file in filenames:

        # Load the Dicom
        try:
            image, _, _, photometric, header = sdl.load_DICOM_2D(file)
            if photometric == 1: image *= -1
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information
        try:
            view = header['tags'].ViewPosition.upper()
            laterality = header['tags'].ImageLaterality.upper()
            part = header['tags'].BodyPartExamined.upper()
            accno = header['tags'].AccessionNumber
        except Exception as e:
            #print('Header error: %s' %e)
            continue

        image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        """

        # Sometimes lateraity is off
        if laterality != 'L' and laterality != 'R':
            if 'RIGHT' in header['tags'].StudyDescription.upper(): laterality = 'R'
            elif 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Skip non wrists or hands
        if 'WRIST' not in part and 'HAND' not in part:
            print ('Skipping: ', dst_File)
            continue

        # Filename
        # savedir = (save_path + accno + '/')
        # savefile = savedir + dst_File + '.png'
        # if not os.path.exists(savedir): os.makedirs(savedir)
        savefile = save_path + dst_File + '.png'

        # Save the image
        sdl.save_image(image, savefile)
        print('Saving pt %s of %s to dest: %s' % (index, len(filenames), dst_File))

        # Increment counters
        index += 1

    print('Done with %s images saved' % index)


def check_raw():

    """
    This function takes the 9883 raw DICOMs and removes non image files, then saves the DICOMs
    as filename: Accno_BodyPart/Side_View
    Base = 9833, Headers = 8245, images = 5458, header only fails = 0 (5458 saved)
    """

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'Cleaned_CR/'
    filenames = sdl.retreive_filelist('**', True, rawCR_dir)
    shuffle(filenames)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    for file in filenames:

        # Load the Dicom
        try:
            img, _, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            #print('DICOM Error: %s' %e)
            continue

        # Retreive the view information. Only body part fail will fail us
        view, laterality, part, accno = 0, 0, 0, 0

        # Laterality: L, R or UK
        try: laterality = header['tags'].ImageLaterality.upper()
        except:
            try: laterality = header['tags'].Laterality.upper()
            except:
                try:
                    if 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'
                    else: laterality = 'R'
                except:
                    try:
                        if 'LEFT' in header['tags'].SeriesDescription.upper(): laterality = 'L'
                        else: laterality = 'R'
                    except Exception as e:
                        laterality = 'UKLAT'

        # Accession number
        try: accno = header['tags'].AccessionNumber
        except:
            try: accno = header['tags'].StudyID
            except:
                dir = os.path.dirname(file)
                accno = dir.split('/')[-3]

        # View 'LAT, PA, OBL, TAN. Maybe use SeriesNumber 1 = PA, 2 = LAT if 2 views etc
        try:
            view = header['tags'].ViewPosition.upper()
        except:
            try: accno = header['tags'].StudyID
            except Exception as e:
                view = 'UKVIEW'

        # PART: WRIST, HAND
        try:
            if 'WRIST' in header['tags'].StudyDescription.upper(): part = 'WRIST'
            elif 'HAND' in header['tags'].StudyDescription.upper(): part = 'HAND'
            elif 'WRIST' in header['tags'].SeriesDescription.upper(): part = 'WRIST'
            elif 'HAND' in header['tags'].SeriesDescription.upper(): part = 'HAND'
            else: part = 'UKPART'
        except:
            try: part = header['tags'].BodyPartExamined.upper()
            except Exception as e:
                print('Header error: %s' % e)
                continue

        """
            Some odd values that appear:
            view: TAN, LATERAL, NAVICULAR, LLO
            part: PORT WRIST, 
        # """

        # # Sometimes lateraity is off
        # if laterality != 'L' and laterality != 'R':
        #     if 'RIGHT' in header['tags'].StudyDescription.upper(): laterality = 'R'
        #     elif 'LEFT' in header['tags'].StudyDescription.upper(): laterality = 'L'
        #
        # # Set destination filename info
        # dst_File = accno + '_' + laterality + '-' + part + '_' + view
        #
        # # Skip non wrists or hands
        # if 'WRIST' not in part and 'HAND' not in part:
        #     print ('Skipping: ', dst_File)
        #     continue

        # Increment counters
        index += 1
        del img

    print('%s images saved (header)' % index)


def check_empty():

    """
    Check the raw CR subfolders for empty ones
    :return:
    """

    # First retreive lists of the the filenames
    folders = list()
    for (dirpath, dirnames, filenames) in os.walk(rawCR_dir):
        folders.append(filenames)

    # Load the filenames and randomly shuffle them
    save_path = home_dir + 'Cleaned_CR/'
    filenames = sdl.retreive_filelist('*', True, rawCR_dir)
    filenames2 = sdl.retreive_filelist('**', True, rawCR_dir)
    filenames += filenames2
    filenames3 = sdl.retreive_filelist('***', True, rawCR_dir)
    split = [x for x in filenames if len(x.split('/')) == 9]
    subsplit = [x for x in filenames if len(x.split('/')) == 8]
    print ('%s of %s Empty folders in %s' %(len(split), len(filenames), save_path))


def filter_DICOM(file, show_debug=False):
    """
    Loads the DICOM image, filters by what body part and view we want, then returns the info
    :return:
    """

    # Load the Dicom
    try:
        img, _, _, _, header = sdl.load_DICOM_2D(file)
    except Exception as e:
        if show_debug: print('DICOM Error: %s' %e)
        return

    # Retreive the view information. Only body part fail will fail us
    view, laterality, part, accno = 0, 0, 0, 0

    """
    Some odd values that appear:
    view: TAN, LATERAL, NAVICULAR, LLO
    part: PORT WRIST, 
    """

    # Laterality: L, R or UK
    try:
        laterality = header['tags'].ImageLaterality.upper()
    except:
        try:
            laterality = header['tags'].Laterality.upper()
        except:
            try:
                if 'LEFT' in header['tags'].StudyDescription.upper():
                    laterality = 'L'
                else:
                    laterality = 'R'
            except:
                try:
                    if 'LEFT' in header['tags'].SeriesDescription.upper():
                        laterality = 'L'
                    else:
                        laterality = 'R'
                except Exception as e:
                    if show_debug: print('Laterality Error: %s' % e)
                    laterality = 'UKLAT'

    # Accession number
    try:
        accno = header['tags'].AccessionNumber
    except:
        try:
            accno = header['tags'].StudyID
        except:
            dir = os.path.dirname(file)
            accno = dir.split('/')[-3]

    # View 'LAT, PA, OBL, TAN. Maybe use SeriesNumber 1 = PA, 2 = LAT if 2 views etc
    try: view = header['tags'].ViewPosition.upper()
    except: view = 'UKVIEW'

    # PART: WRIST, HAND
    try:
        if 'WRIST' in header['tags'].StudyDescription.upper():
            part = 'WRIST'
        elif 'HAND' in header['tags'].StudyDescription.upper():
            part = 'HAND'
        elif 'WRIST' in header['tags'].SeriesDescription.upper():
            part = 'WRIST'
        elif 'HAND' in header['tags'].SeriesDescription.upper():
            part = 'HAND'
        else:
            part = 'UKPART'
    except:
        try:
            part = header['tags'].BodyPartExamined.upper()
        except Exception as e:
            if show_debug: print('Header Error: %s' %e)
            return

    # Return everything
    return img, view, laterality, part, accno, header


def check_stats():

    """
    This function checks the stats for the bounding boxes and prints averages
    """

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder)
    shuffle(filenames)
    gtboxes = sdl.load_CSV_Dict('filename', 'gtboxes.csv')

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0
    heights, widths, ratio = [], [], []
    nheights, nwidths = [], []
    iheights, iwidths = [], []

    for file in filenames:

        # Load the info
        image, view, laterality, part, accno, header = filter_DICOM(file)

        # Set destination filename info
        dst_File = accno + '_' + laterality + '-' + part + '_' + view

        # Now compare to the annotations folder
        try: annotations = gtboxes[dst_File+'.png']['region_shape_attributes']
        except: continue

        # Success, convert the string to a dictionary
        gtbox = json.loads(annotations)
        if not gtbox: continue

        # Normalize the gtbox, to [ymin, xmin, ymax, xmax, cny, cnx, height, width, origshapey, origshapex]
        shape = image.shape
        gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                            gtbox['y']+(gtbox['height']/2), gtbox['x']+(gtbox['width']/2), gtbox['height'], gtbox['width']])
        norm_gtbox = np.asarray([gtbox[0]/shape[0], gtbox[1]/shape[1],
                                 gtbox[2]/shape[0], gtbox[3]/shape[1],
                                 gtbox[4]/shape[0], gtbox[5]/shape[1],
                                 gtbox[6] / shape[0], gtbox[7] / shape[1],
                                 shape[0], shape[1]]).astype(np.float32)


        # Increment counters
        index += 1
        print ('%s of %s Done' %(index, len(filenames)))
        nheights.append(norm_gtbox[4])
        nwidths.append(norm_gtbox[5])
        heights.append(gtbox[4])
        widths.append(gtbox[5])
        ratio.append(gtbox[4]/gtbox[5])
        iheights.append(image.shape[0])
        iwidths.append(image.shape[1])
        del image

    # Done with all patients
    heights, widths, ratio = np.asarray(heights), np.asarray(widths), np.asarray(ratio)
    iheights, iwidths = np.asarray(iheights), np.asarray(iwidths)
    nheights, nwidths = np.asarray(nheights), np.asarray(nwidths)
    print('%s bounding boxes. H/W AVG: %.3f/%.3f Max: %.3f/%.3f STD: %.3f/%.3f' % (
    index, np.average(heights), np.average(widths), np.max(heights), np.max(widths), np.std(heights), np.std(widths)))
    print('%s Norm bounding boxes. H/W AVG: %.3f/%.3f Max: %.3f/%.3f STD: %.3f/%.3f' % (
        index, np.average(nheights), np.average(nwidths), np.max(nheights), np.max(nwidths),  np.std(nheights), np.std(nwidths)))
    print('%s Images. H/W AVG: %.1f/%.1f Max: %.1f/%.1f STD: %.3f/%.3f' % (
        index, np.average(iheights), np.average(iwidths), np.max(iheights), np.max(iwidths),  np.std(iheights), np.std(iwidths)))
    print('%s Ratios. Max: %.2f, Min: %.2f, Avg: %.2f STD: %.3f' % (
        index, np.max(ratio), np.min(ratio), np.average(ratio),  np.std(ratio)))


# check_empty()
# process_raw()
#raw_to_PNG()
#check_raw()
check_stats()
