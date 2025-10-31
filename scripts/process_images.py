#!/usr/bin/env python
import numpy as np
import argparse
import pandas as pd
import os
from plantcv import plantcv as pcv
import plantcv.utils
from plantcv import parallel
from plantcv.plantcv.filters import obj_props
from plantcv.parallel import workflow_inputs
from plantcv.plantcv.filters.eccentricity import eccentricity
import cv2
import json

# Funtion to extract the name of the image
def extractName(a):
    base=os.path.basename(a)
    return os.path.splitext(base)[0]

# Initialize instance of options called args to mimic the use of argparse in workflow script
args = workflow_inputs()
pcv.params.debug = args.debug
    
# Read in image
img, imgpath, imgname = pcv.readimage(filename=args.image1)

# STEP 1.1: Use the output file from `plantcv-train.py` to run the multiclass naive bayes
 # classification on the image. The function below will print out 4 masks (background, mesocarp,
 # endocarp, seeds). Inputs:   rgb_img - RGB image data and  pdf_file - Output file containing PDFs
 # from `plantcv-train.py`
mask = pcv.naive_bayes_classifier(rgb_img=img, pdf_file="./data/Bayes.txt")

# STEP 2.1: Post-processing of background,meso and endo mask to fill objects and dilate to separate from salt noise
 # invert to get fruit_mask
b_step = pcv.fill(bin_img=mask['background'], size=50)
b_step = pcv.dilate(b_step, ksize=9, i=1)
m_step = pcv.fill(bin_img=mask['mesocarp'], size=50)
e_step = pcv.fill(bin_img=mask['endocarp'], size=2500)
fruit_mask = pcv.invert(b_step)

# STEP 3.1 : Analyze color for endocarp and mesocarp and stored Hue circle value mean
pcv.params.sample_label = 'Endocarpo'
analysis_endo = pcv.analyze.color(img, labeled_mask=e_step,n_labels=1)
pcv.params.sample_label = 'Mesocarpo'
analysis_meso = pcv.analyze.color(img, labeled_mask=m_step,n_labels=1)
endo_hue_mean = pcv.outputs.observations['Endocarpo_1']['hue_circular_mean']['value']
meso_hue_mean = pcv.outputs.observations['Mesocarpo_1']['hue_circular_mean']['value']
 	## Access data stored out from analyze_color and remove NA's
if np.isnan(endo_hue_mean):  
    endo_hue_mean = 0  
    ## Filter on area- pixels on the mesocarp
mesocarp_area = np.count_nonzero(m_step)

# STEP 3.2 : Loop for the classification of masks based on color of guava
 # by defining hue ranges for endocarps and for mesocarps
pink_endocarp_lower, pink_endocarp_upper = 2.49, 30
green_endocarp_lower, green_endocarp_upper = 0, 2.5
green_endocarp_lower2, green_endocarp_upper2 = 35, 65
pink_mesocarp_lower, pink_mesocarp_upper = 25, 40.99
green_mesocarp_lower, green_mesocarp_upper = 41, 65

	## For mesocarp green endocarp pink
if (pink_endocarp_lower <= endo_hue_mean <= pink_endocarp_upper) and (green_mesocarp_lower <= meso_hue_mean <= green_mesocarp_upper) and (mesocarp_area > 200000):
    s_step = pcv.fill_holes(bin_img=mask['seeds'])
    s_step = pcv.erode(s_step, ksize=2, i=2)
    s_step = pcv.fill(bin_img=s_step, size=150)
    s_step2 = pcv.dilate(s_step, ksize=3, i=1)
    lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
    color_end_mes = "rosa.verde"
	## For mesocarp green endocarp green
elif((green_endocarp_lower <= endo_hue_mean <= green_endocarp_upper) or (green_endocarp_lower2 <= endo_hue_mean <= green_endocarp_upper2)) and (green_mesocarp_lower <= meso_hue_mean <= green_mesocarp_upper):
    s_step = pcv.dilate(mask['seeds'], ksize=5, i=1)
    s_step = pcv.fill(bin_img=s_step, size=150)
    s_step2 = pcv.fill_holes(bin_img=s_step)
    lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
    color_end_mes = "verde.verde"
	## For mesocarp pink endocarp pink (mesocarp area small)
elif(3 <= endo_hue_mean <= 12.9) and ((pink_mesocarp_lower <= meso_hue_mean <= pink_mesocarp_upper) or (mesocarp_area <= 140000)):
    s_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    s_thresh = pcv.threshold.binary(gray_img=s_img, threshold=155, object_type='light')
    s_step = pcv.filters.obj_props(s_thresh, cut_side="upper", thresh=0.45, regprop="solidity")
    s_step = pcv.dilate(s_step, ksize=2, i=1) 
    s_step = pcv.erode(s_step, ksize=2, i=1)
    s_step = pcv.fill(bin_img=s_step, size=300)
    s_step2 = pcv.fill_holes(bin_img=s_step)
    lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
    color_end_mes = "rosa.rosa"
elif(13 <= endo_hue_mean <= 15.29) and ((pink_mesocarp_lower <= meso_hue_mean <= pink_mesocarp_upper) or (mesocarp_area <= 140000)):
    s_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    s_thresh = pcv.threshold.binary(gray_img=s_img, threshold=158, object_type='light')
    s_step = pcv.filters.obj_props(s_thresh, cut_side="upper", thresh=0.45, regprop="solidity")
    s_step = pcv.erode(s_step, ksize=2, i=1)
    s_step = pcv.fill(bin_img=s_step, size=300)
    s_step2 = pcv.fill_holes(bin_img=s_step)
    lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
    color_end_mes = "rosa.rosa"
elif(15.3 <= endo_hue_mean <= 30) and ((pink_mesocarp_lower <= meso_hue_mean <= pink_mesocarp_upper) or (mesocarp_area <= 140000)):
    s_img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s_img, threshold=140, object_type='light')
    s_step = pcv.erode(s_thresh, ksize=2, i=1)
    s_step = pcv.filters.obj_props(s_step, cut_side="upper", thresh=0.52, regprop="solidity")
    s_step = pcv.fill(bin_img=s_step, size=380)
    s_step2 = pcv.fill_holes(bin_img=s_step)
    lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
    color_end_mes = "rosa.rosa"
	## For guava's with very light colors
else:
    if (endo_hue_mean <= 8.59) and (green_mesocarp_lower <= meso_hue_mean <= green_mesocarp_upper): 
        color_end_mes = "rosa.verde" 
        s_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
        s_thresh = pcv.threshold.binary(gray_img=s_img, threshold=152, object_type='light')    
        s_step = pcv.filters.obj_props(s_thresh, cut_side="upper", thresh=0.45, regprop="solidity")
        s_step = pcv.erode(s_step, ksize=2, i=1)
        s_step = pcv.fill(bin_img=s_step, size=480)
        s_step2 = pcv.fill_holes(bin_img=s_step)
        lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
    elif (endo_hue_mean >= 8.6) and (green_mesocarp_lower <= meso_hue_mean <= green_mesocarp_upper): ### CHANGED ELIF 
        color_end_mes = "rosa.verde" 
        s_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
        s_thresh = pcv.threshold.binary(gray_img=s_img, threshold=159, object_type='light') 
        s_step = pcv.filters.obj_props(s_thresh, cut_side="upper", thresh=0.45, regprop="solidity")
        s_step = pcv.erode(s_step, ksize=2, i=1)
        s_step = pcv.fill(bin_img=s_step, size=350) 
        s_step2 = pcv.fill_holes(bin_img=s_step)
        lab_mask, num_seeds = pcv.create_labels(mask=s_step2)
if 's_step2' not in locals(): ### ADDED **
    print("⚠️ ADVERTENCIA: s_step2 no se definió.")
    print(args.image1)
    print(f"endo_hue_mean: {endo_hue_mean}, meso_hue_mean: {meso_hue_mean}")
    raise RuntimeError("Ninguna condición del bloque 'else' se cumplió, revisar imagen y umbrales.")

# STEP 4.1 : Post-processing for mesocarp and endocarp masks defined by circumference of seeds
 	## Find contours of the seeds
fruit_image = cv2.imread(args.image1)
contours, _ = cv2.findContours(s_step2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 	## Collect all contour points into one list
all_points = np.vstack([contour.squeeze() for contour in contours if contour.shape[0] > 1])
	## Ensure all_points is 2D (sometimes contours can be odd-shaped)
if len(all_points.shape) == 1:
    all_points = np.array([all_points])
    ## Find minimum circle from seed'points array
(center_x, center_y), radius = cv2.minEnclosingCircle(all_points)
center = (int(center_x), int(center_y))
 	## Doing binary masks 
height, width = fruit_image.shape[:2]
 	## Create a blank image for the endocarp mask
e_step = np.zeros((height, width), dtype=np.uint8)
 	## Fill the convex hull to create the endocarp mask
cv2.circle(e_step, center, int(radius), 255, thickness=-1)  # filled white circle
 	## Create the mesocarp mask by inverting the endocarp mask but with background
m_step = cv2.bitwise_and(fruit_mask, cv2.bitwise_not(e_step))

	## Creating and saving image with applied post-processed masks 
masked_img = pcv.visualize.colorize_masks(masks=[m_step, e_step, s_step2], colors=['dark green', 'red', 'gold'])
image_name = "./outputs/mask-images/" + extractName(args.image1) + "_masked.jpg"
pcv.print_image(masked_img,image_name)

# STEP 5.1 : Feature extraction by calculation of interested traits through PlantCV function
 # the results will be in pixels , and the conversion scale is set at 119 pixels per cm
pcv.params.sample_label = 'Fruto'
analysis_image = pcv.analyze.size(img=img, labeled_mask=fruit_mask,n_labels=1)
pcv.params.sample_label = 'Endocarpo'
analysis_endocarp = pcv.analyze.size(img= img, labeled_mask=e_step,n_labels=1)

# STEP 5.2 : Feature extraction by calculation of interested traits through pixel counting
mesocarp_area = np.count_nonzero(m_step)
endocarp_area = np.count_nonzero(e_step)
seed_area = np.count_nonzero(s_step2)
ratio_seeds = seed_area / endocarp_area
ratio_endocarp = endocarp_area / mesocarp_area

    ## STEP 5.2.1 : Estimation of mesocarp width (in pixels) by calculating the mean between 4 euclidean distances -top,bottom,left and on right positions- from endocarp to fruit perimeter
    ## Find contours
contours_outer, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_inner, _ = cv2.findContours(e_step, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ## Assume the largest contour is the main one
outer_contour = max(contours_outer, key=cv2.contourArea)
inner_contour = max(contours_inner, key=cv2.contourArea)
    ## Function to get extreme points
def get_extreme_points(contour):
    contour = contour[:, 0, :]  # reshape (N, 1, 2) -> (N, 2)
    top = contour[np.argmin(contour[:, 1])]
    bottom = contour[np.argmax(contour[:, 1])]
    left = contour[np.argmin(contour[:, 0])]
    right = contour[np.argmax(contour[:, 0])]
    return top, bottom, left, right
    ## Get extreme points for both contours
top_outer, bottom_outer, left_outer, right_outer = get_extreme_points(outer_contour)
top_inner, bottom_inner, left_inner, right_inner = get_extreme_points(inner_contour)
    ## Compute Euclidean distances between corresponding points
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

d_top = euclidean(top_outer, top_inner)
d_bottom = euclidean(bottom_outer, bottom_inner)
d_left = euclidean(left_outer, left_inner)
d_right = euclidean(right_outer, right_inner)
    ## Compute mean width
mean_width = np.mean([d_top, d_bottom, d_left, d_right])

# STEP 6.1 : Saving manually made formulas to custom outputs 
pcv.outputs.add_observation(sample= "Endocarpo_1", variable='ratio',
                            trait='ratio of endocarp area per mesocarp area',
                            method='ratio of pixels', scale='percent', datatype=float,
                            value=ratio_endocarp, label='percent')
pcv.outputs.add_observation(sample= "Semillas_1", variable='num', trait='number of seeds',
                            method='units', scale='units', datatype=int,
                            value=num_seeds, label='units')
pcv.outputs.add_observation(sample= "Semillas_1", variable='area', 
                            trait='seed area in pixels',
                            method='pixel count', scale='pixels', datatype=float,
                            value=seed_area, label='pixels')
pcv.outputs.add_observation(sample= "Semillas_1", variable='ratio', 
                            trait='ratio of seed area per endocarp area',
                            method='ratio of pixels', scale='percent', datatype=float,
                            value=ratio_seeds, label='percent')
pcv.outputs.add_observation(sample="Endocarpo_1",variable='color', 
                            trait='type of color',
                            method='hue_value', scale='color', datatype=str,
                            value=color_end_mes, label='endo.meso')
pcv.outputs.add_observation(sample="Endocarpo_1",variable='thickness', #mesocarp width named change in script_clean_data.py
                            trait='mesocarp width',
                            method='euclidean distance', scale='pixels', datatype=float,
                            value=mean_width, label='pixels')


# STEP 6.2 : Saving data, each image is saved individually,
 # clear out the observations so the next loop adds new ones.
pcv.outputs.save_results(filename=args.result) 
pcv.outputs.clear()