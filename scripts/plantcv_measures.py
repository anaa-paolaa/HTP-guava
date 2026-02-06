#!/usr/bin/env python
"""
#**PlantCV workflow for image analysis in Guava**
Author : Ana Valladares
"""

# Libraries
import numpy as np
import argparse
import pandas as pd
import os
from plantcv import plantcv as pcv
import plantcv.utils
from plantcv import parallel
from plantcv.plantcv.filters import obj_props
from plantcv.parallel import workflow_inputs
import cv2
import json
import scipy
from scipy.spatial.distance import euclidean

# Funtion to extract the name of the image
def extractName(a):
    base=os.path.basename(a)
    return os.path.splitext(base)[0]

# Functions to check concentricy of endocarp with fruit mask
def centroid_from_mask(mask):
    if mask is None:
        return None
    if np.count_nonzero(mask) == 0:
        return None

    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return None

    return np.array([
        int(M["m10"] / M["m00"]),
        int(M["m01"] / M["m00"])
    ])

def get_seed_regions(seed_mask, fruit_center):
    num, labels = cv2.connectedComponents(seed_mask)
    seeds = []

    for i in range(1, num):
        region = (labels == i).astype(np.uint8) * 255
        c = centroid_from_mask(region)
        if c is None:
            continue
        d = euclidean(c, fruit_center)
        seeds.append((d, region))

    seeds.sort(reverse=True, key=lambda x: x[0])  # external seed
    return seeds

def endocarp_from_seeds(seed_regions, shape):
    combined = np.zeros(shape, dtype=np.uint8)
    for _, r in seed_regions:
        combined |= r

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    pts = np.vstack(cnts)
    (center_x, center_y), radius = cv2.minEnclosingCircle(pts)
    center = (int(center_x), int(center_y))

    endo = np.zeros(shape, dtype=np.uint8)
    cv2.circle(endo, center, int(radius), 255, thickness=-1)
    return endo

def concentricity(endocarp_mask, fruit_mask):
    if endocarp_mask is None or fruit_mask is None:
        return np.inf

    fruit_center = centroid_from_mask(fruit_mask)
    endo_center = centroid_from_mask(endocarp_mask)

    if fruit_center is None or endo_center is None:
        return np.inf

    cnts, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack(cnts)
    fruit_radius = np.linalg.norm(pts - fruit_center, axis=1).max()
    
    return euclidean(fruit_center, endo_center) / fruit_radius

def refine_by_removing_one_seed(seed_mask, fruit_mask, max_conc, image_name=None):

    fruit_center = centroid_from_mask(fruit_mask)
    if fruit_center is None:
        print(f"⚠️ Fruit centroid missing: {image_name}")
        return seed_mask, None, 0

    seeds = get_seed_regions(seed_mask, fruit_center)
    
    endo_orig = endocarp_from_seeds(seeds, seed_mask.shape)
    
    if endo_orig is None:
        print(f"⚠️ Endocarp could not be computed (original): {image_name}")
        return seed_mask, None, len(seeds)

    if len(seeds) < 3:
        return seed_mask, endo_orig, len(seeds)

    conc_orig = concentricity(endo_orig, fruit_mask)

    if conc_orig < max_conc:
        return seed_mask, endo_orig, len(seeds)

    # Remove ONE external seed
    seeds_refined = seeds[1:]
    endo_new = endocarp_from_seeds(seeds_refined, seed_mask.shape)

    if endo_new is None:
        print(f"⚠️ Endocarp could not be computed (refined): {image_name}")
        return seed_mask, endo_orig, len(seeds)

    conc_new = concentricity(endo_new, fruit_mask)

    if conc_new < conc_orig:
        refined_seed_mask = np.zeros_like(seed_mask)
        for _, r in seeds_refined:
            refined_seed_mask |= r

        return refined_seed_mask, endo_new, len(seeds_refined)

    return seed_mask, endo_orig, len(seeds)


def compute_mesocarp(fruit_mask, endocarp_mask):
    if endocarp_mask is None:
        return fruit_mask.copy()

    meso = fruit_mask.copy()
    meso[endocarp_mask > 0] = 0
    return meso

# Initialize instance of options called args to mimic the use of argparse in workflow script
args = workflow_inputs()
pcv.params.debug = args.debug
    
# Read in image
img, imgpath, imgname = pcv.readimage(filename=args.image1)

print("PROCESSING:", imgname)

# STEP 1.1: Load seed_mask from `process_seed.py` DL pipeline
mask_dir = "/content/work/outputs/temp/seed_masks"
image_name = extractName(args.image1) + "_seed.png"
mask_path = os.path.join(mask_dir, image_name)
seed_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# STEP 1.2: Use the output file from `plantcv-train.py` to run the multiclass naive bayes to get background substraction 
mask = pcv.naive_bayes_classifier(rgb_img=img, pdf_file="/content/work/docs/Bayes.txt")

# STEP 2.1: Post-processing of background,meso and endo mask to fill objects and dilate to separate from salt noise
 # invert to get fruit_mask
b_step = pcv.fill(bin_img=mask['background'], size=50)
b_step = pcv.dilate(b_step, ksize=9, i=1)
fruit_mask = pcv.invert(b_step)

    # Checks for seed_mask if empty
if seed_mask is None or np.count_nonzero(seed_mask) == 0:
    print(f"⚠️ No seeds detected: {extractName(args.image1)}")

    num_seeds = 0
    endocarp_mask = None
    mesocarp_mask = None

    debug_mask = np.zeros_like(fruit_mask, dtype=np.uint8)
    debug_mask[fruit_mask > 0] = 50

    image_name = extractName(args.image1) + "_mask.png"
    outdir = "/content/work/outputs/mask-images"
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(
        os.path.join(outdir, image_name),
        debug_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )

    # Save NA - outputs
    pcv.outputs.add_observation("Semillas_1", "num", "number of seeds",
                                "units", "units", int, np.nan, "units")
    pcv.outputs.add_observation(sample= "Semillas_1", variable='area', 
                            trait='seed area in pixels',
                            method='pixel count', scale='pixels', datatype=float,
                            value=np.nan, label='pixels')
    pcv.outputs.add_observation("Endocarpo_1", "area", "endocarp area",
                                "pixel count", "pixels", float, np.nan, "pixels")

    pcv.outputs.save_results(filename=args.result)
    pcv.outputs.clear()
    quit()

if seed_mask is not None and seed_mask.shape != fruit_mask.shape:
    seed_mask = cv2.resize(
        seed_mask,
        (fruit_mask.shape[1], fruit_mask.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

# STEP 3.1 : Counting of seeds 
lab_mask, num_seeds = pcv.create_labels(mask=seed_mask)

# STEP 3.2 : Post-processing for mesocarp and endocarp masks defined by circumference of seeds with concentricity-check
seed_mask, endocarp_mask, num_seeds = refine_by_removing_one_seed(
    seed_mask,
    fruit_mask,
    max_conc=0.11,
    image_name=extractName(args.image1)
)

mesocarp_mask = compute_mesocarp(fruit_mask, endocarp_mask) 
e_step = endocarp_mask
m_step = mesocarp_mask

# STEP 3.3 : Re-counting of seeds if affected from step before
lab_mask, num_seeds = pcv.create_labels(mask=seed_mask)

# STEP 4.1 : Analyze color for endocarp and mesocarp and stored Hue circle value mean
pcv.params.sample_label = 'Endocarpo'
analysis_endo = pcv.analyze.color(img, labeled_mask=e_step,n_labels=1)
pcv.params.sample_label = 'Mesocarpo'
analysis_meso = pcv.analyze.color(img, labeled_mask=m_step,n_labels=1)
endo_hue_mean = pcv.outputs.observations['Endocarpo_1']['hue_circular_mean']['value']
meso_hue_mean = pcv.outputs.observations['Mesocarpo_1']['hue_circular_mean']['value']

# STEP 4.2 : Loop for the color classification
 # by defining hue ranges for endocarps and for mesocarps
PINK_ENDO = [(330, 360), (0, 32.99)]
GREEN_ENDO = [(33, 160)]
PINK_MESO = [(330, 360), (0, 40.99)]
GREEN_MESO = [(41, 160)]

def hue_in_ranges(hue, ranges):
    for low, high in ranges:
        if low <= hue <= high:
            return True
    return False

if np.isnan(endo_hue_mean) or np.isnan(meso_hue_mean):
    color_end_mes = "NA"
elif hue_in_ranges(endo_hue_mean, PINK_ENDO) and hue_in_ranges(meso_hue_mean, GREEN_MESO):
    color_end_mes = "rosa.verde"
elif hue_in_ranges(endo_hue_mean, GREEN_ENDO) and hue_in_ranges(meso_hue_mean, GREEN_MESO):
    color_end_mes = "verde.verde"
elif hue_in_ranges(endo_hue_mean, PINK_ENDO) and hue_in_ranges(meso_hue_mean, PINK_MESO):
    color_end_mes = "rosa.rosa"
else:
    color_end_mes = "NA"

	## Creating and saving image with applied post-processed masks 
# Create lightweight grayscale mask
debug_mask = np.zeros_like(seed_mask, dtype=np.uint8)

debug_mask[m_step > 0] = 50       # mesocarp
debug_mask[e_step > 0] = 150      # endocarp
debug_mask[seed_mask > 0] = 255   # seeds

image_name = extractName(args.image1) + "_mask.png"
outdir = "/content/work/outputs/mask-images"
os.makedirs(outdir, exist_ok=True)

cv2.imwrite(
    os.path.join(outdir, image_name),
    debug_mask,
    [cv2.IMWRITE_PNG_COMPRESSION, 9]  
)


# STEP 5.1 : Feature extraction by calculation of interested traits through PlantCV function
 # the results will be in pixels
pcv.params.sample_label = 'Fruto'
analysis_image = pcv.analyze.size(img=img, labeled_mask=fruit_mask,n_labels=1)
pcv.params.sample_label = 'Endocarpo'
if e_step is not None:
    analysis_endocarp = pcv.analyze.size(img= img, labeled_mask=e_step,n_labels=1)

# STEP 5.2 : Feature extraction by calculation of interested traits through pixel counting
if e_step is None or np.count_nonzero(e_step) == 0:
    endocarp_area = np.nan
    mesocarp_area = np.nan
    ratio_seeds = np.nan
    ratio_endocarp = np.nan
    mean_width = np.nan
else:
    endocarp_area = np.count_nonzero(e_step)
    mesocarp_area = np.count_nonzero(m_step)
    seed_area = np.count_nonzero(seed_mask)
    ratio_seeds = seed_area / endocarp_area if endocarp_area > 0 else np.nan
    ratio_endocarp = endocarp_area / mesocarp_area if mesocarp_area > 0 else np.nan
                                                                       
## STEP 5.2.1 : Estimation of mesocarp width (in pixels) by calculating the mean between 4 euclidean distances -top,bottom,left and on right positions- from endocarp to fruit perimeter
    ## Find contours
contours_outer, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if e_step is None:
    contours_inner = []
else:
    contours_inner, _ = cv2.findContours(e_step, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                                                       
    ## Assume the largest contour is the main one
outer_contour = max(contours_outer, key=cv2.contourArea)
inner_contour = max(contours_inner, key=cv2.contourArea)
    ## Function to get extreme points
def get_extreme_points(contour):
    contour = contour[:, 0, :]
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