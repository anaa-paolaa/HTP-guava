#!/usr/bin/env python
# coding: utf-8
# Cleaning data : Guava's image analysis results
# Ana VALLADARES

# Libraries 
import numpy as np
import pandas as pd
import os

def rename_cm2(columns):
        return [col.replace('area_pixels_', 'area_cm2_') if 'area_pixels_' in col else col for col in columns]

def rename_cm(columns):
        return [col.replace('pixels_', 'cm_') if 'pixels_' in col else col for col in columns]

    
def main():
    # Import data .csv change path if necesary 
    res = pd.read_csv("outputs/raw_results-single-value-traits.csv")
    
    # Remove extra files if real raw format is not interested output.json
    files = ['outputs/raw_results-multi-value-traits.csv','outputs/output.json']
    for file in files:
        if(os.path.exists(file) and os.path.isfile(file)): 
          os.remove(file) 

    to_drop=['camera','imgtype','zoom','exposure','gain','frame',
    'rotation','lifter','treatment','cartag','measurementlabel','other',
    'image','object_in_frame_none','in_bounds_none','ellipse_eccentricity_none',
    'ellipse_major_axis_pixels','ellipse_minor_axis_pixels',
    'convex_hull_area_pixels','convex_hull_vertices_none',
    'ellipse_angle_degrees','hue_circular_mean_degrees','hue_circular_std_degrees','hue_median_degrees',
    'in_bounds_none','longest_path_pixels','object_in_frame_none','solidity_none']

    res.drop(columns=to_drop, inplace=True)

    to_drop = [col for col in res.columns if 'NA' in col]
    res.drop(columns=to_drop, inplace=True)


    res['sample'] = res['sample'].str.replace(r'_\d+', '', regex=True)

    # Melt the DataFrame
    melted_df = res.melt(id_vars=['id','barcode','timestamp', 'sample'], var_name='traits', value_name='value')

    # Pivot the DataFrame
    pivot_df = melted_df.pivot_table(index=['id','barcode','timestamp'], columns=['traits', 'sample'], values='value', aggfunc='first')

    # Flatten the MultiIndex columns and rename them
    pivot_df.columns = [f'{traits}_{attr.lower()}' for traits, attr in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    #From pixels to 1cm2 = 14161 px
    cols_list = [col for col in pivot_df.columns if 'area_pixels' in col]
    pivot_df[cols_list] = pivot_df[cols_list]/14161 #cm2

    # Apply the function to the DataFrame's columns
    pivot_df.columns = rename_cm2(pivot_df.columns)

    # From pixels 1 cm = 119 px and 
    cols_list = [col for col in pivot_df.columns if 'pixels' in col]
    pivot_df[cols_list] = pivot_df[cols_list]/119 #cm

    # Apply the function to the DataFrame's columns
    pivot_df.columns = rename_cm(pivot_df.columns)
    
    # Rename column color_endo.meso_endocarpo
    pivot_df = pivot_df.rename(columns={'color_endo.meso_endocarpo': 'color_endo.meso'})
    
    # Rename column thickness_cm_endocarpo, originally measured from mesocarpo
    pivot_df = pivot_df.rename(columns={'thickness_cm_endocarpo': 'width_cm_mesocarpo'})
    
    # Añadir area del mesocarpo
    pivot_df['area_cm2_mesocarpo'] = pivot_df['area_cm2_fruto'] - pivot_df['area_cm2_endocarpo']
    
    # Alphabetically re-order df
    first_cols = list(pivot_df.columns[:4])
    remaining_cols = list(pivot_df.columns[4:])
    remaining_cols.sort() #alphabetically ordered
    ordered_cols = first_cols + remaining_cols

    # Save new df
    res_new = pivot_df[ordered_cols]
    
    # Ask the name for .csv file
    output_name = input("How to name the cleaned file (without- .csv): ")
    if not output_name.endswith(".csv"):
        output_name += ".csv"

    # Save with new name
    path_csv = "./outputs/" + output_name
    res_new.to_csv(path_csv, index=False)

    print(f"Raw data has been cleaned and saved as outputs/{output_name}")

if __name__ == "__main__":
    main()