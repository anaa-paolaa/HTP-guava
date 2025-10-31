#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from oauth2client import file, client, tools
from httplib2 import Http

# Function to authenticate and create the Google Drive client
def authenticate_drive():
    SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('./data/client_secrets.json', SCOPES)
        creds = tools.run_flow(flow, store)
    return build('drive', 'v3', http=creds.authorize(Http()))

## Get folder ID by name
def get_folder_id_by_name(drive, folder_name):
    response = drive.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
        spaces='drive',
        fields='files(id, name)',
    ).execute()

    folders = response.get('files', [])
    if not folders:
        print(f"No folder found with name: {folder_name}")
        return None
    return folders[0]['id']

# Function to get the folder name from its ID
def get_folder_name(drive, folder_id):
    folder = drive.files().get(fileId=folder_id, fields='name').execute()
    return folder['name']

# Function to download images from a folder
def download_images_from_folder(drive, folder_id, destination_folder):
    folder_name_input = get_folder_name(drive, folder_id)
    page_token = None
    
    while True:
        results = drive.files().list(
            q=f"'{folder_id}' in parents and (mimeType contains 'image/jpeg' or mimeType contains 'image/png') and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
    
        items = results.get('files', [])
        for item in items:
            file_id = item['id']
            N = 8
            new_folder_name =  folder_name_input[-N:]
            file_name = f"{new_folder_name}_{item['name']}"
            file_path = os.path.join(destination_folder, file_name)
            print(f'Downloading {item["name"]} as {file_name} to {file_path}')
            request = drive.files().get_media(fileId=file_id)
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f'Download {int(status.progress() * 100)}%.')
            
        page_token = results.get('nextPageToken')
        if not page_token:
            break


def main():
    drive = authenticate_drive()

    # Specify the folder IDs that contain the images, *CHANGE/ADD when need it*
    #folder_ids = [
    #    '1Eq9qdPJJJ90f96p-hfigM0zcy1wGrWHX',  #L7 TH 211222
    #    '1AZvcwnxBSQvCameAcr5pK5yWiiOkKEPG',  #L7 TH 231222
    #    '1rkmwemuHHzr2O0MXkzw8_HwWl6pkx8i8'   #L8 TH 311222
    #]
    
    ## Get the folder name from the user
    folder_name_input = input("Enter the folder names separated by commas: ")
    folder_names = [folder_name.strip() for folder_name in folder_name_input.split(",")]
    
    # Destination directory on the desktop
    ##* desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')  **For Windows
    desktop_path = os.path.join(os.path.expanduser('~'), '/Users/anapaola/Desktop/Guava/Inputs/')

    # Create a folder on the desktop to store the downloaded images
    destination_folder = os.path.join(desktop_path, 'Pool_images2')
    os.makedirs(destination_folder, exist_ok=True)

    # Download images from each specified folder
    #for folder_id in folder_ids:
    for folder_name in folder_names:
        print(f"Processing folder: {folder_name}")
            
        # Get the folder ID by name
        folder_id = get_folder_id_by_name(drive, folder_name)
        if not folder_id:
            continue

        download_images_from_folder(drive, folder_id, destination_folder)

    print('Download completed.')

if __name__ == "__main__":
    main()