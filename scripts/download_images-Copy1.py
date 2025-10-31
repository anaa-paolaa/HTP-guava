#!/usr/bin/env python
# coding: utf-8

import os
import re
import io
from datetime import datetime
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

# Get folder's date
def parse_date_from_folder(name):
    match = re.search(r'(\d{2})-(\d{2})-(\d{2})', name)
    if match:
        return datetime.strptime(match.group(0), "%d-%m-%y")
    return None

# Get folders in date range
def get_folders_in_date_range(drive, start_date, end_date):
    folders_in_range = []
    page_token = None

    while True:
        results = drive.files().list(
            q="mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        folders = results.get('files', [])
        for folder in folders:
            folder_date = parse_date_from_folder(folder['name'])
            if folder_date and start_date <= folder_date <= end_date:
                folders_in_range.append((folder['id'], folder['name']))

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    return folders_in_range

# Function to get the folder name from its ID****
def get_folder_name(drive, folder_id):
    folder = drive.files().get(fileId=folder_id, fields='name').execute()
    return folder['name']

# Function to download images from a folder ***
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
           # N = 8
            #new_folder_name =  date
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

    # Ask user for start and end dates
    start_date_str = input("Enter start date (dd-mm-yy): ")
    end_date_str = input("Enter end date (dd-mm-yy): ")
    start_date = datetime.strptime(start_date_str, "%d-%m-%y")
    end_date = datetime.strptime(end_date_str, "%d-%m-%y")

    # Find folders within date range
    selected_folders = get_folders_in_date_range(drive, start_date, end_date)

    print("\n✅ Found folders in date range:")
    for fid, name in selected_folders:
        print(f"  - {name} (ID: {fid})")

    # You can now use `selected_folders` (list of tuples with ID and name)
    # in your later scripts directly to process images from Drive without downloading


if __name__ == "__main__":
    main()
