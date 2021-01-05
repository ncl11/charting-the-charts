import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from df2gspread import df2gspread as d2g

def get_spreadsheet():  # adapted from https://www.techwithtim.net/tutorials/google-sheets-python-api-tutorial/
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope) 
    client = gspread.authorize(creds)
    sheet = client.open("3_mo_weekly").sheet1  
    data = sheet.get_all_records()  
    df = pd.DataFrame(data)
    return df

def save_spreadsheet(df):
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
    spreadsheet_key = '1Lrz2J2vb-OFlpB-ld6gs3wq-ducnb4QkZzEl_Xm0ZSA'
    name = '3_mo_weekly'
    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope) 
    d2g.upload(df, spreadsheet_key, name, credentials=creds, row_names=False)


