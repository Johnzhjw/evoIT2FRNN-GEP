#!/bin/bash
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model lstm --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model cnn --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model lstm --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model cnn --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_naive --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_avg --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_mov_win --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model lstm --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model cnn --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model lstm --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model cnn --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_naive --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_avg --n_epochs 200 --shuffle
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_mov_win --n_epochs 200 --shuffle