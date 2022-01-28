#!/bin/bash
module load anaconda/2020.11
source activate torch1.4
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0941.HK.all.csv --col_name_tar Close --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.1288.HK.all.csv --col_name_tar Close --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/Stk.0005.HK.all.csv --col_name_tar Close --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi2.csv --col_name_tar humidity --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi3.csv --col_name_tar humidity --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi4.csv --col_name_tar humidity --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/gnfuv-pi5.csv --col_name_tar humidity --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/hungaryChickenpox.csv --col_name_tar BUDAPEST --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-1.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/NEW-DATA-2.T15.csv --col_name_tar 3:Temperature_Comedor_Sensor --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --name_model TSA_uni_naive --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --name_model TSA_uni_avg --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --name_model TSA_uni_mov_win --n_epochs 200 --shuffle --lstm_hid_size 10 --weight_decay 1e-8 --early_stopping --patience 10
