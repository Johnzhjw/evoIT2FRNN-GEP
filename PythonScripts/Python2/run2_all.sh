#!/bin/bash
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --flag_multi --name_model lstm --n_epochs 200 --shuffle --lstm_hid_size 10
python main.py --file_name ../dataset/_TS/traffic.csv --col_name_tar Slowness_in_traffic --flag_multi --name_model cnn --n_epochs 200 --shuffle --lstm_hid_size 10
