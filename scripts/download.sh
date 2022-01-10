#!/usr/bin/env bash
FILE=$1

if [ $FILE == "moco400" ]; then
    URL=https://1drv.ms/u/s!AtvUxcft_YQ-g-kV29FAUS-f8hWecg?e=OQtPKa
    ZIP_FILE=./saved_ckpt/moco400.zip
    mkdir -p ./saved_ckpt/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./saved_ckpt/
    rm $ZIP_FILE

elif [ $FILE == "moco100" ]; then
    URL=https://1drv.ms/u/s!AtvUxcft_YQ-g-kUXSn_CX7o_nn5Yw?e=54u59f
    ZIP_FILE=./saved_ckpt/moco100.zip
    mkdir -p ./saved_ckpt/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./saved_ckpt/
    rm $ZIP_FILE

fi