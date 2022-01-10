#!/usr/bin/env bash
FILE=$1

if [ $FILE == "moco" ]; then
    URL=https://1drv.ms/u/s!AtvUxcft_YQ-g-kV29FAUS-f8hWecg?e=OQtPKa
    ZIP_FILE=./saved_ckpt/moco400.zip
    mkdir -p ./saved_ckpt/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./saved_ckpt/
    rm $ZIP_FILE

 elif [ $FILE == "imagenette2" ]; then
    URL=https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
    ZIP_FILE=./saved_ckpt/imagenette2.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE


fi