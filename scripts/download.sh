#!/usr/bin/env bash
FILE=$1

if [ $FILE == "moco" ]; then
    URL="https://onedrive.live.com/download?cid=F025222A8799E567&resid=F025222A8799E567%21106&authkey=AKfcQ-VpY8HMUBQ"
    ZIP_FILE=./saved_ckpt/moco.ckpt
    mkdir -p ./saved_ckpt/
    wget --no-check-certificate -N $URL -O $ZIP_FILE

 elif [ $FILE == "imagenette2-320" ]; then
    URL=https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    ZIP_FILE=./saved_ckpt/imagenette2-320.tgz
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    tar zxvf $ZIP_FILE -C ./datasets/
    rm $ZIP_FILE


fi