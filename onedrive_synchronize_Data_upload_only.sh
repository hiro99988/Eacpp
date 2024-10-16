#!/bin/sh

mv out/data/Moead/* ~/OneDrive/HaradaLab/Data/Moead/
mv out/data/MpMoead/* ~/OneDrive/HaradaLab/Data/MpMoead/
onedrive --synchronize --single-directory --upload-only "HaradaLab/Data/"