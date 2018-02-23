#! /bin/bash
rm data -rf
mkdir data
sr333_url="http://users.itk.ppke.hu/~godma/PhD/fPCG/short_fPCG/333Hz.7z"
sr9600_url="http://users.itk.ppke.hu/~godma/PhD/fPCG/short_fPCG/9600Hz.7z"
wget -nc $sr9600 -O "data.7z"
