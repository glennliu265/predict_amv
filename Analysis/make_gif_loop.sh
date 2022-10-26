#!/bin/bash

for i in /Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20221014/TP/*/; do
    echo "${i}animation.gif"
    convert -delay 150 -loop 0 ${i}*.png "${i}animation.gif"
done