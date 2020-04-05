#!/usr/bin/env bash
conda create --name translation python=3.7
conda activate translation
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch