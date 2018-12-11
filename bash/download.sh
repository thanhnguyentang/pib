#!/bin/bash
echo "Download mnist variants from"
echo "http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007"
DATA_DIR="../pib_run/data/mnist-variant/amat"
mkdir -p $DATA_DIR
echo $DATA_DIR

#dataset paper: http://www.iro.umontreal.ca/~lisa/twiki/pub/Public/DeepVsShallowComparisonICML2007/icml-2007-camera-ready.pdf
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip 
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip
wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip

echo "Unzipping ..."
for i in *.zip; do 
    j=$(echo $i| cut -d'.' -f 1) 
    mkdir "$DATA_DIR/$j"
    cd "$DATA_DIR/$j"
    unzip "../../../$i"
    cd ../../..
    rm -rf "$i" 
done
