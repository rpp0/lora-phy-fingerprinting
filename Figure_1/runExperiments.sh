#!/bin/sh

echo "Generating Figure 1..."
cd ..
#rm -f /tmp/randomized_mongo
./tf_train.py test conf/experiment_2msps_mlp.cfg
cp /tmp/tf_2msps_mlp-w.pdf ./Figure_1/
echo "Done, please see tf_2msps_mlp-w.pdf in this folder."
