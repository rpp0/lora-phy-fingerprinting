#!/bin/sh

echo "Generating Table 4 (this will take a while)..."
rm -f /tmp/table4.tex
touch /tmp/table4.tex

experiments=(
    "experiment_2msps_mlp_zeroshot_intervendor.cfg"
    "experiment_2msps_mlp_zeroshot_intervendor2.cfg"
    "experiment_2msps_mlp_zeroshot_intravendor.cfg"
    "experiment_2msps_mlp_zeroshot_intravendor2.cfg"
    "experiment_2msps_mlp_zeroshot_intravendor3.cfg"
    "experiment_2msps_mlp_zeroshot_intravendor4.cfg"
)

cd ..
for i in ${experiments[@]}
do
    #rm -f /tmp/randomized_mongo
    sudo systemctl restart mongodb  # Fix caching memory issue
    ./tf_train.py zeroshot conf/$i
    cat /tmp/tmp_result >> /tmp/table4.tex
done
cp /tmp/table4.tex ./Table_4/table4.tex
echo "Done, please see table4.tex in this folder."
