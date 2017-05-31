#!/bin/sh

echo "Generating Table 3 (this will take a while)..."
rm -f /tmp/table3.tex
touch /tmp/table3.tex

experiments=(
    "experiment_1msps_svm.cfg"
    "experiment_1msps_cnn.cfg"
    "experiment_1msps_mlp.cfg"
    "experiment_1msps_svm_aftertrain.cfg"
    "experiment_1msps_cnn_aftertrain.cfg"
    "experiment_1msps_mlp_aftertrain.cfg"
    "experiment_2msps_svm.cfg"
    "experiment_2msps_cnn.cfg"
    "experiment_2msps_mlp.cfg"
    "experiment_2msps_svm_aftertrain.cfg"
    "experiment_2msps_cnn_aftertrain.cfg"
    "experiment_2msps_mlp_aftertrain.cfg"
    "experiment_5msps_svm.cfg"
    "experiment_5msps_cnn.cfg"
    "experiment_5msps_mlp.cfg"
    "experiment_5msps_svm_aftertrain.cfg"
    "experiment_5msps_cnn_aftertrain.cfg"
    "experiment_5msps_mlp_aftertrain.cfg"
    "experiment_10msps_svm.cfg"
    "experiment_10msps_cnn.cfg"
    "experiment_10msps_mlp.cfg"
    "experiment_10msps_svm_aftertrain.cfg"
    "experiment_10msps_cnn_aftertrain.cfg"
    "experiment_10msps_mlp_aftertrain.cfg"
)

cd ..
for i in ${experiments[@]}
do
    #rm -f /tmp/randomized_mongo
    sudo systemctl restart mongodb  # Fix caching memory issue
    ./tf_train.py test conf/$i
    cat /tmp/tmp_result >> /tmp/table3.tex
done
cp /tmp/table3.tex ./Table_3/table3.tex
echo "Done, please see table3.tex in this folder."
