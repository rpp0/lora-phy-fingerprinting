#!/bin/sh

echo "Generating results for the claim 'When fingerprinting the 3 chipset vendors, the accuracy is 99% - 100% for all datasets and classifiers.'"
rm -f /tmp/results2.tex
touch /tmp/results2.tex

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
    ./tf_train.py test conf/$i --vendor
    cat /tmp/tmp_result >> /tmp/results2.tex
done
cp /tmp/results2.tex ./Results_2/results2.tex
echo "Done, please see results2.tex in this folder."
