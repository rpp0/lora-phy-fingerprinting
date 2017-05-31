#!/bin/sh

echo "Generating results from section 3.2.4 for D1, D2, D3 (this will take a while)..."
rm -f /tmp/results1.tex
touch /tmp/results1.tex

experiments=(
    "distance_experiment_train_meeting_test_meeting.cfg"
    "distance_experiment_train_lower_test_lower.cfg"
    "distance_experiment_train_printer_test_printer.cfg"
    "distance_experiment_train_meeting_test_lower.cfg"
    "distance_experiment_train_printer_test_lower.cfg"
)

cd ..
for i in ${experiments[@]}
do
    #rm -f /tmp/randomized_mongo
    sudo systemctl restart mongodb  # Fix caching memory issue
    ./tf_train.py test conf/$i
    cat /tmp/tmp_result >> /tmp/results1.tex
done
cp /tmp/results1.tex ./Results_1/results1.tex
echo "Done, please see results1.tex in this folder."
echo "Results given in order of appearance in the paper."
echo "Only accuracy (first column) was reported."
