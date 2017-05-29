#!/bin/sh

echo "Generating Table 2..."
rm -f /tmp/table2.tex
touch /tmp/table2.tex

echo "Dataset I (1 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps.find({}).count()' lora1msps >> /tmp/table2.tex

echo "Dataset II (1 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps_test.find({}).count()' lora1msps >> /tmp/table2.tex

echo "Dataset III (2 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps.find({}).count()' lora2msps >> /tmp/table2.tex

echo "Dataset IV (2 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps_test.find({}).count()' lora2msps >> /tmp/table2.tex

echo "Dataset V (5 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps.find({}).count()' lora5msps >> /tmp/table2.tex

echo "Dataset VI (5 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps_test.find({}).count()' lora5msps >> /tmp/table2.tex

echo "Dataset VII (10 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps.find({}).count()' lora10msps >> /tmp/table2.tex

echo "Dataset VIII (10 Msps)" >> /tmp/table2.tex
mongo --quiet --eval 'db.chirps_test.find({}).count()' lora10msps >> /tmp/table2.tex

cp /tmp/table2.tex .
echo "Done, please see table2.tex in this folder."
echo "Note that the IDs, sampling rate, and date were reformatted in the paper (from MongoDB) to make them more readable."
