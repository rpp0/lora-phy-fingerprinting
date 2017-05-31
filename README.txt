Readme for reproducibility submission of paper ID 41


A) Source code info
Repository: https://github.com/rpp0/lora-phy-fingerprinting
List of Programming Languages: Python
Compiler Info: N/A
Packages/Libraries Needed: python2-tensorflow, python2-scipy, python2-colorama, python2-matplotlib, python2-scikit-learn, python2-pymongo, python2-configparser, mongodb, cuda, cudann

B) Datasets info
Repository: MongoDB raw data included in VM, imported from DOI 10.5281/zenodo.583965
Data generators: N/A

C) Hardware Info
No special hardware is required on order to reproduce the results from the paper. However, a modern laptop / desktop can speed up the training process should the reproducibility committee desire to retrain all models. Memory usage can be high depending on which model is trained. For this reason (and because retraining can take up to a day), we have provided pre-trained models as well. See section D for the usage instructions.
C1) Intel(R) Xeon(R) CPU E5-1620 v2 @ 3.70GHz: configuration: cores=4 enabledcores=4 threads=8
C2) 256KiB L1, 1MiB L2, 10MiB L3
C3) 16GiB DIMM DDR3 1866 MHz (0.5 ns)
C4) 229G SSD
C5) N/A
C6) N/A
C7) Radio traces captured with an Ettus USRP B210 (stored as complex floats (raw) in MongoDB database).

D) Experimentation Info

VM login
user: wisec
pass: wisec

The VM needs at least 8 GB of memory in order to run the experiments!

D1) Scripts and how-tos to generate all necessary data or locate datasets
Data preinstalled on VM. Imported by performing `mongoimport --gzip .` inside the extracted Zenodo dataset (10.5281/zenodo.583965) folder.

D2) Scripts and how-tos to prepare the software for system
Software preinstalled on VM. Imported by performing `git clone https://github.com/rpp0/lora-phy-fingerprinting .` in the /home/wisec/lora-phy-fingerprinting folder.

D3) Scripts and how-tos for all experiments executed for the paper
See: ./runExperiments.sh

E) Additional notes
- The Results_1 and Results_2 folders contain results not mentioned in tables or figures.

- Each run of tf_train.py will parse the raw samples in the MongoDB and perform normalization, fft, etc. for each symbol. Therefore, running the experiments could take a while. To reduce the time taken, all samples were pre-randomized in the database and the models are pre-trained.

- The committee can - if they have the time - execute the "./tf_train train conf/<dataset> --save" command to retrain a model. All models were trained for 10,000 epochs, which took about 2-3 hours per model (so about 12 hours / full working day in total). The ./runExperiments.sh files can be executed after training to obtain identical results to the paper.

- *Warning: this action will permanently modify the dataset. Please take a snapshot of the VM if you want to perform this step* The committee can also uncomment "rm -f /tmp/randomized_mongo" in the ./runExperiments scripts to re-randomize the dataset before each run. This will significantly increase the required time to test, and will result in slightly different results (+- 1 percent differences in accuracy) since the models will be trained on other data from the set. The randomization in MongoDB is done by storing a float from 0 to 1 as a field alongside the data. Therefore, re-randomizing will irreversibly modify the training and test samples order used in the experiments.

- If you get a timeout exception from PyMongo: normally shouldn't happen, but it could occur due to the VM running out of memory (e.g. if you run multiple scripts at the same time), crashing MongoDB. You can try to do "sudo systemctl restart mongodb" to restart Mongo. Only one experiment should be executed at a time.
