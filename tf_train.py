#!/usr/bin/python2
# tf_train.py
#
# Collection of ML algorithms to fingerprint radio devices using Tensorflow.
# A high level overview of the functionality provided by this code is given in
# the paper entitled "Physical-Layer Fingerprinting of LoRa devices using
# Supervised and Zero-Shot Learning", which was presented at WiSec 2017. A VM
# containing the training data and scripts required to reproduce the results
# from our paper will be published on Zenodo. Please contact one of the authors
# for more information.
#
# The code provides an abstraction layer on top of Tensorflow, consisting of
# "Models" and "Layers", in order to build a "Classifier" for raw radio signals.
# If you plan on using this framework for your research, I would recommend using
# the library "Keras" to build the models instead of "raw" Tensorflow. Keras was
# developed concurrently with this work, and provides a more concise and mature
# implementation for the same types of models that are used here.
#
# Author: Pieter Robyns
# Contact: pieter.robyns@uhasselt.be

import tensorflow as tf
import colorama
import random
import numpy as np
import scipy.io as sio
import os
import configparser
import argparse
import preprocessing
import visualization
import pickle
import json
import sklearn
import utilities
from colorama import Fore,Back,Style
from pymongo import MongoClient
from pymongo.errors import OperationFailure, AutoReconnect
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from mapping import Mapping
from cache import GenericCache
from datetime import datetime
from random import randint
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.cluster import DBSCAN
from itertools import combinations
from collections import defaultdict

# ----------------------------------------------------
# Globals
# ----------------------------------------------------
colorama.init(autoreset=True)
EPSILON = 0.00000000001
defaults = {
    'exclude_classes': '',
    'epochs': -1,
    'num_zs_test_samples': 40,
}
cp = configparser.RawConfigParser(defaults)
flags = tf.app.flags
FLAGS = flags.FLAGS

# ----------------------------------------------------
# Static functions
# ----------------------------------------------------
def load_conf(conf):  # Configure the classifier using settings from conf file
    cp.read(conf)

    # Flags
    flags.DEFINE_string('logdir', '/tmp/tensorboard', 'Tensorboard summaries directory')
    flags.DEFINE_string('trainedmodelsdir', cp.get("DEFAULT", "trained_models_path"), 'Trained models directory')
    flags.DEFINE_string('dataset', cp.get("DEFAULT", "dataset"), 'Dataset type (mongo or matlab)')
    flags.DEFINE_string('classifier', cp.get("DEFAULT", "classifier"), 'Type of classifier to use')
    flags.DEFINE_string('clustering', cp.get("DEFAULT", "clustering"), 'Type of clustering to use if doing open set classification')
    flags.DEFINE_string('model_name', cp.get("DEFAULT", "model_name"), 'Name of the experiment / model. Used for saving it')
    flags.DEFINE_integer('limit', cp.getint("DEFAULT", "limit"), 'Limit input tensor to n samples')
    flags.DEFINE_integer('num_train_samples', cp.getint("DEFAULT", "num_train_samples"), 'Number of training samples')
    flags.DEFINE_integer('num_test_samples', cp.getint("DEFAULT", "num_test_samples"), 'Number of test samples')
    flags.DEFINE_integer('num_zs_test_samples', cp.getint("DEFAULT", "num_zs_test_samples"), 'Number of zero shot test samples')
    flags.DEFINE_integer('batch_size', cp.getint("DEFAULT", "batch_size"), 'Training batch size')
    flags.DEFINE_integer('print_step', cp.getint("DEFAULT", "print_step"), 'Print step')
    flags.DEFINE_integer('epochs', cp.getint("DEFAULT", "epochs"), 'Epochs to train')
    flags.DEFINE_integer('sampling_freq', cp.getint("DEFAULT", "sampling_freq"), 'Sampling frequency')
    flags.DEFINE_string('mode', cp.get("DEFAULT", "mode"), 'Analysis mode (ifreq, iphase, or fft)')
    flags.DEFINE_float('keep_prob', cp.getfloat("DEFAULT", "keep_prob"), 'Probability to keep neuron when using CNN')
    flags.DEFINE_integer('retrain_batch', cp.getint("DEFAULT", "retrain_batch"), 'Number of times to retrain the same batch (speeds up, but also overfits)')
    flags.DEFINE_string('exclude_classes', cp.get("DEFAULT", "exclude_classes"), 'Classes to exclude from training')

    # Mode specific options
    if cp.get("DEFAULT", "dataset") == 'matlab':  # TODO: Bug in Tensorflow: once FLAGS.dataset is accessed it's no longer possible to define new strings
        flags.DEFINE_string('matlabfile', cp.get("matlab", "matlabfile"), 'MATLAB LoRa database')
        flags.DEFINE_integer('chirp_length', cp.getint("matlab", "chirp_length"), 'Length of a single chirp')
    elif cp.get("DEFAULT", "dataset") == 'mongo':
        flags.DEFINE_string ('ip',         cp.get("mongo", "ip"), 'MongoDB server IP')
        flags.DEFINE_integer('port',       cp.get("mongo", "port"), 'MongoDB server port')
        flags.DEFINE_string ('db',         cp.get("mongo", "db"), 'MongoDB database name')
        flags.DEFINE_string ('collection', cp.get("mongo", "collection"), 'MongoDB chirp collection name')
        flags.DEFINE_string ('test_collection', cp.get("mongo", "test_collection"), 'MongoDB test chirp collection name')
        flags.DEFINE_integer ('random_mode', RandomMode.s2e(cp.get("mongo", "random_mode")), 'Data randomization approach')
        flags.DEFINE_string ('random_date', cp.get("mongo", "random_date"), 'Date for split date mode')
        flags.DEFINE_string ('filter', cp.get("mongo", "filter"), 'Query filter for "find" queries')
    elif cp.get("DEFAULT", "dataset") == 'random':
        flags.DEFINE_integer('num_classes', cp.get("random", "num_classes"), 'Number of random classes')
        flags.DEFINE_integer('num_samples', cp.get("random", "num_samples"), 'Number of random samples')

    # Classifier specific options
    if cp.get("DEFAULT", "classifier") == 'mlp':
        flags.DEFINE_integer('num_hidden_layers', cp.getint("mlp", "num_hidden_layers"), 'Number of hidden layers')
        flags.DEFINE_integer('num_hidden_neurons', cp.getint("mlp", "num_hidden_neurons"), 'Number of hidden neurons in a hidden layer')
    elif cp.get("DEFAULT", "classifier") == 'cnn':
        flags.DEFINE_integer('conv_kernel_width', cp.getint("cnn", "conv_kernel_width"), 'Convolution kernel width')
        flags.DEFINE_integer('pooling_kernel_width', cp.getint("cnn", "pooling_kernel_width"), 'Max pooling kernel width')
    elif cp.get("DEFAULT", "classifier") == 'mdn':
        flags.DEFINE_integer('num_hidden_layers', cp.getint("mdn", "num_hidden_layers"), 'Number of hidden layers')
        flags.DEFINE_integer('num_hidden_neurons', cp.getint("mdn", "num_hidden_neurons"), 'Number of hidden neurons in a hidden layer')

def print_conf(cp):  # Print settings to terminal
    for e in cp.defaults():
        print("[+] " + Fore.YELLOW + Style.BRIGHT + e + ": " + str(cp.get("DEFAULT", e)))

def select_cols(matrix, c1, c2):  # Select two columns from a numpy matrix
    return matrix[:, [c1, c2]]

# ----------------------------------------------------
# Dataset classes
# ----------------------------------------------------
class TensorIO():
    def __init__(self, x, y):
        self.x = x  # Input
        self.y = y  # Output

class Dataset():  # Dataset base class
    def __init__(self):
        self.num_training_samples = FLAGS.num_train_samples
        self.num_test_samples = FLAGS.num_test_samples

    # Based on the tag, get the LoRa ID
    def _determine_id(self, tag):
        if 'lora' in tag:
            lora_id = int(tag[4:])
            return lora_id
        print("[!] Warning: unable to determine lora_id for entry " + str(tag))
        return None

    # Preprocess an input so that it can be learned by Tensorflow
    def _data_to_tf_record(self, lora_id, chirp, debug=False):
        features = []

        #visualization.dbg_plot(preprocessing.iphase(chirp), title='Preprocessed chirp')
        chirp = preprocessing.roll_to_base(chirp)

        for m in FLAGS.mode.split(','):
            if m == 'iphase':
                features.append(preprocessing.iphase(chirp))
            elif m == 'fft':
                features.append(preprocessing.fft(chirp))
            elif m == 'ifreq':
                features.append(preprocessing.ifreq(chirp, FLAGS.sampling_freq))
            elif m == 'iamp':
                features.append(preprocessing.iamp(chirp))
            elif m == 'raw':
                features.append(preprocessing.normalize(chirp))
            else:
                print(Fore.RED + Style.BRIGHT + "[-] Analysis mode must be configured to be either 'fft', 'iphase', 'ifreq', or a comma separated combination.")
                exit(1)

        if debug:
            if lora_id == 1:
                visualization.dbg_plot(features[0], title='First feature vector of LoRa 1 chirp')

        tf_record = {"lora_id": lora_id, "iq": features}
        return tf_record

class GNURadioDataset(Dataset): # Convert pmt of IQ samples to numpy complex 64
    def __init__(self, pmt, symbol_length):
        self.pmt = pmt
        self.symbol_length = symbol_length

    def get(self):
        data = []

        frame = np.frombuffer(self.pmt, dtype=np.complex64)
        symbols = [frame[i:i+self.symbol_length] for i in range(0, len(frame), self.symbol_length)]
        for symbol in symbols:
            tf_record = self._data_to_tf_record(None, symbol)
            data.append(tf_record)

        return data

class FakeSampleDataset(Dataset):
    def __init__(self, host='localhost', port=27017, name="chirps"):
        Dataset.__init__(self)
        self.name = name

    def get(self, projection={}, num_records=500):
        return [{"lora_id": 1, "iq": [0+0j] * 74200}] * num_records

class UniformRandomDataset(Dataset):  # Sanity check dataset
    def __init__(self):
        Dataset.__init__(self)
        self.num_classes = FLAGS.num_classes
        self.lora_ids = set()
        for i in range(1, self.num_classes+1):
            self.lora_ids.add(i)

    def get(self, projection={}):
        data = []

        for i in range(0, FLAGS.num_samples):
            record = {"lora_id": random.randint(1,self.num_classes), "iq": [random.random() for x in range(0, FLAGS.limit)]}
            data.append(record)

        return data

class MatlabDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.path = FLAGS.matlabfile
        self.data = []
        self.lora_ids = set()

        # Load the file and contents
        mat_contents = sio.loadmat(self.path)
        self.all_samples = mat_contents['all_samples']

        # Determine number of classes
        for entry in self.all_samples:
            entry_name = os.path.basename(entry[0][0])
            lora_id = self._determine_id(entry_name)
            if lora_id is None:
                continue
            self.lora_ids.add(lora_id)

    def _determine_id(self, filename):
        for elem in filename.split('-'):
            if 'lora' in elem:
                return Dataset._determine_id(self, elem)

    def get(self, projection={}, num_records=0):  # TODO: projection
        data = []

        # Parse class data
        for entry in self.all_samples:
            entry_name = os.path.basename(entry[0][0])
            entry_data = entry[1]
            lora_id = self._determine_id(entry_name)
            if lora_id is None:
                continue
            print("Parsing " + entry_name + " (class " + str(lora_id) + ", " + str(len(entry_data)) + " samples)")

            for record in entry_data:
                for i in range(0, 8):
                    chirp = record[i*FLAGS.chirp_length:(i+1)*FLAGS.chirp_length]
                    tf_record = self._data_to_tf_record(lora_id, chirp, debug=args.debug)
                    data.append(tf_record)

        return data

class RandomMode:
    RANDOMIZE_SYMBOLS = 0
    RANDOMIZE_FRAMES = 1
    SPLIT_DATE = 2
    SPLIT_COLLECTION = 3

    _STR_RANDOMIZE_SYMBOLS = 'randomize_symbols'
    _STR_RANDOMIZE_FRAMES = 'randomize_frames'
    _STR_SPLIT_DATE = 'split_date'
    _STR_SPLIT_COLLECTION = 'split_collection'

    @staticmethod
    def e2s(enum):
        if enum == RandomMode.RANDOMIZE_SYMBOLS:
            return RandomMode._STR_RANDOMIZE_SYMBOLS
        elif enum == RandomMode.RANDOMIZE_FRAMES:
            return RandomMode._STR_RANDOMIZE_FRAMES
        elif enum == RandomMode.SPLIT_DATE:
            return RandomMode._STR_SPLIT_DATE
        elif enum == RandomMode.SPLIT_COLLECTION:
            return RandomMode._STR_SPLIT_COLLECTION
        else:
            print(Fore.YELLOW + Style.BRIGHT + "[!] Warning: unknown enum %d. Defaulting to 0." % enum)
            return 0

    @staticmethod
    def s2e(string):
        if string == RandomMode._STR_RANDOMIZE_SYMBOLS:
            return RandomMode.RANDOMIZE_SYMBOLS
        elif string == RandomMode._STR_RANDOMIZE_FRAMES:
            return RandomMode.RANDOMIZE_FRAMES
        elif string == RandomMode._STR_SPLIT_DATE:
            return RandomMode.SPLIT_DATE
        elif string == RandomMode._STR_SPLIT_COLLECTION:
            return RandomMode.SPLIT_COLLECTION
        else:
            print(Fore.YELLOW + Style.BRIGHT + "[!] Warning: unknown randomization mode '%s'. Defaulting to randomize_symbols." % string)
            return RandomMode.RANDOMIZE_SYMBOLS

class MongoDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.ip = FLAGS.ip
        self.port = FLAGS.port
        self.client = MongoClient(self.ip, self.port)
        self.db = self.client[FLAGS.db]
        self.collection = self.db[FLAGS.collection]
        self.collection_test = self.db[FLAGS.test_collection]
        self.lora_ids = set()
        self.random_mode = FLAGS.random_mode
        self.filter = json.loads(FLAGS.filter)
        self.num_samples = self.collection.find(self.filter).count()
        print(Fore.MAGENTA + Style.BRIGHT + "[+] Filter: %s" % str(self.filter))
        self.sort = '$natural' if args.natural else 'rand'

        # Randomize mongo set
        self.randomize()

        # Randomize all symbols and divide into training and test set
        if self.random_mode == RandomMode.RANDOMIZE_SYMBOLS:
            self.cursor_train = self.collection.find(self.filter).sort(self.sort, 1).skip(0).limit(self.num_training_samples)
            self.cursor_test = self.collection.find(self.filter).sort(self.sort, 1).skip(self.num_training_samples).limit(self.num_test_samples)
        elif self.random_mode == RandomMode.RANDOMIZE_FRAMES:
            self.collection.create_index("fn")
            # Find out how many test frames we need
            frames_for_test = int(self.num_test_samples / 36)  # 36 = number of symbols in frame

            # Find highest frame number
            print("[+] Finding highest frame number")
            last_fn = self.collection.find(self.filter).sort("fn", -1).limit(1)[0]['fn']

            # Generate list of random frame numbers to be used as test set
            test_fns = []
            for i in range(0, frames_for_test):
                test_fns.append(randint(0, last_fn))

            # Assign the cursors
            train_query = self.filter.copy()
            train_query["fn"] = {"$nin": test_fns}
            self.cursor_train = self.collection.find(train_query).sort(self.sort, 1).limit(self.num_training_samples)

            test_query = self.filter.copy()
            test_query["fn"] = {"$in": test_fns}
            self.cursor_test = self.collection.find(test_query).sort(self.sort, 1).limit(self.num_test_samples)
        elif self.random_mode == RandomMode.SPLIT_DATE:
            self.collection.create_index("date")

            print("[+] Splitting test set after date: %s" % FLAGS.random_date)
            the_date = datetime.strptime(FLAGS.random_date,'%Y-%m-%dT%H:%M:%SZ')

            train_query = self.filter.copy()
            train_query["date"] = {"$lt": the_date}
            self.cursor_train = self.collection.find(train_query).sort(self.sort, 1).limit(self.num_training_samples)

            test_query = self.filter.copy()
            test_query["date"] = {"$gte": the_date}
            self.cursor_test = self.collection.find(test_query).sort(self.sort, 1).limit(self.num_test_samples)
        elif self.random_mode == RandomMode.SPLIT_COLLECTION:
            self.cursor_train = self.collection.find(self.filter).sort(self.sort, 1).limit(self.num_training_samples)
            self.cursor_test = self.collection_test.find(self.filter).sort(self.sort, 1).limit(self.num_test_samples)

        # Determine number of classes
        print("[+] Determining number of classes")
        for tag in self.cursor_train.distinct('tag'):
            lora_id = self._determine_id(tag)
            if lora_id is None:
                continue
            self.lora_ids.add(lora_id)
        self.cursor_train.rewind()

        # Create caches
        self.cache_train = GenericCache(name="train")
        self.cache_test = GenericCache(name="test")

    def randomize(self):
        if os.path.isfile('/tmp/randomized_mongo'):
            print("[+] MongoDB dataset is already randomized")
            return

        self._randomize(self.collection, "")
        if self.random_mode == RandomMode.SPLIT_COLLECTION:  # If random mode is set to split collection, also randomize this collection
            self._randomize(self.collection_test, "(test set)")

        with open('/tmp/randomized_mongo', "w") as f:
            f.write('')

    def _randomize(self, collection, label=""):
        print("[+] Randomizing MongoDB dataset %s" % label)
        progress = 0
        for entry in collection.find(self.filter):
            collection.update({"_id": entry["_id"]}, {"$set": {"rand": random.random()}}, upsert=False, multi=False)
            progress += 1
            print("\r[+] Progress: %d / %d (estimation)                          " % (progress, self.num_samples)),
        print("")
        print("[+] Creating index")
        collection.create_index("rand")

    def get(self, train=True, projection={}, num_records=1000):
        data = []
        set_in_memory = False

        if train:
            cursor = self.cursor_train
            cache = self.cache_train
            num_records_total = self.num_training_samples
        else:
            cursor = self.cursor_test
            cache = self.cache_test
            num_records_total = self.num_test_samples

        if len(cache) == num_records_total:
            set_in_memory = True

        # Set is already loaded in cache memory
        if set_in_memory:
            for i in range(0, num_records):
                try:
                    tf_record = cache.next()
                except StopIteration:
                    cache.rewind()
                    tf_record = cache.next()
                data.append(tf_record)
        else: # Go through each record in the MongoDB
            for i in range(0, num_records):
                try:
                    record = cursor.next()
                except StopIteration:
                    cursor.rewind()
                    record = cursor.next()
                except (OperationFailure, AutoReconnect) as e:
                    print("[!] Warning: Got other exception than StopIteration: "),
                    print(e)
                    cursor.rewind()
                    record = cursor.next()

                lora_id = self._determine_id(record['tag'])
                if lora_id is None:
                    continue

                tf_record = cache.get(record['_id'])
                if tf_record is None:
                    chirp = np.frombuffer(record['chirp'], dtype=np.complex64)
                    tf_record = self._data_to_tf_record(lora_id, chirp, debug=args.debug)
                    cache.store(record['_id'], tf_record)

                data.append(tf_record)

        return data

# The Instances class is responsible for providing:
# - Preprocessing of the raw chirp data into features
# - Separation of dataset into training and test sets
# - Random shuffling of training and test data
class Instances():
    def __init__(self, limit=None, exclude_classes=[], name="", mapping=None):
        self.name = name
        self.num_excluded_samples = 0
        self.limit = limit
        self.exclude_classes = exclude_classes

        # Select dataset type
        if cp.get("DEFAULT", "dataset") == 'matlab':
            self.dataset = MatlabDataset()
        elif cp.get("DEFAULT", "dataset") == 'mongo':
            self.dataset = MongoDataset()
        elif cp.get("DEFAULT", "dataset") == 'random':
            self.dataset = UniformRandomDataset()
        else:
            print(Fore.RED + Style.BRIGHT + "[-] Unknown dataset type '" + cp.get("DEFAULT", "dataset") + "'. Exiting")
            exit(1)

        # Make sure we don't underestimate available data
        print("[+] Got " + Fore.GREEN + Style.BRIGHT + str(self.dataset.num_samples) + Style.RESET_ALL + " samples")
        if self.dataset.num_test_samples + self.dataset.num_training_samples > self.dataset.num_samples:
            print(Fore.RED + Style.BRIGHT + "[-] Sum of training and test samples exceeds available samples. Exiting")
            exit(1)

        # Get length of input samples (= number of features) and configure limit
        print("[+] Getting number of features (1 record get from test set)")
        self.num_features = self._get_num_features(self.dataset.get(train=False, num_records=1))
        if self.limit == -1 or self.limit is None:
            self.limit = self.num_features
        print("[+] First sample contains %d features (limited to %d)" % (self.num_features, self.limit))

        # Create mapping from LoRa ID to One Hot Vector if necessary
        if mapping is None:
            self.mapping = Mapping(self.dataset.lora_ids, exclude_classes=self.exclude_classes)
            self.mapping.display()
        else:  # Update existing map with any new entries found
            self.mapping = mapping
            self.mapping.update(self.dataset.lora_ids, exclude_classes=self.exclude_classes)
            self.mapping.display()

    def next_batch(self, train, size):
        temp = list(self.dataset.get(train=train, num_records=size))

        if len(temp) > 0:
            # Randomize (already done in Mongo, but not for other datasets)
            random.shuffle(temp)

            # Create instances
            instances_x = []
            instances_y = []
            for i in range(0, size):
                processed_record = self.process_record(temp[i])
                if not (processed_record is None):
                    instances_x.append(processed_record.x[0:self.limit])
                    instances_y.append(processed_record.y)

            instances_x = np.array(instances_x, dtype=np.float32)
            instances_y = np.array(instances_y, dtype=np.float32)

            # Done!
            #if len(self.exclude_classes) > 0:
            #    print(Fore.GREEN + Style.BRIGHT + "[+] EXCLUDING %d samples" % self.num_excluded_samples)
        else:
            print("[-] No samples found in dataset. Exiting")
            exit(1)

        if len(instances_x) == 0:
            raise Exception

        return instances_x, instances_y

    def _get_num_features(self, x):
        return len(np.array(x[0]["iq"]).flatten())

    def process_record(self, record):
        # Do some preprocessing on the records here
        if record["lora_id"] in self.exclude_classes:
            self.num_excluded_samples += 1
            return None

        one_hot_vector = self.mapping.lora_id_to_oh(record["lora_id"])
        features = np.array(record["iq"]).flatten()

        return TensorIO(features, one_hot_vector)

# ----------------------------------------------------
# ML models
# Some of these models are based on the reference im-
# plementations provided by Aymeric Damien. See
# https://github.com/aymericdamien/TensorFlow-Examples
# for more information.
# ----------------------------------------------------
class MLModel():  # Base class for ML models
    def __init__(self):
        self.learning_rate = None
        self.layers = []
        self.output_layer = None
        self.cost_function = None
        self.correct_prediction = None


class MLPModel(MLModel):
    def __init__(self, x, num_inputs, y, num_classes, hidden_layers=0, hidden_neurons=0, name='mlp'):
        MLModel.__init__(self)
        self.learning_rate = 0.0001 #0.001 works pretty good too
        next_layer = x
        next_layer_size = num_inputs
        for i in range(0, hidden_layers):
            self.layers.append(LinearReluLayer(next_layer, next_layer_size, hidden_neurons, name=name+'lin' + str(i)))
            self.output_layer = self.layers[-1]
            next_layer = self.output_layer.h
            next_layer_size = hidden_neurons

        self.layers.append(LinearLayer(next_layer, next_layer_size, num_classes, name=name+'clin', init_zero=True))  # Since it will be softmaxed later, init to zero. Seems to affect training speed and making the weights align on a diagonal faster
        self.output_layer = self.layers[-1]

        #self.cost_function = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.nn.softmax(self.output_layer.h)+EPSILON), reduction_indices=[1]))  # Doesn't deal with edge cases so we need to add EPSILON
        self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer.h, labels=y))
        #self.cost_function = tf.reduce_mean(tf.reduce_sum(tf.square(y - tf.nn.softmax(self.output_layer.h)), reduction_indices=[1]))

        self.correct_prediction = tf.equal(tf.argmax(self.output_layer.h,1), tf.argmax(y,1))

class ConvNeuralNetModel(MLModel):
    def __init__(self, x, num_inputs, y, num_classes, keep_prob=None, name='cnn'):
        MLModel.__init__(self)
        self.learning_rate = 0.001 # 0.0001

        # Make image
        x_shaped = tf.reshape(x, shape=[-1, 1, num_inputs, 1])

        # Append convolution layers
        self.layers.append(NNLayer(x_shaped, [1, FLAGS.conv_kernel_width, 1, 32], [32], name=name+'wc1'))
        self.output_layer = self.layers[-1]
        self.layers.append(NNLayer(self.output_layer.h, [1, FLAGS.conv_kernel_width, 32, 64], [64], name=name+'wc2'))
        self.output_layer = self.layers[-1]

        # Reshape conv2 output to fit fully connected layer input
        relu_inputs = (num_inputs/pow(FLAGS.pooling_kernel_width, 2))*64  # 64 = output channels per sample from conv. 4 = k from polling (see paper notes). Power of two because max pooling twice
        relu_outputs = num_inputs
        out_shaped = tf.reshape(self.output_layer.h, [-1, relu_inputs])

        # Append fully connected layer
        self.layers.append(LinearReluDropLayer(out_shaped, relu_inputs, relu_outputs, keep_prob))
        self.output_layer = self.layers[-1]

        # Output, class prediction
        self.layers.append(LinearLayer(self.output_layer.h, relu_outputs, num_classes, name=name+'lin'))
        self.output_layer = self.layers[-1]

        self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer.h, labels=y))

        self.correct_prediction = tf.equal(tf.argmax(self.output_layer.h,1), tf.argmax(y,1))

class MDNModel(MLModel):
    def __init__(self, x, num_inputs, y, num_classes, hidden_layers=0, hidden_neurons=0, name='mdn'):
        MLModel.__init__(self)
        self.num_classes = num_classes
        self.learning_rate = 0.001
        next_layer = x
        next_layer_size = num_inputs

        # Hidden layers
        for i in range(0, hidden_layers):
            self.layers.append(LinearLayer(next_layer, next_layer_size, hidden_neurons, name=name+'lin' + str(i)))
            self.output_layer = self.layers[-1]
            next_layer = self.output_layer.h
            next_layer_size = hidden_neurons

        # MDN layer
        self.layers.append(MixtureLayer(next_layer, next_layer_size, num_classes, name=name+"mix"))
        self.output_layer = self.layers[-1]
        self.pi, self.mu, self.sigma = self._get_components(self.output_layer)
        self.gauss = tf.contrib.distributions.Normal(mu=self.mu, sigma=self.sigma)

        # Cost function
        self.cost_function = self._get_cost_function(y)

        # Evaluation
        self.correct_prediction = tf.equal(tf.argmax(tf.mul(self.pi,self.gauss.mean()), 1), tf.argmax(y,1))

    def _get_components(self, layer):
        pi = tf.placeholder("float", [None, layer.num_components])
        mu = tf.placeholder("float", [None, layer.num_components])
        sigma = tf.placeholder("float", [None, layer.num_components])

        pi, mu, sigma = tf.split(1, layer.num_components, layer.h)

        pi = tf.nn.softmax(pi)
        #assert_op = tf.Assert(tf.equal(tf.reduce_sum(pi), 1.), [pi])
        #pi = tf.with_dependencies([assert_op], pi)

        sigma = tf.exp(sigma)

        return pi, mu, sigma

    def _get_cost_function(self, y):
        return tf.reduce_mean(-tf.log(tf.reduce_sum(tf.mul(self.pi, self.gauss.pdf(y)), 1, keep_dims=True)))

    def _sample(self, n):
        # Randomly sample x times according to pi distribution
        mixture_indices = tf.reshape(tf.multinomial(tf.log(self.pi), n), [-1]) # Pi must be a log probability

        # Sample all gaussian distributions x times
        samples = tf.reshape(self.gauss.sample(n), [-1, self.num_classes])

        # Select only the one according to pi
        select_gaussians = tf.reduce_sum(tf.one_hot(mixture_indices, self.num_classes) * samples, 1)

        return select_gaussians

    def _mean(self):
        # Get the indices of the most likely mixtures beloning to each x
        mixture_indices = tf.argmax(self.pi, 1)

        # Get the expected values of all gaussians
        exp_values = self.gauss.mean()

        # Get expected value of most likely mixture
        select_exp = tf.reduce_sum(tf.one_hot(mixture_indices, self.num_classes) * exp_values, 1)

        return select_exp

class ModelType:
    MLP = 0
    CONVNET = 1
    MDN = 2

    @staticmethod
    def str2type(string):
        if string == "mlp":
            return ModelType.MLP
        elif string == "cnn":
            return ModelType.CONVNET
        elif string == "mdn":
            return ModelType.MDN
        else:
            print(Fore.RED + Style.BRIGHT + "[-] Model type "+ string +" does not exist.")
            exit(1)

# ----------------------------------------------------
# ML classifiers
# ----------------------------------------------------
class SVM():
    def __init__(self, name="svc"):
        print("[+] SVM Classifier")
        self.m = SVC()
        self.name = name

    def _get_lora_id_labels(self, instances, oh_labels):
        result = []
        for i in range(0, len(oh_labels)):
            result.append(instances.mapping.oh_to_lora_id(oh_labels[i]))
        return result

    def _to_vendor(self, instances, lora_id_labels):
        result = []
        for i in range(0, len(lora_id_labels)):
            result.append(instances.mapping.lora_id_to_vendor_id(lora_id_labels[i]))
        return result

    def train(self, instances, batch_size=2500):
        print("[+] Getting %d training samples" % batch_size)
        train_samples_x, train_samples_y = instances.next_batch(True, batch_size)
        train_samples_y = self._get_lora_id_labels(instances, train_samples_y)
        print("[+] Training model")
        self.m.fit(train_samples_x, train_samples_y)

    def save(self):
        path = FLAGS.trainedmodelsdir + self.name + "/"

        if not os.path.exists(path):
            os.makedirs(path)

        # Save model
        pickle.dump(self.m, open(path + 'svc_model.p', "wb"))

    @staticmethod
    def load():
        path = FLAGS.trainedmodelsdir + FLAGS.model_name + "/"

        # Set up classifier based on config and stored data
        net = SVM()
        net.m = pickle.load(open(path + 'svc_model.p', "rb"))

        return net

    def bin_class_per_sample(self, instances, limit=200, adv_detect=True, vendor_only=False):
        print("[+] Getting %d test samples" % limit)
        test_samples_x, test_samples_y = instances.next_batch(False, limit)
        test_samples_y = self._get_lora_id_labels(instances, test_samples_y)

        print("[+] Evaluating model")
        predicted_y = self.m.predict(test_samples_x)

        if vendor_only:
            metrics = utilities.get_eval_metrics_percent(self._to_vendor(instances, test_samples_y), self._to_vendor(instances, predicted_y))
        else:
            metrics = utilities.get_eval_metrics_percent(test_samples_y, predicted_y)
        utilities.print_metrics(metrics)
        return

    def visualize_embeddings(self, instances, limit=200, train=True):
        print("[!] Warning: visualize_embeddings not implemented for SVM")
        return

class Classifier():
    # Build the classifier
    def __init__(self, num_inputs, num_classes, name, modeltype=ModelType.MLP):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.name = name
        self.step = 0
        self.modeltype = modeltype
        self.expected_values = None
        self.std = None
        self.distance_threshold = np.zeros(num_classes)
        self.sess = None
        self.instances_mapping = None
        model_summaries = []

        self.x = tf.placeholder("float", [None, self.num_inputs], name='inputs')
        self.y = tf.placeholder("float", [None, self.num_classes], name='map-id-oh')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')

        if modeltype == ModelType.MLP:
            self.m = MLPModel(self.x, self.num_inputs, self.y, self.num_classes, hidden_layers=FLAGS.num_hidden_layers, hidden_neurons=FLAGS.num_hidden_neurons, name="mlp") # Build MLP model
        elif modeltype == ModelType.CONVNET:
            self.m = ConvNeuralNetModel(self.x, self.num_inputs, self.y, self.num_classes, keep_prob=self.keep_prob, name="cnn") # Build Convolutional Neural Network model
        elif modeltype == ModelType.MDN:
            self.m = MDNModel(self.x, self.num_inputs, self.y, self.num_classes, hidden_layers=FLAGS.num_hidden_layers, hidden_neurons=FLAGS.num_hidden_neurons, name="mdn") # Build MDN model
        else:
            raise Exception("No model type specified")

        # Define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.m.learning_rate).minimize(self.m.cost_function)

        # Define accuracy model
        self.accuracy = tf.reduce_mean(tf.cast(self.m.correct_prediction, tf.float32))

        # Merge TensorBoard summaries for the model
        model_summaries.append(tf.summary.scalar('accuracy', self.accuracy))
        model_summaries.append(tf.summary.scalar('cost', self.m.cost_function))
        self.merged_model_summaries = tf.summary.merge(model_summaries, collections=None, name=None)

        # Define session object and summary writers
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', graph=self.sess.graph)
        self.test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')

    def __del__(self):
        if not (self.sess is None):
            self.sess.close()
            self.train_writer.close()
            self.test_writer.close()

    # Plot sample data to Tensorboard
    def _plot_samples(self, samples_x, samples_y):
        # Register plot summaries
        plot_summaries = []
        plots_to_show = 5

        learned_weights_tensor = tf.identity(self.m.output_layer.W)
        learned_weights = self.sess.run(learned_weights_tensor)
        plot_summaries.append(visualization.plot_values(samples_x[0], self.instances_mapping, height=500, width=self.num_inputs, tag="weights", title="Weights", label=np.argmax(samples_y[0]), backdrop=learned_weights))

        for i in range(1, 6):
            label = np.argmax(samples_y[i])
            guess = self.get_accuracy([samples_x[i]], [samples_y[i]])
            plot_summaries.append(visualization.plot_values(samples_x[i], self.instances_mapping, height=500, width=self.num_inputs, tag="trd" + str(i) + "c" + str(label) + "g" + str(guess), title="Training data", label=label))

        # Merge TensorBoard summaries for plots
        merged_plot_summaries = tf.summary.merge(plot_summaries, collections=None, name=None)
        summary_plot = self.sess.run(merged_plot_summaries)
        self.train_writer.add_summary(summary_plot)

    # Plot kernel data to Tensorboard
    def _plot_kernels(self):
        plot_summaries = []

        # TODO go through layers and check .startswith("wc")
        kernels_tensor = self.m.layers[0].W
        kernels_shaped_tensor = tf.reshape(kernels_tensor, [-1, FLAGS.conv_kernel_width])  # Arrange kernels so that there is one per row
        kernels_shaped = self.sess.run(kernels_shaped_tensor)

        plot_summaries.append(visualization.plot_kernels(kernels_shaped, FLAGS.conv_kernel_width, height=4096, width=1024, tag="kernels", title="CNN Kernels"))

        # Merge TensorBoard summaries for plots TODO dup code
        merged_plot_summaries = tf.summary.merge(plot_summaries, collections=None, name=None)
        summary_plot = self.sess.run(merged_plot_summaries)
        self.train_writer.add_summary(summary_plot)

    def get_output_weights(self, samples_x):
        return self.sess.run(self.m.output_layer.h, feed_dict={self.x: samples_x, self.keep_prob: 1.0})

    def _plot_output_weights_2d(self, samples_x, samples_y, predictions_y, instances, metrics):  # Do not use new samples from instances
        plot_summaries = []

        # Get the output weight values for all classes
        output_weights = self.get_output_weights(samples_x)

        # OLD: Get first two weights to visualize
        # weights = select_cols(output_weights, 0, 1)

        # Reduce dimensionality of weights to 2
        tsne = TSNE(n_components=2, init='pca', n_iter=5000)
        weights = tsne.fit_transform(output_weights)

        #xlabel = "Weight #" + str(0) + " values"
        #ylabel = "Weight #" + str(1) + " values"
        xlabel = "t-SNE dimension 1"
        ylabel = "t-SNE dimension 2"
        plot_summaries.append(visualization.plot_weights(weights, samples_y, predictions_y, self.expected_values, self.distance_threshold, instances.mapping, tag=self.name+"-w", title="2D projection of output feature weights", xlabel=xlabel, ylabel=ylabel, metrics=metrics))

        # Merge TensorBoard summaries for plots TODO dup code
        merged_plot_summaries = tf.summary.merge(plot_summaries, collections=None, name=None)
        summary_plot = self.sess.run(merged_plot_summaries)
        self.train_writer.add_summary(summary_plot)

    def train(self, instances, batch_size=2500):
        # Let's go
        print("[+] Training")
        self.sess.run(tf.global_variables_initializer())

        # Start learning weights
        try:
            while True:
                train_batch_x, train_batch_y = instances.next_batch(True, batch_size)
                test_batch_x, test_batch_y = instances.next_batch(False, batch_size)

                # Execute training step(s) on batch
                #print(self.sess.run(self.m.tmp, feed_dict={self.x: train_batch_x, self.y: train_batch_y, self.keep_prob: FLAGS.keep_prob}))  # To test something inside model with the same data
                for i in range(0, FLAGS.retrain_batch):
                    self.sess.run(self.optimizer, feed_dict={self.x: train_batch_x, self.y: train_batch_y, self.keep_prob: FLAGS.keep_prob})

                    # Print progress
                    if self.step % FLAGS.print_step == 0:
                        # Print stats about step
                        summary_train, c_train, a_train = self.sess.run([self.merged_model_summaries, self.m.cost_function, self.accuracy], feed_dict={self.x: train_batch_x, self.y: train_batch_y, self.keep_prob: 1.0})
                        summary_test = self.sess.run(self.merged_model_summaries, feed_dict={self.x: test_batch_x, self.y: test_batch_y, self.keep_prob: 1.0})

                        # Add summaries
                        self.train_writer.add_summary(summary_train, self.step)
                        self.test_writer.add_summary(summary_test, self.step)

                        # Print info about training
                        print("Epoch {:d}: cost={:.6f}, tr_acc={:.6f}, W0_0={:.6f}".format(self.step, c_train, a_train, self.sess.run(self.m.output_layer.W)[0][0]))

                    # Next step
                    self.step += 1

                    if self.step == FLAGS.epochs:
                        raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        # Save the mapping used during training from LoRa ID to Map ID
        self.instances_mapping = instances.mapping

        # Mixture components
        self.expected_values, self.std = self.calculate_mixture_components(instances)

        # Show results
        print(Fore.GREEN + Style.BRIGHT + "[+] Done training!")

        if self.modeltype == ModelType.MLP:
            print(Fore.GREEN + Style.BRIGHT + "[+] Plotting training samples")
            self._plot_samples(train_batch_x, train_batch_y)
        else:
            print(Fore.GREEN + Style.BRIGHT + "[+] Plotting model kernels")
            self._plot_kernels()

        # Evaluation
        print("[+] Training set accuracy")
        print(self.get_accuracy(train_batch_x, train_batch_y))
        print("[+] Test set accuracy")
        print(self.get_accuracy(test_batch_x, test_batch_y))

        # Assert that nothing unexpected happened during the whole process
        GenericCache.assert_disjunction(instances.dataset.cache_train, instances.dataset.cache_test)
        print(Fore.GREEN + Style.BRIGHT + "[+] Training assertions passed")


    def determine_ideal_threshold(self, map_id, samples_x, expected_values):
        output_weights = self.sess.run(self.m.output_layer.h, feed_dict={self.x: samples_x, self.keep_prob: 1.0})
        threshold = 0.0

        for output_weight in output_weights:
            #threshold = max(np.linalg.norm(output_weight - expected_values), threshold)
            #threshold = (np.linalg.norm(output_weight - expected_values) + threshold) / 2.0
            threshold += np.linalg.norm(output_weight - expected_values)
        threshold /= len(output_weights)
        return threshold

    def calculate_mixture_components(self, instances, num_samples_to_use=10000):
        print("[+] Determining mixture model components")
        train_batch_x, train_batch_y = instances.next_batch(True, num_samples_to_use)
        expected_values = np.ndarray(shape=(self.num_classes,self.num_classes), dtype=np.float32)
        std = np.ndarray(shape=(self.num_classes,self.num_classes), dtype=np.float32)

        for lora_id in instances.mapping.keys():
            map_id = instances.mapping.lora_to_map_id(lora_id)
            samples_x = []

            # Collect samples belonging to class map_id
            for i in range(0, len(train_batch_x)):
                if np.argmax(train_batch_y[i]) == map_id:
                    samples_x.append(train_batch_x[i])

            if len(samples_x) == 0:
                print(train_batch_y)
                print("[-] Error: no samples in training set for LoRa %d. Dumped y training set" % lora_id)
                exit()

            # Determine mean and std deviation for all features
            nn_output_weights = self.sess.run(tf.identity(self.m.output_layer.h), feed_dict={self.x: samples_x, self.keep_prob: 1.0})
            expected_values[map_id] = np.mean(nn_output_weights, axis=0)
            std[map_id] = np.std(nn_output_weights, axis=0)

            # Determine ideal threshold based on expected values
            # this threshold is used when doing nearest neighbor classification
            # as the outlier detection (not discussed in paper)
            if args.distance_threshold == 'auto':
                print("\r[+] Determining expected value distance threshold for LoRa %d  " % lora_id),
                self.distance_threshold[map_id] = self.determine_ideal_threshold(map_id, samples_x, expected_values[map_id])
            else:
                self.distance_threshold[map_id] = args.distance_threshold

            # Clean up
            del samples_x
        print("")
        return expected_values, std

    # Calculates the distance between a point and a centroid
    def calculate_expected_values_distance(self, samples_x):
        if self.expected_values is None or self.distance_threshold is None:
            raise Exception("Tried to evaluate expected value MSE without training values")

        output_weights = self.sess.run(self.m.output_layer.h, feed_dict={self.x: samples_x, self.keep_prob: 1.0})

        distances = []
        for output_weight_v in output_weights:
            distances.append(np.linalg.norm(output_weight_v - self.expected_values, axis=1))  # Distance from E(X) for each class to X

        return distances

    def get_accuracy(self, samples_x, samples_y):
        return self.sess.run(self.accuracy, feed_dict={self.x: samples_x, self.y: samples_y, self.keep_prob: 1.0})

    def save(self):
        path = FLAGS.trainedmodelsdir + self.name + "/"

        if not os.path.exists(path):
            os.makedirs(path)

        # Save number of inputs
        np.save(path + 'value-inputs', self.num_inputs)

        # Save number of classes
        np.save(path + 'value-classes', self.num_classes)

        # Save layers
        for layer in self.m.layers:
            filename = path + 'layer-' + layer.name
            layer.saver.save(self.sess, filename, global_step=0)

        # Save expected classification output
        np.save(path + 'value-expected', self.expected_values)
        np.save(path + 'value-std', self.std)

        # Save distance threshold
        np.save(path + 'value-dt', self.distance_threshold)

        # Save instance mapping
        pickle.dump(self.instances_mapping, open(path + 'value-mapping.p', "wb"))

    @staticmethod
    def load(self, step=0):
        path = FLAGS.trainedmodelsdir + FLAGS.model_name + "/"

        # Load inputs and classes. Required to set up models.
        num_inputs = np.load(path + 'value-inputs' + '.npy')
        num_classes = np.load(path + 'value-classes' + '.npy')

        # Set up classifier based on config and stored data
        net = Classifier(num_inputs=num_inputs, num_classes=num_classes, name=FLAGS.model_name, modeltype=ModelType.str2type(FLAGS.classifier))

        for layer in net.m.layers:
            filename = path + 'layer-' + layer.name + '-' + str(step)
            layer.saver.restore(net.sess, filename)

        try:
            net.expected_values = np.load(path + 'value-expected' + '.npy')
            net.std = np.load(path + 'value-std' + '.npy')
        except IOError:
            print("[!] Warning: model does not have 'value-expected' and/or 'value-std', and will not be able to perform zero shot classification as a result.")
            pass

        net.distance_threshold = np.load(path + 'value-dt' + '.npy')

        net.instances_mapping = pickle.load(open(path + 'value-mapping.p', "rb"))

        return net

    def test(self, instances, limit=200):
        test_samples_x, test_samples_y = instances.next_batch(False, limit)

        # Metrics
        accuracy = self.get_accuracy(test_samples_x, test_samples_y)
        print(Fore.GREEN + Style.BRIGHT + "[+] Evaluation accuracy for %d samples: %.2f percent" % (limit, accuracy * 100.0))

    # Determine to which class a (set of) symbols belongs.
    # If clustering is used, then the frame is sent by an attacker if it does not belong to any cluster
    def _predict(self, samples_x, adv_detect):
        if FLAGS.clustering == "l1nn":
            return self._predict_nearest_neighbor_l1(samples_x, adv_detect)
        elif FLAGS.clustering == "argmax" or FLAGS.clustering == "none":
            if adv_detect:  # TODO: Threshold in this case?
                print("[!] Warning: adv_detect cannot be used with argmax clustering at the moment")
            return self._predict_argmax(samples_x)
        else:  # Don't do clustering, but use the closest predicted class
            print("[!] Warning: unknown clustering approach '%s'; defaulting to 'none'" % FLAGS.clustering)
            return self._predict_argmax(samples_x)

    # Predict class with least L1 distance to expected weight
    def _predict_nearest_neighbor_l1(self, samples_x, adv_detect):
        expected_values_distance = self.calculate_expected_values_distance(samples_x)

        idmap_predictions = []
        for ed in expected_values_distance:
            map_id = np.argmin(ed)
            if adv_detect and (ed[map_id] > self.distance_threshold[map_id]):
                map_id = -1
            idmap_predictions.append(map_id)

        most_probable = stats.mode(idmap_predictions)[0][0]
        return most_probable, idmap_predictions

    # Predict class with highest weight
    def _predict_argmax(self, samples_x):
        idmap_predictions = self.sess.run(tf.argmax(self.m.output_layer.h, 1), feed_dict={self.x: samples_x, self.keep_prob: 1.0})

        most_probable = stats.mode(idmap_predictions)[0][0]
        return most_probable, idmap_predictions

    def _predict_zeroshot(self, samples_x):
        weights = self.sess.run(self.m.output_layer.h, feed_dict={self.x: samples_x, self.keep_prob: 1.0})
        probabilities = self.sess.run(tf.nn.softmax(self.m.output_layer.h), feed_dict={self.x: samples_x, self.keep_prob: 1.0})
        return weights, probabilities

    # Function to visualize confusion matrix and calculate the metrics ourselves
    def _print_statistics(self, confusion_matrix):
        num_classes = confusion_matrix.shape[0]
        true_positives = np.zeros(num_classes)
        false_positives = np.zeros(num_classes)
        false_negatives = np.zeros(num_classes)
        true_negatives = np.zeros(num_classes)
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        accuracy = np.zeros(num_classes)

        # Calculate metrics
        for i in range(num_classes):
            true_positives[i] = confusion_matrix[i,i]
        for i in range(num_classes):
            false_positives[i] = np.sum(confusion_matrix[:,i]) - true_positives[i]
        for i in range(num_classes):
            false_negatives[i] = np.sum(confusion_matrix[i,:]) - true_positives[i]
        for i in range(num_classes):
            true_negatives[i] = np.sum(confusion_matrix) - (false_positives[i] + false_negatives[i] + true_positives[i])
        for i in range(num_classes):
            precision[i] = true_positives[i] / (true_positives[i] + false_positives[i])
        for i in range(num_classes):
            recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i])
        for i in range(num_classes):
            accuracy[i] = (true_positives[i] + true_negatives[i]) / (true_positives[i] + false_positives[i] + false_negatives[i] + true_negatives[i])

        np.set_printoptions(threshold='nan', linewidth=200)
        print("Confusion matrix")
        print(confusion_matrix)
        print("TP")
        print(true_positives)
        print("FP")
        print(false_positives)
        print("FN")
        print(false_negatives)
        print("TN")
        print(true_negatives)
        print("Precision")
        print(precision)
        print("Recall")
        print(recall)
        print("Accuracy")
        print(accuracy)

        # Accuracy according to Wikipedia. This metric is not correct because
        # it counts partially correct samples in the true negatives part of the
        # confusion matrix. For example: when class 5 is a true negative with
        # respect to a class 3 one-v-all classifier, it is considered correct
        # even though the true class is 7.
        model_accuracy_partial_correct = np.mean(accuracy)

        # Decent metrics
        model_accuracy = np.sum(true_positives) / np.sum(confusion_matrix)
        model_precision_macro = np.mean(precision)
        model_recall_macro = np.mean(recall)
        print("Macc_PARTIAL : %.2f" % (model_accuracy_partial_correct*100.0))
        print("Macc         : %.2f" % (model_accuracy*100.0))
        print("Mprec (macro): %.2f" % (model_precision_macro*100.0))
        print("Mrec (macro) : %.2f" % (model_recall_macro*100.0))


    # Perform a per-sample classification of whether it belongs to a class or not
    # This is done by calculating the distance to the expected value (mode) of the
    # Gaussian distribution of output weights for each class, and choosing the shortest
    # distance.
    def bin_class_per_sample(self, instances, limit=200, adv_detect=True, vendor_only=False):
        test_samples_x, test_samples_y = instances.next_batch(False, limit)
        num_samples = len(test_samples_x)
        num_classes = instances.mapping.size+1 if adv_detect else instances.mapping.size  # If adv_detect: use extra class for unknown

        # Metrics
        predicted_y = []
        true_y = []
        true_y_vis = []
        confusion_matrix = np.zeros(shape=(num_classes,num_classes))

        print('[+] Predicting %d samples...' % num_samples)
        for i in range(0, num_samples):
            true_class_map = np.argmax(test_samples_y[i])  # Get the true map ID from the dataset
            predicted_class_map,_ = self._predict([test_samples_x[i]], adv_detect)  # Get the map ID according to the model
            true_class = instances.mapping.map_to_lora_id(true_class_map) # Get the LoRa ID from the dataset
            predicted_class = self.instances_mapping.map_to_lora_id(predicted_class_map) # Get the LoRa ID according to the model
            if predicted_class is None:
                predicted_class = -1
            if vendor_only:
                true_class = instances.mapping.lora_id_to_vendor_id(true_class)
                predicted_class = self.instances_mapping.lora_id_to_vendor_id(predicted_class)
            predicted_y.append(predicted_class)

            if adv_detect:
                if not true_class in self.instances_mapping.keys():  # self.instances_mapping = learned mapping from model
                    true_y_vis.append(true_class)
                    true_y.append(-1)
                    confusion_matrix[0, predicted_class_map+1] += 1  # Make it so adv class(=-1) becomes class 0
                else:
                    true_y.append(true_class)
                    true_y_vis.append(true_class)
                    confusion_matrix[true_class_map+1, predicted_class_map+1] += 1
            else:
                true_y.append(true_class)
                true_y_vis.append(true_class)
                confusion_matrix[true_class_map, predicted_class_map] += 1

        print("[+] True classes encountered: %s" % len(set(true_y)))
        self._print_statistics(confusion_matrix)  # For debugging
        assert(np.sum(confusion_matrix) == num_samples)

        metrics = utilities.get_eval_metrics_percent(true_y, predicted_y)
        utilities.print_metrics(metrics)

        print('[+] Plotting output weights for first %d samples' % num_samples)
        self._plot_output_weights_2d(test_samples_x, true_y_vis, predicted_y, instances, metrics)

    def bin_class_per_frame(self, frame, symbol_length, adv_detect=True):
        dataset = GNURadioDataset(frame, symbol_length)
        data_x = [np.array(x["iq"]).flatten() for x in dataset.get()]

        map_id, all_map_id_predictions = self._predict(data_x, adv_detect)

        # Debug
        lora_id_predictions = []
        for map_id in all_map_id_predictions:
            lora_id_predictions.append(self.instances_mapping.map_to_lora_id(map_id))
        print("%s: %s" % (FLAGS.clustering, str(lora_id_predictions)))

        return stats.mode(lora_id_predictions)[0][0]

    def _labels_to_tsv_file(self, labels, mapping, out=None):
        result = ""
        for i in range(len(labels)):
            result += str(mapping.oh_to_lora_id(labels[i])) + "\n"

        if out:
            with open(out, "w") as f:
                f.write(result)

    # TODO: Actually doesn't need to be inside the Classifier class
    def visualize_embeddings(self, instances, limit=200, train=True):
        print("[+] Gathering instances...")
        samples_x, samples_y = instances.next_batch(train, limit)
        weights = net.get_output_weights(samples_x)

        print(Fore.GREEN + Style.BRIGHT + "[+] Visualizing embeddings for %d samples" % limit)

        embeddings_instances = tf.Variable(tf.stack(samples_x, axis=0), trainable=False, name='instances')
        embeddings_weights = tf.Variable(tf.stack(weights, axis=0), trainable=False, name='weights')
        self.sess.run(tf.variables_initializer([embeddings_instances, embeddings_weights]))
        embeddings_saver = tf.train.Saver([embeddings_instances, embeddings_weights])
        embeddings_writer = tf.summary.FileWriter(FLAGS.logdir + '/projector', self.sess.graph)

        conf = projector.ProjectorConfig()

        # Add embeddings
        # Instances
        e = conf.embeddings.add()
        e.tensor_name = embeddings_instances.name
        self._labels_to_tsv_file(samples_y, instances.mapping, out=FLAGS.logdir + '/projector/metadata.tsv')
        e.metadata_path = FLAGS.logdir + '/projector/metadata.tsv'
        # Generate sprite, save to tmp and assign here
        #e.sprite.image_path = FLAGS.logdir +
        #e.sprite.single_image_dim.extend([1024, 768])

        # Weights
        e = conf.embeddings.add()
        e.tensor_name = embeddings_weights.name
        self._labels_to_tsv_file(samples_y, instances.mapping, out=FLAGS.logdir + '/projector/metadata.tsv')
        e.metadata_path = FLAGS.logdir + '/projector/metadata.tsv'

        projector.visualize_embeddings(embeddings_writer, conf)
        embeddings_saver.save(self.sess, FLAGS.logdir + '/projector/model_embeddings.ckpt')

    # Calculates distance between pairs of centroids
    def _intercluster_distance(self, centroids, method='min'):
        num_centroids = len(centroids)

        if not method in ['min','mean','mean_of_min']:
            print("[!] Warning: _intercluster_distance: no such method '%s'. Defaulting to 'min'." % method)
            method = 'min'

        print("[+] Finding %s distance between %d centroids" % ("minimum" if method == 'min' else ("mean" if method == "mean" else "mean of minimum"), num_centroids))
        if method == 'mean_of_min':
            minimums = []
            for i in range(len(centroids)):
                first = centroids[i]
                distances = []
                for j in range(len(centroids)):
                    if i == j:
                        continue
                    second = centroids[j]
                    distance = np.linalg.norm(second - first)
                    distances.append(distance)
                minimums.append(np.min(distances))

            return np.mean(minimums)
        else:
            distances = []
            for pair in combinations(range(num_centroids), 2):
                distance = np.linalg.norm(centroids[pair[0]] - centroids[pair[1]])
                distances.append(distance)

            if method == 'min':
                return np.min(distances)
            elif method == 'mean':
                return np.mean(distances)

    # Convert predicted labels to real labels so that they can be compared
    # in terms of accuracy
    def _get_zeroshot_labels(self, dbscan_labels, real_labels):
        counts = defaultdict(list)

        # Get dbscan labels for each real label
        for i in range(len(real_labels)):
            counts[real_labels[i]].append(dbscan_labels[i])

        # Get most frequent dbscan label for each real label
        # and use dbscan label as key for lookup dict
        keys = {}
        keys_counts = defaultdict(lambda: 0)
        for key in set(real_labels):
            mode_count = stats.mode(counts[key])[1][0]
            mode_value = stats.mode(counts[key])[0][0]
            if mode_count > keys_counts[mode_value]:
                keys[mode_value] = key
                keys_counts[mode_value] = mode_count

        # Apply lookup dict to transform labels
        result = []
        for i in range(len(dbscan_labels)):
            try:
                result.append(keys[dbscan_labels[i]])
            except KeyError:  # No prevalent real label for this dbscan label found, so use outlier
                result.append(-1)

        return np.array(result)

    def classify_zeroshot(self, instances, limit=40, threshold_outlier=0.0001, vendor_only=False):
        num_mixtures = len(self.std)
        mixtures = []
        outlier_points = []
        outlier_labels = []

        print("[+] Gathering test samples")
        test_samples_x, test_samples_y = instances.next_batch(False, limit)
        num_samples = len(test_samples_x)

        print("[+] Building %d gaussian mixtures based on trained parameters" % num_mixtures)
        from scipy.stats import multivariate_normal
        for i in range(num_mixtures):
            # TF method
            #g = tf.contrib.distributions.Normal(mu=self.expected_values[i], sigma=self.std[i])

            # Numpy method
            #g = multivariate_normal(self.expected_values[i], np.diag(np.power(self.std[i], 2)))
            g = NumpyNormWrapper(mu=self.expected_values[i], sigma=self.std[i])
            mixtures.append(g)

        print("[+] Finding inter-cluster distance of training samples")
        icd = self._intercluster_distance(self.expected_values, method='mean_of_min')
        print("[+] ICD is %f" % icd)

        print("[+] Calculating weights and probabilities")
        weights, probabilities = self._predict_zeroshot(test_samples_x)

        print("[+] Calculating marginals")
        marginals = np.zeros(shape=(num_samples, num_mixtures))

        for i in range(num_samples):
            point = weights[i]
            pi = probabilities[i]

            for j in range(num_mixtures):
                # TF method
                #marginals[i] += pi[j] * self.sess.run(mixtures[j].pdf(point))

                # Numpy method
                marginals[i] += pi[j] * mixtures[j].pdf(point)

            outlier = False
            for j in range(num_mixtures):
                if marginals[i][j] < threshold_outlier:
                    outlier = True
                    outlier_points.append(point)
                    lora_id = instances.mapping.oh_to_lora_id(test_samples_y[i])
                    if vendor_only:  # If we only care about classifying correct vendor
                        lora_id = instances.mapping.lora_id_to_vendor_id(lora_id)
                    outlier_labels.append(lora_id)
                    break

            #print("%02d: %s | marg:%s, pi:%s, meanmarg:%s (%d/%d)" % (instances.mapping.oh_to_lora_id(test_samples_y[i]), str(outlier), str(marginals[i]), pi, str(np.mean(marginals[i])),i,num_samples))

        print("[+] Finding nearest neighbors based on inter-cluster distance of training data")
        db = DBSCAN(eps=icd, min_samples=1).fit(outlier_points)
        zeroshot_labels = self._get_zeroshot_labels(db.labels_, outlier_labels)
        guess_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print(db.labels_)
        print(np.array(outlier_labels))
        print(zeroshot_labels)
        print(guess_clusters)

        metrics = utilities.get_eval_metrics_percent(outlier_labels, zeroshot_labels)
        utilities.print_metrics(metrics)

        # Reduce dimensionality of weights to 2
        tsne = TSNE(n_components=2, init='pca', n_iter=5000)
        vis = tsne.fit_transform(outlier_points)
        visualization.plot_weights(vis, outlier_labels, zeroshot_labels, None, None, instances.mapping, tag=self.name+"-zero-w", metrics=metrics, tf=True)

# Class to make numpy normal distribution act the same as TF normal distribution
class NumpyNormWrapper():
    def __init__(self, mu, sigma):
        from scipy.stats import norm
        if len(mu) != len(sigma):
            raise Exception

        # Initialize
        self.num_distributions = len(mu)
        self.distributions = []

        for i in range(self.num_distributions):
            self.distributions.append(norm(mu[i], sigma[i]))

    def pdf(self, values):
        if len(values) != self.num_distributions:
            raise Exception

        result = []
        for i in range(self.num_distributions):
            result.append(self.distributions[i].pdf(values[i]))

        return np.array(result)

# ----------------------------------------------------
# ML layers
# ----------------------------------------------------
class NNLayer():
    def __init__(self, inputs, Wshape, bshape, name=''):  # input features and outputs
        self.inputs = inputs
        self.Wshape = Wshape
        self.bshape = bshape
        self.name = name

        # Define model
        self.W = tf.Variable(tf.random_normal(Wshape))  # Filter kernel
        self.b = tf.Variable(tf.random_normal(bshape))

        # Input: [batch, height, width, channels]
        # Kernel: [filter_height, filter_width, in_channels, out_channels]
        k = FLAGS.pooling_kernel_width
        s = 1
        self.conv = tf.nn.conv2d(inputs, self.W, strides=[1, 1, s, 1], padding='SAME') #  Convolution Layer
        self.conv_b = tf.nn.bias_add(self.conv, self.b)  # Convolution layer bias
        self.relu = tf.nn.relu(self.conv_b)  # ReLU activation layer
        self.h = tf.nn.max_pool(self.relu, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='SAME')  # Max pooling layer (down-sampling)

        self.saver = tf.train.Saver([self.W, self.b])

class LinearLayer():
    def __init__(self, inputs, num_inputs, num_outputs, name='', init_zero=False):  # input features and outputs
        self.inputs = inputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.name = name

        # Define model
        if init_zero:
            self.W = tf.Variable(tf.zeros([num_inputs, num_outputs]))
            self.b = tf.Variable(tf.zeros([num_outputs]))
        else:
            self.W = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
            self.b = tf.Variable(tf.random_normal([num_outputs]))
        self.h = tf.add(tf.matmul(inputs, self.W), self.b)

        self.saver = tf.train.Saver([self.W, self.b])

class LinearReluLayer():
    def __init__(self, inputs, num_inputs, num_outputs, name='', init_zero=False):  # input features and outputs
        self.inputs = inputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.name = name

        # Define model
        if init_zero:
            self.W = tf.Variable(tf.zeros([num_inputs, num_outputs]))
            self.b = tf.Variable(tf.zeros([num_outputs]))
        else:
            self.W = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
            self.b = tf.Variable(tf.random_normal([num_outputs]))
        self.h = tf.nn.relu(tf.add(tf.matmul(inputs, self.W), self.b))

        self.saver = tf.train.Saver([self.W, self.b])

class MixtureLayer():
    def __init__(self, inputs, num_inputs, num_mixtures, mixture_type='gaussian', name='', init_zero=False):  # input features and outputs
        self.inputs = inputs
        self.num_inputs = num_inputs
        self.num_mixtures = num_mixtures
        self.num_components = 3
        self.num_outputs = self.num_mixtures * self.num_components
        self.name = name

        # Define model
        if init_zero:
            self.W = tf.Variable(tf.zeros([self.num_inputs, self.num_outputs]))
            self.b = tf.Variable(tf.zeros([self.num_outputs]))
        else:
            self.W = tf.Variable(tf.random_normal([self.num_inputs, self.num_outputs], stddev=0.1))
            self.b = tf.Variable(tf.random_normal([self.num_outputs], stddev=0.1))

        # Mixture model hypothesis
        tanh_inputs = tf.nn.tanh(inputs)
        self.h = tf.add(tf.matmul(tanh_inputs, self.W), self.b)

        self.saver = tf.train.Saver([self.W, self.b])

class LinearReluDropLayer():
    def __init__(self, inputs, num_inputs, num_outputs, keep, name=''):
        self.inputs = inputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.name = name

        # Define model
        self.W = tf.Variable(tf.random_normal([num_inputs, num_outputs]))
        self.b = tf.Variable(tf.random_normal([num_outputs]))
        self.h = tf.add(tf.matmul(inputs, self.W), self.b)
        self.h = tf.nn.relu(self.h)
        self.h = tf.nn.dropout(self.h, keep)

        self.saver = tf.train.Saver([self.W, self.b])

class SoftmaxLayer():
    def __init__(self, inputs, num_inputs, num_outputs, name=''):  # input features and outputs
        self.inputs = inputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.name = name

        # Define model
        self.W = tf.Variable(tf.zeros([num_inputs, num_outputs]))
        self.b = tf.Variable(tf.zeros([num_outputs]))
        self.h = tf.nn.softmax(tf.add(tf.matmul(inputs, self.W), self.b)) # Hypothesis

        # If requested, save weights W and biases b
        self.saver = tf.train.Saver([self.W, self.b])

# ----------------------------------------------------
# Standalone run code
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow based fingerprinting of devices implementing the LoRa PHY layer')
    parser.add_argument('action', type=str, choices=['train', 'test', 'train_embeddings', 'test_embeddings', 'zeroshot'], help='Action to perform')
    parser.add_argument('configfile', type=str, help='Path to the config file to use')
    parser.add_argument('--dt', dest='distance_threshold', type=str, help='Distance threshold to determine whether a device is an adversary. Set to "auto" to calculate automatically', default='auto')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--save', dest='save', action='store_true', default=False, help='Save trained network')
    parser.add_argument('--adv', dest='adv', action='store_true', default=False, help='Treat excluded classes as attackers')
    parser.add_argument('--vendor', dest='vendor', action='store_true', default=False, help='Test on chip model only')
    parser.add_argument('--natural', dest='natural', action='store_true', default=False, help='Natural sorting')
    args, unknown = parser.parse_known_args()

    # Argument preprocessing]
    if args.distance_threshold != 'auto':  # Define distance threshold
        args.distance_threshold = float(args.distance_threshold)

    # Conf stuff
    load_conf(args.configfile)
    print_conf(cp)

    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)  # Clean tmp dir

    if type(FLAGS.exclude_classes) == str and FLAGS.exclude_classes != '':  # Exclude classes from training
        exclude_classes = [int(x) for x in FLAGS.exclude_classes.split(',')]
    else:
        exclude_classes = []

    # Let's go
    if args.action == 'train':
        print("[+] Excluding %s" % str(exclude_classes))
        instances = Instances(limit=FLAGS.limit, exclude_classes=exclude_classes, name="train")
        if cp.get("DEFAULT", "classifier") == 'svm':
            net = SVM(name=FLAGS.model_name)
        else:
            net = Classifier(num_inputs=instances.limit, num_classes=instances.mapping.size, name=FLAGS.model_name, modeltype=ModelType.str2type(FLAGS.classifier))
        net.train(instances, batch_size=FLAGS.batch_size)
        if args.save:
            net.save()
        net.bin_class_per_sample(instances, limit=1000, adv_detect=False, vendor_only=False)  # Never adv detect during training
        net.visualize_embeddings(instances, limit=1000, train=True)
    elif args.action == 'test':
        instances = Instances(limit=FLAGS.limit, exclude_classes=[], name="test")
        if cp.get("DEFAULT", "classifier") == 'svm':
            net = SVM.load()
        else:
            net = Classifier.load(0)
        print("[+] Testing...")
        net.bin_class_per_sample(instances, limit=1500, adv_detect=args.adv, vendor_only=args.vendor)
        net.visualize_embeddings(instances, limit=1000, train=False)
    elif args.action == 'train_embeddings':
        instances = Instances(limit=FLAGS.limit, exclude_classes=exclude_classes, name="train")
        print("[+] Loading model...")
        net = Classifier.load(0)
        net.visualize_embeddings(instances, limit=1000, train=True)
    elif args.action == 'test_embeddings':
        instances = Instances(limit=FLAGS.limit, exclude_classes=[], name="test")
        print("[+] Loading model...")
        net = Classifier.load(0)
        net.visualize_embeddings(instances, limit=1000, train=False)
    elif args.action == 'zeroshot':
        instances = Instances(limit=FLAGS.limit, exclude_classes=[], name="test")
        net = Classifier.load(0)
        net.classify_zeroshot(instances, FLAGS.num_zs_test_samples, vendor_only=args.vendor)
