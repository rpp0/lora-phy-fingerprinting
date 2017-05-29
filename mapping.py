import numpy as np

# Mapping from device ID to vendor ID
HARDCODED_VENDOR_DICT = {  # TODO: Make this non-hardcoded in a distant future
    # 1: RN24
    # 2: SX
    # 3: RF96
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 3,
    6: 3,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
}

# The Mapping class provides a mapping between a LoRa ID from the dataset, and one-hot label vector used in Tensorflow
class Mapping():
    def __init__(self, lora_ids, exclude_classes=[], name="mapping"):
        self.name = name
        self._dict = {}
        self._lid = 0
        self.size = 0

        # Fill mapping
        self.update(lora_ids, exclude_classes)

    def add(self, lora_id):
        if not lora_id in self._dict.keys():
            self._dict[lora_id] = self._lid
            self._lid += 1
            self.size += 1

    def update(self, lora_ids, exclude_classes):
        for lora_id in lora_ids:
            if not lora_id in exclude_classes:
                self.add(lora_id)

    def keys(self):
        return self._dict.keys()

    def display(self):
        if len(self._dict.keys()) > 0:
            print("[+] Mapping from LoRa to TF is:")
            for lora_id in self._dict.keys():
                print("\t LoRa " + str(lora_id) + " -> " + str(self.lora_id_to_oh(lora_id)) + " (" + str(self.lora_to_map_id(lora_id)) + ")")
        else:
            print("[-] Warning: no mapping created yet from LoRa ID to Mapping ID / one hot vector.")

    def lora_to_map_id(self, lora_id):
        try:
            return self._dict[lora_id]
        except KeyError:
            return None

    def map_to_lora_id(self, map_id):
        for lora_id in self._dict.keys():
            if self._dict[lora_id] == map_id:
                return lora_id
        return None

    def oh_to_lora_id(self, oh):
        map_id = np.argmax(oh)
        return self.map_to_lora_id(map_id)

    def lora_id_to_oh(self, lora_id):
        map_id = self.lora_to_map_id(lora_id)
        oh = [0] * self.size
        if not (map_id is None):
            oh[map_id] = 1
        return oh

    def lora_id_to_vendor_id(self, lora_id):
        global HARDCODED_VENDOR_DICT
        return HARDCODED_VENDOR_DICT[lora_id]
