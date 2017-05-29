# Generic cache based on a standard dictionary. Used to prevent recalculations
# of the same data when iterating over training data multiple times.
class GenericCache():
    def __init__(self, name="", max_elements=0):
        self.max_elements = max_elements
        self._internal = {}
        self._g = self._generator()
        self.name = name

    def store(self, identifier, element):
        if self.max_elements and len(self._internal.keys()) >= self.max_elements:
            return False

        self._internal[identifier] = element
        return True

    def get(self, identifier):
        try:
            return self._internal[identifier]
        except KeyError:
            return None

    def _generator(self):
        for v in self._internal.values():
            yield v

    # Get an element from the cache values
    def next(self):
        return self._g.next()

    def rewind(self):
        self._g = self._generator()

    def keys(self):
        return self._internal.keys()

    def values(self):
        return self._internal.values()

    def __len__(self):
        return len(self._internal)

    # Test that fails if any of the keys in cache1 are also in cache2.
    # In other words, it makes sure that cache1 and cache2 are disjunct.
    @staticmethod
    def assert_disjunction(cache1, cache2):
        k_c1 = set(cache1.keys())
        k_c2 = set(cache2.keys())

        i = k_c1 & k_c2

        assert(len(i) == 0)

# Unit tests
# TODO use the unittest model
t = GenericCache()
t.store('a', 1)
t.store('b', 2)
assert(len(t) == 2)
assert(t.get('a') == 1)
assert(t.get('b') == 2)

a = list()
count = 0
for i in range(11):
    try:
        a.append(t.next())
    except StopIteration:
        assert(sorted(a) == sorted(t.values()))
        count += 1
        t.rewind()
        a = list()
        a.append(t.next())
assert(count == 10/len(t))  # The 11th iteration should trigger a StopIteration, bringing to count to 5
