class Trie:

    def __init__(self):
        self.lookup = {}

    def insert(self,word):
        tree = self.lookup
        for char in word:
            if char not in tree:
                tree[char] = {}
            tree = tree[char]

        #word ends sign
        tree['#'] = {}  #OR tree['#'] = {}

    def search(self,word):
        tree = self.lookup
        for c in word:
            if c not in tree:
                return False
            tree = tree[c]
        if '#' in tree:
            return True
        return False

    def startswith(self, prefix):
        tree = self.lookup
        for c in prefix:
            if c not in tree:
                return False
            tree = tree[c]
        return True





