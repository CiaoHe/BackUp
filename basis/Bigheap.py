class MaxHeap(object):
    """docstring for MaxHeap"""
    def __init__(self, max_size, fn):
        self.max_size = max_size
        self.fn = fn
        self._items = [None]*max_size
        self.size = 0

    @property
    def full(self):
        return self.size == self.max_size

    def value(self,idx):
        item = self._items[idx]
        if item is None:
            ret = -float('inf')
        else:
            ret = self.fn(item)
        return ret

    def add(self, item):
        if self.full:
            if self.fn(item)<self.value(0): 
                self._items[0] = item
                self._shift_down(0)
        else:
            self._items[self.size] = item
            self.size += 1
            self._shift_up(self.size-1)

    def pop(self):
        assert self.size>0, 'cannot pop item, the heap is empty'
        ret = self._items[0]
        self._items[0], self._items[self.size-1] = self._items[self.size-1],self._items[0]
        self.size-=1
        self._shift_down(0)
        return ret

    def _shift_up(self,idx):
        assert idx<self.size, "the parameter idx must be less than heap's size"
        parent = (idx-1)//2

        while parent >= 0 and self.value(parent)<self.value(idx):
            self._items[parent], self._items[idx] = self._items[idx], self._items[parent]
            idx = parent
            parent = (idx-1)//2

    def _shift_down(self,idx):
        child = 2*idx+1
        while child<self.size:
            #choose the biggest child(default left, if right>left, choose right)
            if child+1<self.size and self.value(child+1)>self.value(child):
                child+=1
            if self.value(idx) < self.value(child):
                self._items[child], self._items[idx] = self._items[idx], self._items[child]
            idx = child
            child = 2*idx + 1

#-----------------------------------------------------------
'''简化版'''
class MaxHeap(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self._items = [None]*max_size
        self.size = 0

    def add(self, num):
        if self.size == self.max_size:
            if num<self._items[0]:
                self._items[0] = num
                self._shift_down(0)
        else:
            self._items[self.size] = num
            self.size += 1
            self._shift_up(self.size - 1)

    def pop(self):
        res = self._items[0]
        self._items[0], self._items[self.size-1] = self._items[self.size-1], self._items[0]
        self.size-=1
        self._shift_down(0)
        return res

    def _shift_up(self,idx):
        parent = (idx-1)//2
        while parent>=0 and self._items[parent]<self._items[idx]:
            self._items[parent], self._items[idx] = self._items[idx], self._items[parent]
            idx = parent
            parent = (idx-1)//2

    def _shift_down(self,idx):
        child = 2*idx+1
        while child<self.size:
            if child+1<self.size and self._items[child+1]>self._items[child]:
                child+=1
            if self._items[child]>self._items[idx]:
                self._items[child], self._items[idx] = self._items[idx], self._items[child]
            idx = child
            child = 2*idx+1


    



    








































