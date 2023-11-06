import random


class buffer():
    def __init__(self,dim_buffer):
        self.buf = []
        self.dim_buffer = dim_buffer

    def buffer_update(self,new_datasamples):
        if len(self.buf) < self.dim_buffer: #in this case the buffer is not yet full
            if len(new_datasamples) <= self.dim_buffer-len(self.buf):
                self.buf = self.buf + new_datasamples
            else:
                num = len(new_datasamples) - (self.dim_buffer-len(self.buf))
                self.buf=self.buf[num:]
                self.buf = self.buf + new_datasamples
        else:
            num = len(new_datasamples)
            self.buf=self.buf[num:]
            self.buf = self.buf + new_datasamples

    def get_samples(self,total):
        if len(self.buf) < self.dim_buffer:
            if len(self.buf) < total:
                return self.buf
            else:
                return self.buf[-total:]
        else:
            return random.sample(self.buf,k=total)
        
    def buffer_initialization(self,list_samples):
        if len(list_samples) <= self.dim_buffer:
            self.buf = []
            self.buf = list_samples
        else:
            self.buf = []
            self.buf = list_samples[:self.dim_buffer]

    def buffer_resize(self,new_dimension):
        if new_dimension < self.dim_buffer:
            self.dim_buffer = new_dimension
            self.buf = self.buf[:new_dimension]
        else:
            self.dim_buffer = new_dimension
