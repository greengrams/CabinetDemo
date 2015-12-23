import numpy as np
class AlignmentDoor():
    init_data_len=300
    mean_mask_size=10
    start_sum_threshold=100
    
    init_data_buffer=[]
    mean_mask_buffer=[]
    noise_shift=0
    sum_val=0
    min_sum_val=0
    
    mask_val=0
    
    close_rate=0.25

    def clean_init(self):
        self.init_data_buffer=[]
        self.mean_mask_buffer=[]
        self.noise_shift=0
        self.sum_val=0
        self.min_sum_val=0

    def init_shift(self,data):
        init_flag=False
        if len(self.init_data_buffer)<self.init_data_len:
            # use 5-th axis to detect_start
            self.init_data_buffer.append(data[4])
        else:
            self.noise_shift=np.mean(self.init_data_buffer)
            init_flag=True
        return init_flag

    def is_start(self,data):
        start_flag=False
        
        if len(self.mean_mask_buffer)<self.mean_mask_size:
            self.mean_mask_buffer.append(self.sum_val)
        else:
            self.mean_mask_buffer.remove(self.mean_mask_buffer[0])
            self.mean_mask_buffer.append(self.sum_val)
            mean_before=np.mean(self.mean_mask_buffer[:int(self.mean_mask_size/2)+1])
            mean_after=np.mean(self.mean_mask_buffer[int(self.mean_mask_size/2)+1:])
            self.mask_val=mean_after-mean_before
            start_flag=(self.mask_val>self.start_sum_threshold)|(self.mask_val<-self.start_sum_threshold)
        return start_flag
    
    def update_sum_val(self,data):
        self.sum_val=self.sum_val+float(data[4])-self.noise_shift
        # get min_sum_val
        if self.sum_val<self.min_sum_val:
            self.min_sum_val=self.sum_val
    
    def is_close(self):
        return self.sum_val-self.min_sum_val>self.close_rate*abs(self.min_sum_val)
    
    def update_init_buffer(self,data):
        # update value to init_data_buffer
        self.init_data_buffer.remove(self.init_data_buffer[0])
        self.init_data_buffer.append(data[4])
    
    def get_sum_val(self):
        return self.sum_val
    
    def get_min_sum_val(self):
        return self.min_sum_val
        
    def get_noise_shift(self):
        return self.noise_shift
        
    def get_mask_val(self):
        return self.mask_val
    
    def open_init(self):
        self.noise_shift=np.mean(self.init_data_buffer)
    
    def clean_start(self):
        self.sum_val=0
        self.min_sum_val=0
        self.mean_mask_buffer=[]
        self.noise_shift=np.mean(self.init_data_buffer)
