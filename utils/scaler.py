import numpy as np

class MinMaxScaler():
    """The normalizer, given the data shape is LxN
    """
    def __init__(self, if_col_norm):
        self.if_col_norm = if_col_norm
        self.max = None
        self.min = None

    def fit(self, data):
        if self.if_col_norm:
            self.max = np.max(data, axis=0)
            self.min = np.min(data, axis=0)
        else:
            self.max = np.max(data)
            self.min = np.min(data)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)
    
    def inver_transform(self, scaled_data):
        return scaled_data * (self.max - self.min + 1e-8) + self.min
    
    def inver_transform_col(self, scaled_data):
        return scaled_data * (self.max[None, None, :] - self.min[None, None, 1] + 1e-8) + self.min


class ZScoreScaler():
    '''
    Z-score normalization: transforming the data to have a mean of zero
    and a standard deviation of one. 
    
    Attributes:
        mean:
        std:
        target_channel: 
    '''
    def __init__(self, if_col_norm):
        self.if_col_norm = if_col_norm
        self.max = None
        self.std = None
    
    def fit(self, data):
        if self.if_col_norm:
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True)
        else:
            self.mean = np.mean(data)
            self.std = np.std(data)
    
    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
    
    def inver_transform(self, scaled_data):
        return scaled_data * (self.std + 1e-8) + self.mean
    
    def inver_transform_col(self, scaled_data):
        return scaled_data * (self.std + 1e-8) + self.mean
    