from skimage.metrics import structural_similarity as ssim
from functools import partial
import numpy as np
from sklearn import metrics as skmetrics

class SSIM:
    def __init__(self, data_range=255):
        self.ssim_metric = partial(ssim, data_range=data_range)

        self.scores = []

    def name(self):
        return 'SSIM'

    def accumulate(self, outputs, targets):
        for output, target in zip(outputs, targets):
            output = (output[0]*255).astype(np.uint8)
            target = (target[0]*255).astype(np.uint8)

            score = self.ssim_metric(output, target)

            self.scores.append(score)
    
    def calculate_score(self):
        return np.mean(self.scores)
    
class MSE:
    def __init__(self, data_range=255):
        self.mse = skmetrics.mean_squared_error

        self.scores = []

    def name(self):
        return 'MSE'

    def accumulate(self, outputs, targets):
        for output, target in zip(outputs, targets):
            score = self.mse(output[0], target[0])
            self.scores.append(score)
    
    def calculate_score(self):
        return np.mean(self.scores)