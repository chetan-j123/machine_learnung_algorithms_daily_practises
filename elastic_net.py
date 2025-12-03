import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
df=pd.read_csv("elasticnet.csv")
model_elasticnet=ElasticNet(alpha=1,l1_ratio=0.5)
