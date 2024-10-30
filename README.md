# Ramp-Score and time distortion index 
Two metrics to quantify the solar ramp events prediction accuracy.

## How to use the code
```bash
pip install tslearn
# example
import pandas as pd
import numpy as np
from Metrics_solarforecasting import ramp_score, TDI
data = pd.read_csv('your data.csv')
y_true = data['true']
y_pred = data['pred']
results = {}
results['ramp_score'] = ramp_score(y_pred, y_true)
results['TDI'] = TDI(y_pred, y_true)
```

