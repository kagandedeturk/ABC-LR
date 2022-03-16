# ARTIFICIAL BEE COLONY ALGORITHM WITH LOGISTIC REGRESSION (ABC-LR)

ABC-LR is a binary classification method in which the ABC algorithm is used instead of the Gradient Descent algorithm to train the weights in the Logistic Regression classification model. The purpose of the ABC algorithm in the ABC-LR method is to minimize the value of the cost function. The ABC algorithm is a very popular metaheuristic method that can search for solutions both locally and globally in the solution space. In addition, it has been shown in the study that the ABC-LR classification method achieves superior classification success compared to the LR classification method.

Although the ABC-LR classification method can handle complex and high-dimensional data, its runtime can be high. Therefore, CPU and GPU parallelized versions of the ABC-LR classification method are presented here, and significant improvements in runtime can be achieved.

ABC-LR is written in Python3 and continuously tested with Python 3.7 and 3.10.

## Installation

Install ABC-LR via PyPI:

```
pip install abcLR
```

Or alternatively, clone the environment:

```
git clone https://github.com/kagandedeturk/ABC-LR.git
```

## CPU Version Usage

```py

import numpy as np
parallelType = np
from abcLR import ABC_LR_Model
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1,1)

lb = -16
ub = 16
evaluationNumber = 80000
# FVS = trainData.shape[1]
limit = 50
P = 60
MR = 0.3
L2 = 0

model = ABC_LR_Model(lb=lb, ub=ub, evaluationNumber=evaluationNumber, limit=limit, P=P, MR=MR, L2=L2, parallelType=parallelType)
#start_time = dt.datetime.now()
model.fit(X, y)
#print(f"Run time: {dt.datetime.now()-start_time}")
print(f"Result: {model.score(X, y)}")

```

## GPU Version Usage

```py
import cupy as cp
parallelType = cp
from abcLR import ABC_LR_Model
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

X = parallelType.array(X)
y = parallelType.array(y.reshape(-1,1))

lb = -16
ub = 16
evaluationNumber = 80000
# FVS = trainData.shape[1]
limit = 50
P = 60
MR = 0.3
L2 = 0

model = ABC_LR_Model(lb=lb, ub=ub, evaluationNumber=evaluationNumber, limit=limit, P=P, MR=MR, L2=L2, parallelType=parallelType)
#start_time = dt.datetime.now()
model.fit(X, y)
#print(f"Run time: {dt.datetime.now()-start_time}")
print(f"Result: {model.score(X, y)}")

```

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the 3-clause BSD license (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the 3-clause BSD license
along with this program (see LICENSE file).
If not, see [here](https://opensource.org/licenses/BSD-3-Clause).

Copyright (c) 2022, Bilge Kagan Dedeturk (kagandedeturk@gmail.com)
