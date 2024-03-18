# ARTIFICIAL BEE COLONY ALGORITHM WITH LOGISTIC REGRESSION (ABC-LR)

ABC-LR is an innovative binary classification technique that leverages the Artificial Bee Colony (ABC) algorithm, substituting the traditional Gradient Descent approach for training weights in the Logistic Regression model. This method primarily aims to minimize the cost function's value through the ABC algorithm. Renowned for its dual capacity for local and global solution space exploration, the ABC algorithm significantly enhances this classification method.

Our research indicates that ABC-LR outperforms the standard Logistic Regression (LR) method in terms of classification accuracy. This is particularly notable in managing complex and high-dimensional datasets.

However, it's important to note that the ABC-LR method may exhibit increased runtime, especially with complex data. To address this, we offer both CPU and GPU parallelized versions of the ABC-LR classification method, ensuring substantial runtime efficiency gains.

ABC-LR is developed in Python3 and undergoes regular testing on Python versions 3.7 and 3.10, ensuring consistent performance and compatibility.

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
y = parallelType.array(y)

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

## Miscellaneous

abcLR has been developed by [Dr. Bilge Kagan Dedeturk](https://avesis.erciyes.edu.tr/bilgededeturk), an Assistant Professor in the Department of Software Engineering at Erciyes University. For any queries or further information, Dr. Dedeturk can be contacted at kagandedeturk@gmail.com or bilgededeturk@erciyes.edu.tr.

### Reporting Issues

Encountered a bug? Your contributions to refining abcLR are invaluable. Please report any issues you find at our GitHub repository: [abcLR Issues](https://github.com/kagandedeturk/ABC-LR/issues).

### Citation

Are you incorporating abcLR into your research? We'd be honored. Please refer to the following studies in your citations to acknowledge the use of abcLR:

```
@article{DEDETURK2020106229,
    title = {Spam filtering using a logistic regression model trained by an artificial bee colony algorithm},
    journal = {Applied Soft Computing},
    volume = {91},
    pages = {106229},
    year = {2020},
    issn = {1568-4946},
    doi = {https://doi.org/10.1016/j.asoc.2020.106229},
    author = {Bilge Kagan Dedeturk and Bahriye Akay}
}

@Inbook{Dedeturk2021,
    author = {Dedeturk, Bilge Kagan and Akay, Bahriye and Karaboga, Dervis},
    editor = {Carbas, Serdar and Toktas, Abdurrahim and Ustun, Deniz},
    title = {Artificial Bee Colony Algorithm and Its Application to Content Filtering in Digital Communication},
    bookTitle = {Nature-Inspired Metaheuristic Algorithms for Engineering Optimization Applications},
    year = {2021},
    publisher = {Springer Singapore},
    address = {Singapore},
    pages = {337--355},
    doi = {10.1007/978-981-33-6773-9_15}
}

@article{KOLUKISA2024103808,
    title = {An efficient network intrusion detection approach based on logistic regression model and parallel artificial bee colony algorithm},
    journal = {Computer Standards & Interfaces},
    volume = {89},
    pages = {103808},
    year = {2024},
    issn = {0920-5489},
    doi = {https://doi.org/10.1016/j.csi.2023.103808},
    author = {Burak Kolukisa and Bilge Kagan Dedeturk and Hilal Hacilar and Vehbi Cagri Gungor}
}
```

Copyright (c) 2022, Bilge Kagan Dedeturk (kagandedeturk@gmail.com).
