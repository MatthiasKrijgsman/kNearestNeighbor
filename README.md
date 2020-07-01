# k-Nearest Neighbor
An implementation of the k-Nearest Neighbor algorithm in Python using the Manhattan Distance function.
$\newline dist(\vec{v_1}, \vec{v_2}) = \sqrt{(x_2 - x_1)^2+(y_2-y_1)^2}$

### Usage
```python
import kNearestNeighbor as kNN
import numpy as np

# Example data from BMI labels
height = np.array([ 180, 175, 194, 178, 182, 173, 194, 180, 175, 179, 182, 187])
weight = np.array([ 65, 73, 102, 73, 65, 82, 130, 54, 55, 51, 65, 97 ])
weight_labels = np.array([ 'Normal', 'Normal', 'Overweight', 'Normal', 'Normal', 'Overweight', 'Overweight', 'Normal', 'Underweight', 'Underweight', 'Underweight', 'Overweight'])

kNN.classify(
    row = [183, 55],
    features = [height, weight],
    labels = weight_labels
)
```
Note that feature values must be quantitative. So strings and objects are not permitted.

### Output
```python
>>> "Underweight"
```