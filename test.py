import os
import json

import matplotlib.pyplot as plt
import numpy as np
from MVHSC_Aux import create_instances, parser

S = parser()
IT = create_instances(S)

# 调用函数
results = IT.EV.process_json_files_multi_keys()

x = []
y = []
for i in range(4):
    x.append(results[i][1][IT.S["key_x"]])
    y.append(results[i][1][IT.S["key_y"]][1])

# Sorting both lists based on x values
sorted_pairs = sorted(zip(x, y))
x_sorted, y_sorted = zip(*sorted_pairs)

# Plotting
plt.plot(x_sorted, y_sorted, marker='o', linestyle='-', label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot of Sorted Data')
plt.grid(True)
plt.legend()
plt.show()
