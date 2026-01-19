import scipy.io
import numpy as np

# load .mat
mat_data = scipy.io.loadmat('mat_name')

key = [k for k in mat_data.keys() if not k.startswith("__")][0]
data = mat_data[key] 

# Keep first 10,000 row
# data = data[:10000, :]

# Set 0 when smaller than 0
data[data < 0] = 0

num_rows = data.shape[0]
image_names = [f"{i:012d}" for i in range(1, num_rows + 1)]  # 生成 000000000001, 000000000002, ...

# Save as text
with open('./dataset/OC_OAM_spectrum_bit13/train.txt', 'w') as f:
    for name, row in zip(image_names, data):
        row_str = ' '.join(f"{x:.6f}" for x in row) 
        f.write(f"{name} {row_str}\n") 

print("数据已成功保存")
