import os
import json
import pandas as pd

# 指定子文件夹名称
subfolder_name = "logs2"  # 替换为实际子文件夹名称
output_file = "logs2_data.tsv"

# 要提取的字段和嵌套字段
fields = [
    "hypergrad_method", "seed_num", "proj_x", "proj_y",
    "orth_x", "orth_y", ("best_ul_acc", 0), ("best_ul_acc", 1),
    ("best_ul_nmi", 0), ("best_ul_nmi", 1)
]

# 获取子文件夹中的所有 JSON 文件
subfolder_path = os.path.join(os.getcwd(), subfolder_name)
json_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.json')]

# 用于存储提取的数据
data = []

# 遍历所有 JSON 文件并提取字段
for file in json_files:
    with open(file, 'r') as f:
        try:
            content = json.load(f)
            row = {}
            for field in fields:
                if isinstance(field, tuple):  # 处理嵌套字段
                    key, index = field
                    value = content.get(key, None)
                    row[f"{key}[{index}]"] = value[index] if isinstance(value, list) and len(value) > index else None
                else:  # 处理普通字段
                    row[field] = content.get(field, None)
            data.append(row)
        except json.JSONDecodeError:
            print(f"Error decoding {file}, skipping.")

# 将数据存储到 DataFrame 中
df = pd.DataFrame(data)

# 保存为制表符分隔的文件
df.to_csv(output_file, index=False, sep='\t')
print(f"Data has been extracted and saved to {output_file}")