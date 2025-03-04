import os
import json
import pandas as pd

# 要提取的字段和嵌套字段
fields = [
    "hypergrad_method", "seed_num", "proj_x", "proj_y", 
    "orth_x", "orth_y", ("best_ul_acc", 0), ("best_ul_acc", 1), 
    ("best_ul_nmi", 0), ("best_ul_nmi", 1)
]

# 获取当前目录中的所有 JSON 文件
json_files = [f for f in os.listdir('logs2') if f.endswith('.json')]

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

# 保存为 CSV 或输出为行排列的格式
output_file = "logs2/extracted_data.csv"
df.to_csv(output_file, index=False)
print(f"Data has been extracted and saved to {output_file}")
