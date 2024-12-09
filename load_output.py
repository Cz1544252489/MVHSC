import json

# 打开文件并逐行读取内容

with open('output2.txt', 'r', encoding='utf-8') as file:
    # 读取所有行
    lines = file.readlines()

# 创建一个列表，用于存储解析后的每行内容
parsed_data = []

# 遍历每一行并解析制表符分隔的数据
for line in lines:
    # 去掉行末换行符并分割制表符
    parsed_line = line.strip().split('\t')
    # 将解析后的行添加到列表中
    parsed_data.append(parsed_line)

column_1 = [int(row[6]) for row in parsed_data]
column_2 = [float(row[7]) for row in parsed_data]
column_3 = [float(row[8]) for row in parsed_data]
column_4 = [float(row[9]) for row in parsed_data]

set = { "s1": 0, "s2": 0, "s3": 0, "s4": 0,
        "c1": column_1, "c2": column_2, "c3": column_3, "c4": column_4}
result = {}

for ty in range(0,8,1):
    for i in range(1, 5, 1):
        set[f"s{i}"] = 0
        for line in range(ty,800,8):
            set[f"s{i}"] = set[f"s{i}"] + set[f"c{i}"][line]

        result[f"{ty}a{i}"] = set[f"s{i}"]/100

with open("ana2.json","w") as file:
    json.dump(result, file, indent=4)


print("aa")