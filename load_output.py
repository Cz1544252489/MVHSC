import json
from typing import Literal

def process_and_average(numbers, method:Literal["5%", "none"]="5%"):
    """
    对数字列表进行排序，去除最大和最小的5%，并计算剩余数字的平均值。

    :param numbers: 数字列表
    :return: 剩余数字的平均值
    """
    if not numbers or len(numbers) < 20:
        raise ValueError("列表中的数字个数必须至少为20。")

    # 对列表进行排序
    sorted_numbers = sorted(numbers)

    # 计算需要去除的数量
    remove_count = len(sorted_numbers) // 20  # 等价于总数的5%

    match method:
        case "5%":
            # 去除最大和最小的5%
            trimmed_numbers = sorted_numbers[remove_count:-remove_count]
        case "none":
            trimmed_numbers = sorted_numbers

    # 计算剩余数字的平均值
    average = sum(trimmed_numbers) / len(trimmed_numbers)

    return average

def data1():
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
    return parsed_data

def data2():

    parsed_data = data1()

    column = {}
    column[1] = [int(row[6]) for row in parsed_data]
    column[2] = [float(row[7]) for row in parsed_data]
    column[3] = [float(row[8]) for row in parsed_data]
    column[4] = [float(row[9]) for row in parsed_data]

    return column

data = data2()
mapping = [['T', 'T', 'T', 'T'],
           ['T', 'T', 'T', 'F'],
           ['T', 'T', 'F', 'T'],
           ['T', 'T', 'F', 'F'],
           ['T', 'F', 'T', 'T'],
           ['T', 'F', 'T', 'F'],
           ['T', 'F', 'F', 'T'],
           ['T', 'F', 'F', 'F']]

def split1(m, data):
    collection = {}
    for ty in range(0, 8, 1):
        for i in range(1, 5, 1):
            collection[f"for_Px{m[ty][0]}_Py{m[ty][1]}_Ox{m[ty][2]}_Oy{m[ty][3]}_{i}"] = []
            for line in range(ty,800,8):
                collection[f"for_Px{m[ty][0]}_Py{m[ty][1]}_Ox{m[ty][2]}_Oy{m[ty][3]}_{i}"].append(data[i][line])

    return collection

def write1(m, collections, method):
    with open("ana2.txt", "w") as file:
        for ty in range(0,8,1):
            for i in range(1,5,1):
                print(f"{i}", end="\t", file=file)
                for j in range(0,4,1):
                    print(f"{m[ty][j]}", end="\t", file=file)
                temp = process_and_average(collections[f"for_Px{m[ty][0]}_Py{m[ty][1]}_Ox{m[ty][2]}_Oy{m[ty][3]}_{i}"])
                print(temp, file=file)

coll = split1(mapping, data)
write1(mapping, coll, "5%")