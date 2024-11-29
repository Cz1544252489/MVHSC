import matplotlib.pyplot as plt
from MVHSC_Aux import create_instances, parser

S = parser()
IT = create_instances(S)
# 调用函数
results = IT.EV.process_json_files_multi_keys()
plot_num = 3
get_data_num = 10


def run(IT, results, plot_num:int, get_data_num:int=10):
    match plot_num:
        case 2:
            x = []
            y = []
            for i in range(get_data_num):
                x.append(results[i][1][IT.S["key_x"]])
                y.append(results[i][1][IT.S["key_y"]][1])

            # Sorting both lists based on x values
            sorted_pairs = sorted(zip(x, y))
            x_sorted, y_sorted = zip(*sorted_pairs)
            plt.plot(x_sorted, y_sorted, marker='o', linestyle='-', label=f"{IT.S['key_x']}-{IT.S['key_y']}")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Line Plot of Sorted Data')
            plt.grid(True)
            plt.legend()
            plt.legend()
            plt.show()

        case 3:
            x = []
            y = []
            z = []
            for i in range(get_data_num):
                x.append(results[i][1][IT.S["key_x"]])
                y.append(results[i][1][IT.S["key_y"]][1])
                z.append(results[i][1][IT.S["key_y"]][0])

            # Sorting both lists based on x values
            sorted_pairs = sorted(zip(x, y, z))
            x_sorted, y_sorted, z_sorted = zip(*sorted_pairs)

            # 创建图形和第一个 y 轴
            fig, ax1 = plt.subplots()

            # 绘制 x-y 折线图
            ax1.plot(x_sorted, y_sorted, marker='o', linestyle='-', color='b', label='x vs y')
            ax1.set_xlabel(f"{IT.S['key_x']}")
            ax1.set_ylabel(f"{IT.S['key_y']}", color='b')  # 设置y轴标签并指定颜色
            ax1.tick_params(axis='y', labelcolor='b')  # 设置y轴的刻度颜色

            # 创建第二个 y 轴
            ax2 = ax1.twinx()

            # 绘制 x-z 折线图
            ax2.plot(x_sorted, z_sorted, marker='s', linestyle='--', color='r', label='x vs z')
            ax2.set_ylabel("Iter. of getting best", color='r')  # 设置z轴标签并指定颜色
            ax2.tick_params(axis='y', labelcolor='r')  # 设置y轴的刻度颜色

            # 添加图例
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # 添加标题
            plt.title("x-y and x-z Line Plots with Different Y Axes")

            # 显示网格
            ax1.grid(True)

            # 显示图形
            plt.show()

run(IT, results, plot_num, get_data_num)
