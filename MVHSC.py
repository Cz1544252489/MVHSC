from typing import Literal

from MVHSC_Aux import create_instances
import argparse

def main():
    parser = argparse.ArgumentParser(description="None")

    parser.add_argument('--file_name', type=str, default="result.json")
    parser.add_argument('--view2', type=Literal[0,2,4], default=2)
    parser.add_argument('--seed_num', type=int, default=42)
    parser.add_argument('-L','--max_ll_epochs', type= int, default=10, help="下层优化函数内部迭代次数")
    parser.add_argument('-U','--max_ul_epochs', type=int, default=1, help="上层优化函数内部迭代次数")
    parser.add_argument('-E','--Epochs', type=int, default=10, help="总迭代次数")
    parser.add_argument('--s_u', type=float, default=0.5)
    parser.add_argument('--s_l', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lambda_r', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--update_lambda_r', type=bool, default=False)
    parser.add_argument('--result_output', type=str, default="show")


    S0 = parser.parse_args()
    S = vars(S0)

    DI, IN, CL, EV, IT = create_instances(S)

    IT.run()

if __name__ == "__main__":
    main()