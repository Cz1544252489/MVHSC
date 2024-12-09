from MVHSC_Aux import create_instances, parser
import torch

def main():
    S = parser()
    IT = create_instances(S)
    IT.run()

    return IT

if __name__ == "__main__":
    IT = main()
    with open("output2.txt","a") as file:
        print(f"{IT.result['hypergrad_method']}", end="\t", file=file)
        print(f"{IT.result['seed_num']}", end="\t", file=file)
        print(f"{IT.result['comment']}", end="\t", file=file)
        print(f"{IT.result['best_ul_acc'][0]}", end="\t", file=file)
        print(f"{IT.result['best_ul_acc'][1]}", end="\t", file=file)
        print(f"{IT.result['time_elapsed'][IT.result['best_ul_acc'][0]]}", end="\t", file=file)
        print(f"{IT.result['time_elapsed'][0]}", end="\n", file=file)
