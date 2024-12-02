from MVHSC_Aux import create_instances, parser
import torch

def main():
    S = parser()
    IT = create_instances(S)
    IT.run()

    return IT

if __name__ == "__main__":
    IT = main()
    print(IT.result["best_ul_acc"])
    print([IT.ul_val])
    print([torch.linalg.norm(IT.grad_x, ord=2)])
    print([torch.linalg.norm(IT.grad_y, ord=2)])
    print("aa")