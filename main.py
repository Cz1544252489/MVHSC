from main_aux import create_instance, parser
import torch
def main():
    S = parser()
    ITR = create_instance(S)
    ITR.run()

    return ITR

if __name__ == "__main__":
    IT = main()

    # print("aa")