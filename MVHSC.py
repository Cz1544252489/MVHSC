from MVHSC_Aux import create_instances, parser
import torch

def main():
    S = parser()
    IT = create_instances(S)
    IT.run()

    return IT

if __name__ == "__main__":
    IT = main()
