from MVHSC_Aux import create_instances, parser

def main():
    S = parser()
    IT = create_instances(S)
    IT.run()

if __name__ == "__main__":
    main()