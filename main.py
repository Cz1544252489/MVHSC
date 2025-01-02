from main_aux import data_importation, iteration
DI = data_importation()
IT = iteration(DI)

# IT.run_as_adm()
IT.run_as_bda_backward()

print("aa")