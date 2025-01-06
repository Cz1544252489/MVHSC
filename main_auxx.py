# plot and cluster
# avoiding too long aux file
import json
import os
import sys
from datetime import datetime

import pandas as pd

# prefix = "ADM_orth"
# fields = ["opt_method", "hypergrad_method", "rloop0", "time_cost", "last_UL_dval", "last_LL_dval",
#           "orth_x", "orth_y"]
def filte_data(log_rootpath, prefix, fields):
    path = os.path.join(".",log_rootpath)
    ends = ".json"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = prefix+timestamp+".txt"
    output_path = os.path.join(path, output_filename)

    json_files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(prefix) and f.endswith(ends)]

    data = []

    for file in json_files:
        with open(file, 'r') as f:
            try:
                content = json.load(f)
                row = {}
                for field in fields:
                    row[field] = content.get(field, None)
                data.append(row)

            except json.JSONDecodeError:
                print(f"Error decoding {field}")

    df = pd.DataFrame(data)
    result = df.groupby(['hypergrad_method','lr','mu','loop1','orth_x', 'orth_y'])[['rloop0','time_cost','last_UL_dval','last_LL_dval']].mean()
    result.to_csv(output_path, sep='\t')
    # df.to_excel(output_path, index=False)
    print(f"Data has been calculated and saved to {output_path}")
    # return df


if __name__ == "__main__":
    log_rootpath = sys.argv[1]
    prefix = sys.argv[2]
    fields = sys.argv[3:]
    # prefix = "ADM_orth"
    # fields = ["opt_method", "hypergrad_method", "rloop0", "time_cost", "last_UL_dval", "last_LL_dval",
    #           "orth_x", "orth_y"]
    filte_data(log_rootpath, prefix, fields)

    print("aa")
