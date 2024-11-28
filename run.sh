#!/bin/zsh

# 测试使用不同的比例系数mu对结果的影响
testmu(){
for mu in $(seq 0 0.1 1); do
  echo ${mu}
  python MVHSC.py -E 300 --mu ${mu} --file_name "test4mu_gpu"
done
}

# 测试是否使用正交对结果的影响
test_orth(){
  python MVHSC.py --orth_x True --orth_y True --file_name "test_orth" \
  --result_output "save" --figure_name "TT" --plot_content "f1" "acc"
  python MVHSC.py --orth_x False --orth_y True --file_name "test_orth" \
  --result_output "save" --figure_name "FT" --plot_content "f1" "acc"
  python MVHSC.py --orth_x True --orth_y False --file_name "test_orth" \
  --result_output "save" --figure_name "TF" --plot_content "f1" "acc"
  python MVHSC.py --orth_x False --orth_y False --file_name "test_orth" \
  --result_output "save" --figure_name "FF" --plot_content "f1" "acc"
}

# 测试是否更新$lambda_r$对结果的影响
test_lambda_r(){
  python MVHSC.py --update_lambda_r True --file_name "test_lambda_r" \
  --result_output "save" --figure_name "T" --plot_content "nmi" "acc"
  python MVHSC.py --update_lambda_r False --file_name "test_lambda_r" \
  --result_output "save" --figure_name "F" --plot_content "nmi" "acc"
}

test_lambda_r