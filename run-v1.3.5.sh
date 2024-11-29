#!/bin/zsh

# 测试使用不同的比例系数mu对结果的影响
test_mu_1(){
for mu in $(seq 0 0.1 1); do
  echo ${mu}
  python MVHSC.py -E 300 --mu ${mu} --file_name "test4mu_gpu"
done
}

# 测试是否使用正交对结果的影响
test_orth_1(){
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
  filename="V1.3.5_2_lambda_r"
  python MVHSC.py --update_lambda_r True --file_name $filename --seed_num 42\
  --result_output "save" --figure_name "T" --plot_content "nmi" "acc"
  python MVHSC.py --update_lambda_r False --file_name $filename --seed_num 42\
  --result_output "save" --figure_name "F" --plot_content "nmi" "acc"
}
# 根据(v1.3.5)测试结果发现，更新时nmi结果会更好一点，而不更新时acc,ari,f1的结果会更好一点；
test_lambda_r

# 测试使用不同的比例系数mu对结果的影响
test_mu_2(){
for mu in $(seq 0 0.1 1); do
  echo ${mu}
  python MVHSC.py -E 300 --mu ${mu} --file_name "test_mu_2"
done
}
show_data(){
  python test.py --prefix "test_mu_2" --target_keys "mu" "best_ul_f1" --key_x "mu" --key_y "best_ul_f1"
}

#
test_and_show_mu(){
  filename="V1.3.5-mu_F4"
  for mu in $(seq 0 0.1 1); do
    echo $mu
    python MVHSC.py -E 300 --mu ${mu} --update_lambda_r False --file_name $filename --seed_num 42
  done
  python test.py --prefix $filename --target_keys "mu" "best_ul_f1" --key_x "mu" --key_y "best_ul_f1"
}
# 在更新lambda_r时聚合梯度中间部分有更好的f1结果，不更新时两侧有更好的结果