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
  filename="v1.3.5_1_orth"
  python MVHSC.py --orth_x True --orth_y True --file_name $filename \
  --result_output "save" --figure_name "TT" --plot_content "nmi" "ari"
  python MVHSC.py --orth_x False --orth_y True --file_name $filename \
  --result_output "save" --figure_name "FT" --plot_content "nmi" "ari"
  python MVHSC.py --orth_x True --orth_y False --file_name $filename \
  --result_output "save" --figure_name "TF" --plot_content "nmi" "ari"
  python MVHSC.py --orth_x False --orth_y False --file_name $filename \
  --result_output "save" --figure_name "FF" --plot_content "nmi" "ari"
}

# 测试是否更新$lambda_r$对结果的影响
test_lambda_r(){
  filename="v1.3.5_2_lambda_r"
  python MVHSC.py --update_lambda_r True --file_name $filename --seed_num 42\
  --result_output "save" --figure_name "T" --plot_content "nmi" "acc"
  python MVHSC.py --update_lambda_r False --file_name $filename --seed_num 42\
  --result_output "save" --figure_name "F" --plot_content "nmi" "acc"
}
# 根据(v1.3.5)测试结果发现，更新时nmi结果会更好一点，而不更新时acc,ari,f1的结果会更好一点；

# 测试使用不同的比例系数mu对结果的影响
test_mu_2(){
for mu in $(seq 0 0.1 1); do
  echo ${mu}
  python MVHSC.py -E 300 --mu ${mu} --file_name "test_mu_2"
done
}
show_data(){
  filename="v1.3.5-1-mu"
  python test.py --prefix "v1.3.5-1-mu" --target_keys "mu" "best_ul_ari" --key_x "mu" --key_y "best_ul_ari"
}
#
test_and_show_mu(){
  filename="v1.3.5-1-mu"
  for mu in $(seq 0 0.1 1); do
    echo $mu
    python MVHSC.py --mu ${mu} --file_name $filename
  done
  python test.py --prefix $filename --target_keys "mu" "best_ul_acc" --key_x "mu" --key_y "best_ul_acc"
}
# 在更新lambda_r时聚合梯度中间部分有更好的f1结果，不更新时两侧有更好的结果

test_and_show_lambda_x2(){
  filename="v1.3.5-2-lambda_x"
  for x in $(seq 0 0.2 4); do
    echo $x
    python MVHSC.py --lambda_x $x --file_name $filename
  done
  python test.py --prefix $filename --target_keys "lambda_x" "best_ul_acc" --key_x "lambda_x" --key_y "best_ul_acc"
}
# lambda_x 取1附近有较高的上限

test_and_show_lambda_x3(){
  filename="v1.3.5-3-lambda_x"
  for x in $(seq 0 0.2 4); do
    echo $x
    python MVHSC.py --lambda_x $x --file_name $filename --cluster_method "spectral"
  done
  python test.py --prefix $filename --target_keys "lambda_x" "best_ul_acc" --key_x "lambda_x" --key_y "best_ul_acc"
}
# 此处使用 参数谱方法没有意义

test_and_show_max_ll_epochs(){
  filename="v1.3.5-3-max_ll_epochs"
  for x in $(seq 1 2 20); do
    echo $x
    python MVHSC.py --max_ll_epochs $x --file_name $filename
  done
  python test.py --prefix $filename --target_keys "max_ll_epochs" "best_ul_acc" --key_x "max_ll_epochs" --key_y "best_ul_acc"
}
# max_ll_epochs为1的时候就足够了

test_and_show_beta(){
  filename="v1.3.5-1-beta"
  x_plot="clip_free_beta"
  y_plot="best_ul_acc"
  for x in $(seq 0 0.1 1); do
    echo $x
    python MVHSC.py --max_ll_epochs 10 --clip_method_alpha "com" --clip_method_beta "free" --clip_free_beta $x \
    --file_name $filename
  done
  python test.py --prefix $filename --target_keys $x_plot $y_plot --key_x $x_plot --key_y $y_plot
}
# 当取beta为0.5时有最好的结果，最小的迭代次数和最大的准确率

test_and_show_beta2(){
  filename="v1.3.5-2-beta"
  x_plot="clip_free_beta"
  y_plot="best_ul_acc"
  for x in $(seq 0 0.1 1); do
    echo $x
    python MVHSC.py --max_ll_epochs 10 --clip_method_alpha "gaussian" --clip_method_beta "free" --clip_free_beta $x \
    --file_name $filename
  done
  python test.py --prefix $filename --target_keys $x_plot $y_plot --key_x $x_plot --key_y $y_plot
}
# 取0时有最优准确率，取0.6是有最小迭代次数

test_and_show_beta3(){
  filename="v1.3.5-3-beta"
  x_plot="clip_free_beta"
  y_plot="best_ul_acc"
  for x in $(seq 0 0.1 1); do
    echo $x
    python MVHSC.py --max_ll_epochs 10 --clip_method_alpha "sqrt" --clip_method_beta "free" --clip_free_beta $x \
    --file_name $filename
  done
  python test.py --prefix $filename --target_keys $x_plot $y_plot --key_x $x_plot --key_y $y_plot
}
# 取0时有最优准确率，取0.5有最小迭代次数，综合最优为0.6

test_and_show_beta4(){
  filename="v1.3.5-5-beta"
  x_plot="clip_free_beta"
  y_plot="best_ul_acc"
  for x in $(seq 0 0.1 1); do
    echo $x
    python MVHSC.py --max_ll_epochs 1 --clip_method_alpha "gaussian" --clip_method_beta "free" --clip_free_beta $x \
    --file_name $filename
  done
  python test.py --prefix $filename --target_keys $x_plot $y_plot --key_x $x_plot --key_y $y_plot
}

test_orth_xy1(){
  filename="v1.3.5_orth_xy1"
  python MVHSC.py --orth_x True --orth_y True --file_name $filename \
  --result_output "save" --figure_name "v1.3.5_TT" --plot_content "acc" "f1"
  python MVHSC.py --orth_x False --orth_y True --file_name $filename \
  --result_output "save" --figure_name "v1.3.5_FT" --plot_content "acc" "f1"
  python MVHSC.py --orth_x True --orth_y False --file_name $filename \
  --result_output "save" --figure_name "v1.3.5_TF" --plot_content "acc" "f1"
  # python MVHSC.py --orth_x False --orth_y False --file_name $filename \
  # --result_output "save" --figure_name "v1.3.5_FF" --plot_content "acc" "f1"
}
test_orth_xy1
# FF的情况无法收敛，会报错。 TT的情况效果最好。





