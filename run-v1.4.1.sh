test0(){
python MVHSC.py --max_ll_epochs 5 --hypergrad_method "backward" -E 300 --result_output "show" --plot_content "acc"
}

# 测试是否使用正交和投影对结果的影响
test_orth_proj_1(){
  filename="v1.4.1_1_orth_proj"
  ##################  TT
  python MVHSC.py --orth_x T --orth_y T --proj_x T --proj_y T --file_name $filename \
  --result_output "save" --figure_name "TTTT" --plot_content "acc" "grad"
  python MVHSC.py --orth_x T --orth_y T --proj_x T --proj_y F --file_name $filename \
  --result_output "save" --figure_name "TTTF" --plot_content "acc" "grad"
  python MVHSC.py --orth_x T --orth_y T --proj_x F --proj_y T --file_name $filename \
  --result_output "save" --figure_name "TTFT" --plot_content "acc" "grad"
  python MVHSC.py --orth_x T --orth_y T --proj_x F --proj_y F --file_name $filename \
  --result_output "save" --figure_name "TTFF" --plot_content "acc" "grad"
  ##################  TF
  python MVHSC.py --orth_x T --orth_y F --proj_x T --proj_y T --file_name $filename \
  --result_output "save" --figure_name "TFTT" --plot_content "acc" "grad"
  python MVHSC.py --orth_x T --orth_y F --proj_x T --proj_y F --file_name $filename \
  --result_output "save" --figure_name "TFTF" --plot_content "acc" "grad"
  python MVHSC.py --orth_x T --orth_y F --proj_x F --proj_y T --file_name $filename \
  --result_output "save" --figure_name "TFFT" --plot_content "acc" "grad"
  python MVHSC.py --orth_x T --orth_y F --proj_x F --proj_y F --file_name $filename \
  --result_output "save" --figure_name "TFFF" --plot_content "acc" "grad"
  ##################  FT
#   python MVHSC.py --orth_x F --orth_y T --proj_x T --proj_y T --file_name $filename \
#   --result_output "save" --figure_name "FTTT" --plot_content "acc" "grad"
#   python MVHSC.py --orth_x F --orth_y T --proj_x T --proj_y F --file_name $filename \
#   --result_output "save" --figure_name "FTTF" --plot_content "acc" "grad"
#   python MVHSC.py --orth_x F --orth_y T --proj_x F --proj_y T --file_name $filename \
#   --result_output "save" --figure_name "FTFT" --plot_content "acc" "grad"
#   python MVHSC.py --orth_x F --orth_y T --proj_x F --proj_y F --file_name $filename \
#   --result_output "save" --figure_name "FTFF" --plot_content "acc" "grad"
  ##################  FF
#   python MVHSC.py --orth_x F --orth_y F --proj_x T --proj_y T --file_name $filename \
#   --result_output "save" --figure_name "FFTT" --plot_content "acc" "grad"
#   python MVHSC.py --orth_x F --orth_y F --proj_x T --proj_y F --file_name $filename \
#   --result_output "save" --figure_name "FFTF" --plot_content "acc" "grad"
#   python MVHSC.py --orth_x F --orth_y F --proj_x F --proj_y T --file_name $filename \
#   --result_output "save" --figure_name "FFFT" --plot_content "acc" "grad"
#   python MVHSC.py --orth_x F --orth_y F --proj_x F --proj_y F --file_name $filename \
#   --result_output "save" --figure_name "FFFF" --plot_content "acc" "grad"
  #######################
}

test_orth_proj_2(){
  filename="v1.4.1_2_orth_proj"
  for num in $(seq 1 1 100); do
    echo "$num"
    ##################  TT
    python MVHSC.py --orth_x T --orth_y T --proj_x T --proj_y T --file_name $filename --comment "TTTT" --seed_num "$num" --hypergrad_method "forward"
    python MVHSC.py --orth_x T --orth_y T --proj_x T --proj_y F --file_name $filename --comment "TTTF" --seed_num "$num" --hypergrad_method "forward"
    python MVHSC.py --orth_x T --orth_y T --proj_x F --proj_y T --file_name $filename --comment "TTFT" --seed_num "$num" --hypergrad_method "forward"
    python MVHSC.py --orth_x T --orth_y T --proj_x F --proj_y F --file_name $filename --comment "TTFF" --seed_num "$num" --hypergrad_method "forward"
    ##################  TF
    python MVHSC.py --orth_x T --orth_y F --proj_x T --proj_y T --file_name $filename --comment "TFTT" --seed_num "$num" --hypergrad_method "forward"
    python MVHSC.py --orth_x T --orth_y F --proj_x T --proj_y F --file_name $filename --comment "TFTF" --seed_num "$num" --hypergrad_method "forward"
    python MVHSC.py --orth_x T --orth_y F --proj_x F --proj_y T --file_name $filename --comment "TFFT" --seed_num "$num" --hypergrad_method "forward"
    python MVHSC.py --orth_x T --orth_y F --proj_x F --proj_y F --file_name $filename --comment "TFFF" --seed_num "$num" --hypergrad_method "forward"
  done
}
test_orth_proj_2
