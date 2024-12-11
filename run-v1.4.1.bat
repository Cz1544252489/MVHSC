@echo off
setlocal enabledelayedexpansion

set filename=v1.4.1_4_orth_proj
set EPOCHS=100

:: 循环从 1 到 EPOCHS
for /L %%i in (1, 1, %EPOCHS%) do (
    :: 生成随机种子
    set /A seed_num=!RANDOM!

    :: 打印当前状态
    echo Epoch: %%i of %EPOCHS% Seed: !seed_num!

    :: 调用子过程，传递参数
    call :run_python %%i !seed_num! T T T T
    call :run_python %%i !seed_num! T T T F
    call :run_python %%i !seed_num! T T F T
    call :run_python %%i !seed_num! T T F F
    call :run_python %%i !seed_num! T F T T
    call :run_python %%i !seed_num! T F T F
    call :run_python %%i !seed_num! T F F T
    call :run_python %%i !seed_num! T F F F
)

:: 子过程调用 Python 脚本
:run_python
:: 参数说明
:: %1 = epoch_num
:: %2 = seed_num
:: %3 = proj_x
:: %4 = proj_y
:: %5 = orth_x
:: %6 = orth_y

:: 打印传入参数（调试用）
echo Arguments: epoch_num=%1 seed_num=%2 proj_x=%3 proj_y=%4 orth_x=%5 orth_y=%6

:: 调用 Python 脚本
python MVHSC.py --proj_x %3 --proj_y %4 --orth_x %5 --orth_y %6 --file_name %filename% --seed_num %2 --hypergrad_method "forward"
exit /b