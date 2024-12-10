@echo off
setlocal enabledelayedexpansion

:: 固定参数
set filename=v1.4.1_2_orth_proj
set EPOCHS=200

:: 循环从 101 到 EPOCHS
for /L %%i in (101, 1, %EPOCHS%) do (
    :: 将循环变量赋值给 num
    set num=%%i

    :: 打印当前状态
    echo !num! -> %EPOCHS%

    :: 调用 Python 脚本，执行不同的参数组合
    call :run_python !num! T T T T
    call :run_python !num! T T T F
    call :run_python !num! T T F T
    call :run_python !num! T T F F
    call :run_python !num! T F T T
    call :run_python !num! T F T F
    call :run_python !num! T F F T
    call :run_python !num! T F F F
)

:: 子过程调用 Python 脚本
:run_python
:: 参数说明
:: %1 = seed_num
:: %2 = orth_x
:: %3 = orth_y
:: %4 = proj_x
:: %5 = proj_y
python MVHSC.py --orth_x %2 --orth_y %3 --proj_x %4 --proj_y %5 --file_name %filename% --seed_num %1 --hypergrad_method "forward"
exit /b
