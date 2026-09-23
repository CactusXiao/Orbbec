@echo off
setlocal

set DATASET_ROOT=%~1
if "%DATASET_ROOT%"=="" set DATASET_ROOT=/home/ubuntu/orbbec/src/sync/test/test/zyc

python -m src.main --dataset_root "%DATASET_ROOT%" --config configs/default.yaml --log_level INFO
