@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem PRISM full-pipeline runner for Windows CMD.
rem Defaults are conservative: Stage 0 is disabled unless --run-stage0 or RUN_STAGE0=1.

cd /d "%~dp0"

if "%MODEL_NAME%"=="" set "MODEL_NAME=Qwen/Qwen2.5-1.5B" 
if "%STAGE0_MODELS%"=="" set "STAGE0_MODELS=%MODEL_NAME%"
if "%OUT_ROOT%"=="" set "OUT_ROOT=artifacts\prism"
if "%RUN_STAGE0%"=="" set "RUN_STAGE0=0"
if "%STAGE0_USE_SHARDS%"=="" set "STAGE0_USE_SHARDS=0"
if "%NUM_SHARDS%"=="" set "NUM_SHARDS=16"
if "%START_SHARD%"=="" set "START_SHARD=0"
if "%END_SHARD%"=="" set "END_SHARD=%NUM_SHARDS%"
if "%EPOCHS%"=="" set "EPOCHS=100"
if "%GROUP_SIZE%"=="" set "GROUP_SIZE=128"
if "%HIDDEN_SIZE%"=="" set "HIDDEN_SIZE=16"
if "%NUM_LAYERS%"=="" set "NUM_LAYERS=4"
if "%SEQ_LEN%"=="" set "SEQ_LEN=8"
if "%DEVICE%"=="" set "DEVICE=cuda"
if "%FAMILY%"=="" set "FAMILY=auto"
if "%BUDGET%"=="" set "BUDGET=3.0"
if "%BUDGETS%"=="" set "BUDGETS=2.5,2.75,3.0,3.25,3.5"
if "%RUN_STAGE1%"=="" set "RUN_STAGE1=1"
if "%RUN_STAGE2%"=="" set "RUN_STAGE2=1"
if "%RUN_QUIC%"=="" set "RUN_QUIC=1"
if "%RUN_RTN%"=="" set "RUN_RTN=1"
if "%RUN_STAGE4%"=="" set "RUN_STAGE4=1"
if "%EXECUTE%"=="" set "EXECUTE=0"
if "%TRUST_REMOTE_CODE%"=="" set "TRUST_REMOTE_CODE=0"
if "%PROMPT%"=="" set "PROMPT=Hello"
if "%MAX_NEW_TOKENS%"=="" set "MAX_NEW_TOKENS=16"
set "DRY_RUN=0"

:parse_args
if "%~1"=="" goto after_parse
if "%~1"=="--help" goto usage
if "%~1"=="-h" goto usage
if "%~1"=="--model" set "MODEL_NAME=%~2" & shift & shift & goto parse_args
if "%~1"=="--stage0-models" set "STAGE0_MODELS=%~2" & shift & shift & goto parse_args
if "%~1"=="--mlp-path" set "MLP_PATH=%~2" & shift & shift & goto parse_args
if "%~1"=="--out-root" set "OUT_ROOT=%~2" & shift & shift & goto parse_args
if "%~1"=="--run-stage0" set "RUN_STAGE0=1" & shift & goto parse_args
if "%~1"=="--stage0-use-shards" set "STAGE0_USE_SHARDS=1" & shift & goto parse_args
if "%~1"=="--num-shards" set "NUM_SHARDS=%~2" & shift & shift & goto parse_args
if "%~1"=="--start-shard" set "START_SHARD=%~2" & shift & shift & goto parse_args
if "%~1"=="--end-shard" set "END_SHARD=%~2" & shift & shift & goto parse_args
if "%~1"=="--epochs" set "EPOCHS=%~2" & shift & shift & goto parse_args
if "%~1"=="--group-size" set "GROUP_SIZE=%~2" & shift & shift & goto parse_args
if "%~1"=="--hidden-size" set "HIDDEN_SIZE=%~2" & shift & shift & goto parse_args
if "%~1"=="--num-layers" set "NUM_LAYERS=%~2" & shift & shift & goto parse_args
if "%~1"=="--seq-len" set "SEQ_LEN=%~2" & shift & shift & goto parse_args
if "%~1"=="--device" set "DEVICE=%~2" & shift & shift & goto parse_args
if "%~1"=="--family" set "FAMILY=%~2" & shift & shift & goto parse_args
if "%~1"=="--budget" set "BUDGET=%~2" & shift & shift & goto parse_args
if "%~1"=="--budgets" shift & set "BUDGETS=" & goto parse_budgets
if "%~1"=="--skip-stage1" set "RUN_STAGE1=0" & shift & goto parse_args
if "%~1"=="--skip-stage2" set "RUN_STAGE2=0" & shift & goto parse_args
if "%~1"=="--skip-quic" set "RUN_QUIC=0" & shift & goto parse_args
if "%~1"=="--skip-rtn" set "RUN_RTN=0" & shift & goto parse_args
if "%~1"=="--skip-run" set "RUN_STAGE4=0" & shift & goto parse_args
if "%~1"=="--execute" set "EXECUTE=1" & shift & goto parse_args
if "%~1"=="--trust-remote-code" set "TRUST_REMOTE_CODE=1" & shift & goto parse_args
if "%~1"=="--prompt" set "PROMPT=%~2" & shift & shift & goto parse_args
if "%~1"=="--max-new-tokens" set "MAX_NEW_TOKENS=%~2" & shift & shift & goto parse_args
if "%~1"=="--dry-run" set "DRY_RUN=1" & shift & goto parse_args
echo Unknown option: %~1
goto usage

:parse_budgets
if "%~1"=="" goto parse_args
set "NEXT_ARG=%~1"
if "!NEXT_ARG:~0,2!"=="--" goto parse_args
if defined BUDGETS (set "BUDGETS=!BUDGETS!,%~1") else set "BUDGETS=%~1"
shift
goto parse_budgets

:after_parse
set "SANITIZED_MODEL=%MODEL_NAME%"
set "SANITIZED_MODEL=%SANITIZED_MODEL:/=_%"
set "SANITIZED_MODEL=%SANITIZED_MODEL::=_%"
set "SANITIZED_MODEL=%SANITIZED_MODEL:\=_%"
set "SANITIZED_MODEL=%SANITIZED_MODEL: =_%"

if "%STAGE0_DIR%"=="" set "STAGE0_DIR=%OUT_ROOT%\%SANITIZED_MODEL%_stage0"
if "%RUN_DIR%"=="" set "RUN_DIR=%OUT_ROOT%\%SANITIZED_MODEL%_full"
if "%MLP_PATH%"=="" set "MLP_PATH=%STAGE0_DIR%\prism_mlp.pt"

set "PROFILE_PATH=%RUN_DIR%\profile.json"
set "ASSIGNMENT_PATH=%RUN_DIR%\assignment_%BUDGET%.json"
set "QUIC_PATH=%RUN_DIR%\quic_assignment_%BUDGET%.json"
set "RTN_DIR=%RUN_DIR%\rtn"
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TIMESTAMP=%%T"
set "LOG_DIR=%OUT_ROOT%\logs"
set "LOG_PATH=%LOG_DIR%\%RUN_TIMESTAMP%_%SANITIZED_MODEL%_full_pipeline.txt"
set "TRUST_FLAG="
if "%TRUST_REMOTE_CODE%"=="1" set "TRUST_FLAG=--trust-remote-code"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

call :log PRISM full pipeline
call :log Repo: %CD%
call :log Target model: %MODEL_NAME%
call :log Run dir: %RUN_DIR%
call :log Log file: %LOG_PATH%
if "%DRY_RUN%"=="1" call :log Mode: dry-run ^(commands are printed, not executed^)

if not "%DRY_RUN%"=="1" if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

if "%RUN_STAGE0%"=="1" (
  call :stage_start 0 "Build sensitivity dataset and train meta-learner"
  if "%STAGE0_USE_SHARDS%"=="1" (
    call :log Stage 0 shard mode: collecting shards %START_SHARD%..%END_SHARD% of %NUM_SHARDS%
    call :run python ..\scripts\stage0_sharded.py --models "%STAGE0_MODELS%" --output-dir "%STAGE0_DIR%" --num-shards "%NUM_SHARDS%" --start-shard "%START_SHARD%" --end-shard "%END_SHARD%" --group-size "%GROUP_SIZE%" --epochs "%EPOCHS%"
    if errorlevel 1 exit /b 1
    call :log Stage 0 shard merge/train command completed
  ) else (
    set "STAGE0_MODEL_ARGS="
    for %%M in ("%STAGE0_MODELS:,=" "%") do set "STAGE0_MODEL_ARGS=!STAGE0_MODEL_ARGS! %%~M"
    call :log Stage 0 single-job mode: collecting sensitivity data for %STAGE0_MODELS%
    call :run python -m prism.cli.train_meta --output-dir "%STAGE0_DIR%" --epochs "%EPOCHS%" --group-size "%GROUP_SIZE%" --model-names !STAGE0_MODEL_ARGS!
    if errorlevel 1 exit /b 1
  )
  set "MLP_PATH=%STAGE0_DIR%\prism_mlp.pt"
  call :stage_done 0 "Meta-learner ready at %MLP_PATH%"
) else (
  call :stage_skip 0 "Using existing MLP at %MLP_PATH%"
)

if "%RUN_STAGE1%"=="1" (
  call :stage_start 1 "Profile model layers and predict sensitivity"
  call :run python -m prism.cli.profile --model-id-or-path "%MODEL_NAME%" --family "%FAMILY%" --mlp-path "%MLP_PATH%" --group-size "%GROUP_SIZE%" --hidden-size "%HIDDEN_SIZE%" --num-layers "%NUM_LAYERS%" --device "%DEVICE%" --output-path "%PROFILE_PATH%" %TRUST_FLAG%
  if errorlevel 1 exit /b 1
  call :stage_done 1 "Profile written to %PROFILE_PATH%"
) else (
  call :stage_skip 1 "Reusing profile at %PROFILE_PATH%"
)

if "%RUN_STAGE2%"=="1" (
  call :stage_start 2 "Solve mixed-bit assignments"
  call :log Stage 2 main budget: %BUDGET%
  call :run python -m prism.cli.assign --profile-path "%PROFILE_PATH%" --budget "%BUDGET%" --output-path "%ASSIGNMENT_PATH%"
  if errorlevel 1 exit /b 1

  for %%B in ("%BUDGETS:,=" "%") do (
    if not "%%~B"=="" if not "%%~B"=="%BUDGET%" (
      call :log Stage 2 sweep budget: %%~B
      call :run python -m prism.cli.assign --profile-path "%PROFILE_PATH%" --budget "%%~B" --output-path "%RUN_DIR%\assignment_%%~B.json"
      if errorlevel 1 exit /b 1
    )
  )
  call :stage_done 2 "Assignments written under %RUN_DIR%"
) else (
  call :stage_skip 2 "Reusing assignment at %ASSIGNMENT_PATH%"
)

if "%RUN_QUIC%"=="1" (
  call :stage_start 2.5 "Run synthetic-only QUIC correction"
  call :run python -m prism.cli.quic --model-id-or-path "%MODEL_NAME%" --family "%FAMILY%" --device "%DEVICE%" --profile-path "%PROFILE_PATH%" --assignment-path "%ASSIGNMENT_PATH%" --output-path "%QUIC_PATH%" --hidden-size "%HIDDEN_SIZE%" --seq-len "%SEQ_LEN%" %TRUST_FLAG%
  if errorlevel 1 exit /b 1
  set "FINAL_ASSIGNMENT=%QUIC_PATH%"
  call :stage_done 2.5 "QUIC assignment written to %QUIC_PATH%"
) else (
  set "FINAL_ASSIGNMENT=%ASSIGNMENT_PATH%"
  call :stage_skip 2.5 "Using assignment without QUIC"
)

if "%RUN_RTN%"=="1" (
  call :stage_start 3 "Precompute RTN artifacts"
  call :run python -m prism.cli.precompute_rtn --model-id-or-path "%MODEL_NAME%" --family "%FAMILY%" --device "%DEVICE%" --group-size "%GROUP_SIZE%" --hidden-size "%HIDDEN_SIZE%" --num-layers "%NUM_LAYERS%" --output-dir "%RTN_DIR%" %TRUST_FLAG%
  if errorlevel 1 exit /b 1
  call :stage_done 3 "RTN artifacts written to %RTN_DIR%"
) else (
  call :stage_skip 3 "Reusing RTN artifacts at %RTN_DIR%"
)

if "%RUN_STAGE4%"=="1" (
  call :stage_start 4 "Assemble runtime model"
  set "EXECUTE_FLAG="
  if "%EXECUTE%"=="1" set "EXECUTE_FLAG=--execute"
  call :run python -m prism.cli.run --model-id-or-path "%MODEL_NAME%" --family "%FAMILY%" --device "%DEVICE%" --artifact-root "%RTN_DIR%" --assignment-path "%FINAL_ASSIGNMENT%" --hidden-size "%HIDDEN_SIZE%" --num-layers "%NUM_LAYERS%" --prompt "%PROMPT%" --max-new-tokens "%MAX_NEW_TOKENS%" %EXECUTE_FLAG% %TRUST_FLAG%
  if errorlevel 1 exit /b 1
  call :stage_done 4 "Runtime assembly completed"
) else (
  call :stage_skip 4 "Runtime assembly skipped"
)

call :blank
call :log Done.
call :log Profile: %PROFILE_PATH%
call :log Assignment: %ASSIGNMENT_PATH%
call :log Final assignment: %FINAL_ASSIGNMENT%
call :log RTN artifacts path %RTN_DIR%
call :log Log file: %LOG_PATH%
exit /b 0

:run
call :blank
call :log + %*
if "%DRY_RUN%"=="1" exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & %* 2>&1 | Tee-Object -FilePath '%LOG_PATH%' -Append; exit $LASTEXITCODE }"
exit /b %ERRORLEVEL%

:stage_start
call :blank
call :log ==== Stage %~1 START: %~2 ====
exit /b 0

:stage_done
call :log ==== Stage %~1 DONE: %~2 ====
exit /b 0

:stage_skip
call :log ==== Stage %~1 SKIP: %~2 ====
exit /b 0

:blank
echo.
>>"%LOG_PATH%" echo.
exit /b 0

:log
set "LOG_LINE=[%DATE% %TIME%] %*"
echo(!LOG_LINE!
>>"%LOG_PATH%" echo(!LOG_LINE!
exit /b 0

:usage
echo Usage:
echo   run_full_pipeline.bat [options]
echo.
echo Examples:
echo   run_full_pipeline.bat --model Qwen/Qwen2.5-1.5B --mlp-path artifacts\prism\Qwen_Qwen2.5-1.5B_merged\prism_mlp.pt
echo   run_full_pipeline.bat --run-stage0 --stage0-models "facebook/opt-125m,EleutherAI/pythia-160m" --model Qwen/Qwen2.5-1.5B
echo   run_full_pipeline.bat --run-stage0 --stage0-use-shards --stage0-models "Qwen/Qwen2.5-1.5B" --num-shards 16 --model Qwen/Qwen2.5-1.5B
echo.
echo Main options:
echo   --model NAME, --stage0-models LIST, --mlp-path PATH, --out-root DIR
echo   --run-stage0, --stage0-use-shards, --num-shards N, --start-shard N, --end-shard N
echo   --epochs N, --group-size N, --device DEVICE, --family NAME
echo   --hidden-size N, --num-layers N, --seq-len N
echo   --budget B, --budgets LIST
echo   --skip-stage1, --skip-stage2, --skip-quic, --skip-rtn, --skip-run
echo   --execute, --trust-remote-code, --prompt TEXT, --max-new-tokens N, --dry-run
exit /b 2
