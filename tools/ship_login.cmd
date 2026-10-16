@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "LOG=logs\ship_login.log"

echo === [1/4] Проверка окружения ===
where uv >nul 2>&1 || (echo ОШИБКА: uv не найден в PATH & goto :fail)
if not exist ".env" (echo ОШИБКА: нет файла .env & goto :fail)

findstr /R /C:"^COURSE_SHEET_URL=.\+" .env >nul
if errorlevel 1 (
    echo COURSE_SHEET_URL не задан в .env.
    set "IN_VAL="
    set /p "IN_VAL=Вставь ссылку на таблицу курса (Google Sheets): "
    set "IN_KEY=COURSE_SHEET_URL"
    set "IN_RE=^https://docs\.google\.com/spreadsheets/\S+$"
    call :putenv || (echo ОШИБКА: ссылка должна быть ссылкой на Google Sheets & goto :fail)
    echo COURSE_SHEET_URL записан в .env.
    echo COURSE_SHEET_URL записан в .env ^(значение не логируется^). >> "%LOG%"
)

findstr /R /C:"^SHEET_SLOT=.\+" .env >nul
if errorlevel 1 (
    echo SHEET_SLOT не задан в .env.
    set "IN_VAL="
    set /p "IN_VAL=Номер слота в таблице [15]: "
    if not defined IN_VAL set "IN_VAL=15"
    set "IN_KEY=SHEET_SLOT"
    set "IN_RE=^[0-9]+$"
    call :putenv || (echo ОШИБКА: SHEET_SLOT должен быть числом & goto :fail)
    echo SHEET_SLOT записан в .env.
    echo SHEET_SLOT записан в .env. >> "%LOG%"
)

set "CHROME="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME=1"
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME=1"
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "CHROME=1"
if not defined CHROME (echo ОШИБКА: системный Chrome не найден & goto :fail)

echo.
echo === [2/4] Установка playwright (extra ship) ===
uv sync --extra ship >> "%LOG%" 2>&1 || (echo ОШИБКА: uv sync не прошёл, см. %LOG% & goto :fail)

echo.
echo === [3/4] Вход в Google ===
echo Откроется Chrome. Войди в аккаунт, открой таблицу и ЗАКРОЙ окно.
powershell -NoProfile -Command "uv run adventship sheet-login 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE"
if errorlevel 1 (echo ОШИБКА: sheet-login завершился с ошибкой & goto :fail)

echo.
echo === [4/4] Готово ===
echo Профиль сохранён. Лог: %LOG%
pause
exit /b 0

:putenv
rem IN_KEY/IN_VAL/IN_RE come via the environment so the URL never touches a command line or the log.
powershell -NoProfile -Command "$v=([string]$env:IN_VAL).Trim(); if($v -notmatch $env:IN_RE){exit 1}; $k=$env:IN_KEY; $p=(Resolve-Path '.env').Path; $f=$false; $o=foreach($x in [IO.File]::ReadAllLines($p)){ if($x -match ('^'+$k+'=\s*$')){$f=$true; $k+'='+$v}else{$x} }; if(-not $f){$o=@($o)+($k+'='+$v)}; [IO.File]::WriteAllLines($p,[string[]]@($o),(New-Object Text.UTF8Encoding($false)))"
exit /b %errorlevel%

:fail
echo.
echo Остановлено. Лог: %LOG%
pause
exit /b 1
