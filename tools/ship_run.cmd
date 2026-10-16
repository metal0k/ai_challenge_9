@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set "LOG=logs\ship_run.log"

echo === [1/5] Проверка окружения ===
where uv >nul 2>&1 || (echo ОШИБКА: uv не найден в PATH & goto :fail)
where git >nul 2>&1 || (echo ОШИБКА: git не найден в PATH & goto :fail)
if not exist ".env" (echo ОШИБКА: нет файла .env & goto :fail)
findstr /R /C:"^COURSE_SHEET_URL=.\+" .env >nul
if errorlevel 1 (echo ОШИБКА: COURSE_SHEET_URL не задан в .env, запусти tools\ship_login.cmd & goto :fail)

findstr /R /C:"^SHEET_SLOT=.\+" .env >nul
if errorlevel 1 (echo ОШИБКА: SHEET_SLOT не задан в .env, запусти tools\ship_login.cmd & goto :fail)

set "BRANCH="
for /f "delims=" %%B in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "BRANCH=%%B"
if /i not "%BRANCH%"=="main" (echo ОШИБКА: ветка "%BRANCH%", нужна main & goto :fail)

set "DIRTY="
for /f "delims=" %%L in ('git status --porcelain --untracked-files=no') do set "DIRTY=1"
if defined DIRTY (echo ОШИБКА: есть незакоммиченные изменения в отслеживаемых файлах & git status --short --untracked-files=no & goto :fail)

echo.
echo === [2/5] Параметры ===
set "WEEK="
set /p "WEEK=Неделя [7]: "
if not defined WEEK set "WEEK=7"
set "DAY="
set /p "DAY=День [35]: "
if not defined DAY set "DAY=35"
echo Видео должно уже лежать в VIDEO_DIR: запись выполняется отдельно, до этого прогона.
echo Неделя %WEEK%, день %DAY%.

echo.
echo === [3/5] Пайплайн: на gate нужно отвечать здесь, в этом окне ===
echo === %date% %time% week=%WEEK% day=%DAY% === >> "%LOG%"
powershell -NoProfile -Command "uv run adventship run --day %DAY% --week %WEEK% --no-record 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE"
if errorlevel 1 (echo ОШИБКА: пайплайн остановился, см. %LOG% & goto :fail)

echo.
echo === [4/5] Итоговая таблица стадий ===
powershell -NoProfile -Command "uv run adventship status --day %DAY% --week %WEEK% 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE"
if errorlevel 1 (echo ОШИБКА: status завершился с ошибкой & goto :fail)

echo.
echo === [5/5] Готово ===
echo Лог: %LOG%
pause
exit /b 0

:fail
echo.
echo Остановлено. Лог: %LOG%
pause
exit /b 1
