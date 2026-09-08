# Запуск демо в отдельном окне Windows Terminal — именно его снимает OBS.
#
# Отдельный файл, а не строка в командной строке wt.exe: Windows Terminal
# трактует `;` как свой разделитель команд и рвёт вложенную команду пополам.
#
# КАК ЗАПУСКАТЬ (три вещи, каждая из которых уже стоила дубля 2026-09-07).
# Ниже ROOT — корень репозитория, подставляется абсолютным путём:
#
#   wt.exe -w new pwsh -NoExit -WorkingDirectory $ROOT `
#          -File $ROOT/tools/record_demo.ps1 -Day 6 -Week 2
#
#   1. `-w new` ОБЯЗАТЕЛЕН. Без него wt открывает НОВУЮ ВКЛАДКУ в уже
#      запущенном окне. Дальше две беды: OBS снимает окно, то есть активную
#      вкладку, и в кадр уедет чужая; а закрытие окна с «лишней» вкладкой
#      убивает и запись — ровно так погиб дубль.
#   2. Путь к скрипту — АБСОЛЮТНЫЙ. wt стартует не в текущем каталоге, а в
#      профильном каталоге пользователя, и относительный
#      `tools\record_demo.ps1` не находится: pwsh выходит с кодом 64, не
#      добравшись ни до OBS, ни до репетиции.
#   3. `-WorkingDirectory` — на случай, если Set-Location ниже когда-нибудь
#      уедет; дешевле, чем разбираться потом.
#
# Что пошло не так, видно в logs/record_last.log даже когда окно уже закрыто.

param(
    [int]$Day = 1,
    [int]$Week = 1,
    [switch]$DryRun,
    # Интерактивный чат вместо сценария записи — чтобы проверить руками.
    [switch]$Chat
)

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Заголовок задаётся здесь, а не через `wt --title`: источник захвата OBS
# выбирает окно по заголовку, а wt ломает аргумент с пробелами.
$Host.UI.RawUI.WindowTitle = 'AI Advent 9'

Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# Транскрипт: окно записи может закрыться (или быть закрыто) раньше, чем кто-то
# прочитает ошибку, и тогда причина падения теряется совсем — так и вышло
# 2026-09-07. Пишем в logs/ (он gitignored: там же calls.jsonl с промптами).
# В try, потому что диагностика не должна мешать самой записи.
$log = Join-Path (Get-Location) 'logs\record_last.log'
$transcript = $false
try {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
    Start-Transcript -LiteralPath $log -Force | Out-Null
    $transcript = $true
} catch {
    Write-Host "транскрипт не пишется: $_" -ForegroundColor DarkYellow
}

if ($Chat) {
    # Неделя 02 — это `adventagent`, отдельная точка входа: команды
    # `advent w02 chat` не существует, и ручная проверка уехала бы в ошибку.
    if ($Week -eq 2) {
        Write-Host 'AI Advent 9 — агент недели 02. /exit для выхода.' -ForegroundColor Cyan
        Write-Host ''
        & uv run adventagent
    } else {
        Write-Host 'AI Advent 9 — интерактивный чат. /exit для выхода.' -ForegroundColor Cyan
        Write-Host ''
        & uv run advent w01 chat
    }
} else {
    $arguments = @('run', 'advent', 'record', '--day', $Day, '--week', $Week)
    if ($DryRun) { $arguments += '--dry-run' }
    & uv @arguments
}

$code = $LASTEXITCODE
Write-Host ''
Write-Host "=== finished, exit code: $code ==="

if ($transcript) { try { Stop-Transcript | Out-Null } catch { } }
# Код возврата отдельным файлом: ждущей стороне не надо разбирать транскрипт,
# чтобы понять, дошло ли дело до конца.
try { Set-Content -LiteralPath (Join-Path (Get-Location) 'logs\record_last.exit') -Value $code } catch { }
