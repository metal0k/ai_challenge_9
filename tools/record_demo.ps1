# Запуск демо в отдельном окне Windows Terminal — именно его снимает OBS.
#
# Отдельный файл, а не строка в командной строке wt.exe: Windows Terminal
# трактует `;` как свой разделитель команд и рвёт вложенную команду пополам.

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

if ($Chat) {
    Write-Host 'AI Advent 9 — интерактивный чат. /exit для выхода.' -ForegroundColor Cyan
    Write-Host ''
    & uv run advent w01 chat
} else {
    $arguments = @('run', 'advent', 'record', '--day', $Day, '--week', $Week)
    if ($DryRun) { $arguments += '--dry-run' }
    & uv @arguments
}

$code = $LASTEXITCODE
Write-Host ''
Write-Host "=== finished, exit code: $code ==="
