Set-Location -LiteralPath 'D:\Dev\PROJECTS\VibeCoding\AI_Challenge'
$Host.UI.RawUI.WindowTitle = 'AI Advent Live'
$env:VIDEO_DIR = 'D:\Dev\PROJECTS\VibeCoding\AI_Challenge\logs\videos'
Remove-Item Env:NO_COLOR -ErrorAction SilentlyContinue
Start-Sleep -Seconds 4
& '.\.venv\Scripts\advent.exe' record --week 3 --day 11 --live --keep-original
Write-Host "RECORD_EXIT_CODE=$LASTEXITCODE"
