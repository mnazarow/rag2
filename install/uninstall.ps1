<#
.SYNOPSIS  Удаление ассистента. Папка базы знаний не трогается.
.PARAMETER All      Удалить вместе с индексом и журналами.
.PARAMETER KeepEnv  Сохранить файл настроек отдельно.
#>
param([switch]$All, [switch]$KeepEnv)
$ErrorActionPreference = "Stop"
$Dir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
function Ok{param($m)Write-Host "  [+] $m" -ForegroundColor Green}
function Say{param($m)Write-Host "==> $m" -ForegroundColor Cyan}

$kb = ""
if (Test-Path "$Dir\.env") {
  $line = Select-String -Path "$Dir\.env" -Pattern '^KB_ROOT=' | Select-Object -First 1
  if ($line) { $kb = $line.Line.Split("=",2)[1] }
}
Write-Host "`nУдаление ассистента" -ForegroundColor Yellow
Write-Host "Папка программы: $Dir"
if ($kb) { Write-Host "Папка базы знаний: $kb - НЕ БУДЕТ ТРОНУТА" }
if ($All) { Write-Host "Индекс и журналы будут удалены." -ForegroundColor Red }
else { Write-Host "Индекс и журналы сохранятся." }
$answer = Read-Host "`nПродолжить? Введите: удалить"
if ($answer -ne "удалить") { Write-Host "Отменено."; exit 0 }

Say "Останавливаю процессы"
Get-Process python* -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "$Dir*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Ok "Остановлено"

Say "Убираю автозапуск"
if (Get-ScheduledTask -TaskName "KBAssistant" -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName "KBAssistant" -Confirm:$false; Ok "Задание удалено"
}
if ((Get-Command docker -ErrorAction SilentlyContinue) -and (Test-Path "$Dir\docker-compose.yml")) {
  docker compose -f "$Dir\docker-compose.yml" down 2>$null; Ok "Контейнеры остановлены"
}
if ($KeepEnv -and (Test-Path "$Dir\.env")) {
  Copy-Item "$Dir\.env" "$env:USERPROFILE\kb-assistant.env.saved"
  Ok "Настройки сохранены: $env:USERPROFILE\kb-assistant.env.saved"
}
Say "Удаляю файлы"
if ($All) { Remove-Item -Recurse -Force $Dir; Ok "Удалено полностью" }
else {
  Get-ChildItem $Dir -Force | Where-Object { $_.Name -notin @("data","logs",".env","backups") } |
    Remove-Item -Recurse -Force
  Ok "Программа удалена, данные сохранены"
}
Write-Host "`nГотово. Папка базы знаний не изменялась." -ForegroundColor Green
