<#
.SYNOPSIS  Обновление ассистента. Индекс и настройки сохраняются.
.PARAMETER Rollback   Вернуться к предыдущей версии.
.PARAMETER BackupOnly Только сделать резервную копию.
#>
param([switch]$Rollback, [switch]$BackupOnly)
$ErrorActionPreference = "Stop"
$Dir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Backups = Join-Path $Dir "backups"
function Ok{param($m)Write-Host "  [+] $m" -ForegroundColor Green}
function Say{param($m)Write-Host "==> $m" -ForegroundColor Cyan}
function Warn{param($m)Write-Host "  [!] $m" -ForegroundColor Yellow}

trap { Write-Host "  [x] $($_.Exception.Message)" -ForegroundColor Red
       Write-Host "Откат: .\update.ps1 -Rollback" -ForegroundColor Yellow; exit 1 }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path $Backups | Out-Null

if ($Rollback) {
  $last = Get-ChildItem $Backups -Directory | Sort-Object LastWriteTime -Desc | Select-Object -First 1
  if (-not $last) { Warn "Копий нет"; exit 1 }
  Say "Возвращаю версию из $($last.FullName)"
  Copy-Item -Recurse -Force "$($last.FullName)\*" $Dir
  Ok "Откат выполнен"; exit 0
}

Say "Делаю резервную копию"
$dest = Join-Path $Backups $stamp
New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($i in @(".env","data","logs")) {
  if (Test-Path "$Dir\$i") { Copy-Item -Recurse -Force "$Dir\$i" $dest; Ok $i }
}
Get-ChildItem $Backups -Directory | Sort-Object LastWriteTime -Desc |
  Select-Object -Skip 5 | Remove-Item -Recurse -Force
if ($BackupOnly) { Ok "Копия: $dest"; exit 0 }

Say "Останавливаю процессы"
Get-Process python* -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -like "$Dir*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Ok "Остановлено"

if (Test-Path "$Dir\.git") {
  Say "Обновляю код"; git -C $Dir pull --ff-only; Ok "Код обновлён"
} else { Warn "Не git-репозиторий: распакуйте новый архив поверх" }

Say "Обновляю зависимости"
& "$Dir\venv\Scripts\python.exe" -m pip install --quiet --upgrade -r "$Dir\requirements.txt"
Ok "Готово"

Say "Проверяю модули"
& "$Dir\venv\Scripts\python.exe" -c @"
import sys; sys.path.insert(0, r'$Dir')
for m in ('index','search','answer','bot','webui','models','metrics','llm','llm_queue','jobs','security','alerts','retention','backup','ocr','rerank','lsa','analytics','access','tracing','regression','contextual'): __import__(m)
print('  все модули загружаются')
"@
# Процессы, убитые при остановке, могли оставить занятые места в очереди
# к модели. Само освободится по сроку, но это до пяти минут, в течение
# которых ассистент выглядит зависшим.
Say "Освобождаю очередь к модели"
& "$Dir\venv\Scripts\python.exe" "$Dir\llm_queue.py" clear

Write-Host "`nОБНОВЛЕНИЕ ЗАВЕРШЕНО. Копия: $dest" -ForegroundColor Green
