<#
.SYNOPSIS  Обновление ассистента. Индекс и настройки сохраняются.
.PARAMETER Rollback   Вернуться к предыдущей версии.
.PARAMETER BackupOnly Только сделать резервную копию.
#>
param([switch]$Rollback, [switch]$BackupOnly, [string]$Target, [string]$From)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Установка — это папка с окружением Python, а не с исходным кодом. Клон
# репозитория ни тем, ни другим не является: раньше скрипт молча брал
# папку, из которой запущен, и падал на отсутствующем venv сообщением,
# по которому не догадаться, что обновлялась не та папка.
function Is-Install($path) {
  return (Test-Path (Join-Path $path "venv\Scripts\python.exe")) -and
         (Test-Path (Join-Path $path "webui.py"))
}

if (-not $Target) {
  if (Is-Install $Here) { $Target = $Here }
  else {
    foreach ($c in @("$env:USERPROFILE\kb-assistant", "C:\KBAssistant",
                     "C:\kb-assistant", $Here)) {
      if (Is-Install $c) { $Target = $c; break }
    }
  }
}
if (-not $Target -or -not (Is-Install $Target)) {
  Write-Host "  [x] Не нашёл установку ассистента." -ForegroundColor Red
  Write-Host "  Папка $Here — это исходный код, а не установка: в ней нет venv."
  Write-Host "  Укажите установку явно:"
  Write-Host "      .\update.ps1 -Target C:\путь\к\установке -From $Here"
  exit 2
}
if ($From -and ((Resolve-Path $From).Path -eq (Resolve-Path $Target).Path)) { $From = "" }

$Dir = $Target
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

Say "Обновляю код"
Write-Host "  установка: $Dir"
if ($From) {
  # Копируем только код: данные, настройки и ключи установки остаются.
  Write-Host "  источник:  $From"
  if (Test-Path "$From\.git") { git -C $From pull --ff-only 2>$null }
  $skip = @("data", "logs", "venv", "backups", ".env", "secrets.env", ".git")
  Get-ChildItem $From -Force | Where-Object { $skip -notcontains $_.Name } |
    ForEach-Object { Copy-Item $_.FullName $Dir -Recurse -Force }
  Ok "Код обновлён из $From"
} elseif (Test-Path "$Dir\.git") {
  git -C $Dir pull --ff-only; Ok "Код обновлён"
} else {
  Warn "Установка не под git и источник не указан — обновлять нечем."
  Warn "Укажите папку с кодом: .\update.ps1 -Target $Dir -From C:\путь\к\репозиторию"
}

# Сбой установки библиотек не должен рушить обновление: код уже
# скопирован, и прерывание оставляет установку наполовину обновлённой.
Say "Обновляю зависимости"
try {
  & "$Dir\venv\Scripts\python.exe" -m pip install --quiet --upgrade -r "$Dir\requirements.txt"
  Ok "Готово"
} catch {
  Warn "не удалось обновить библиотеки: $($_.Exception.Message)"
  Warn "код обновлён; повторить: $Dir\venv\Scripts\python.exe -m pip install -r $Dir\requirements.txt"
}

Say "Проверяю модули"
& "$Dir\venv\Scripts\python.exe" -c @"
import sys; sys.path.insert(0, r'$Dir')
for m in ('index','search','answer','bot','webui','models','metrics','llm','llm_queue','jobs','security','alerts','retention','backup','ocr','rerank','lsa','analytics','access','tracing','regression','contextual'): __import__(m)
print('  все модули загружаются')
"@
# Процессы, убитые при остановке, могли оставить занятые места в очереди
# к модели. Само освободится по сроку, но это до пяти минут, в течение
# которых ассистент выглядит зависшим.
Say "Снимаю оборванные задачи"
& "$Dir\venv\Scripts\python.exe" "$Dir\jobs.py" reap

Say "Освобождаю очередь к модели"
& "$Dir\venv\Scripts\python.exe" "$Dir\llm_queue.py" clear

Say "Проверяю настройку после обновления"
& "$Dir\venv\Scripts\python.exe" "$Dir\preflight.py"

Write-Host "`nОБНОВЛЕНИЕ ЗАВЕРШЕНО. Копия: $dest" -ForegroundColor Green
