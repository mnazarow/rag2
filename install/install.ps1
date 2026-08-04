<#
.SYNOPSIS
  Установка ассистента корпоративной базы знаний на Windows.

.DESCRIPTION
  Проверяет Python и системные утилиты, создаёт окружение, ставит
  зависимости, готовит настройки и при необходимости регистрирует службу.
  Повторный запуск безопасен: сделанные шаги пропускаются.

.PARAMETER Dir
  Куда установить. По умолчанию C:\KBAssistant

.PARAMETER Docker
  Установка через Docker Desktop вместо обычной.

.PARAMETER WithGpu
  Дополнительно поставить библиотеки для локальных моделей.

.PARAMETER Service
  Оставлен для совместимости: автозапуск теперь включён по умолчанию.

.PARAMETER NoService
  Не регистрировать автозапуск (запуск руками).

.PARAMETER NoNetwork
  Не открывать веб-интерфейс из сети: только с этой машины.
  По умолчанию доступ из сети включается, пароль генерируется сам.

.PARAMETER DryRun
  Показать, что будет сделано, ничего не меняя.

.PARAMETER NoPackages
  Не ставить системные утилиты (ffmpeg, poppler, git, tesseract) через winget.
  Список недостающих всё равно будет показан.

.EXAMPLE
  .\install.ps1 -Dir D:\KB -WithGpu
.EXAMPLE
  .\install.ps1 -NoService -NoNetwork   # только этот компьютер, запуск руками
#>
[CmdletBinding()]
param(
  [string]$Dir = "C:\KBAssistant",
  [switch]$Docker,
  [switch]$WithGpu,
  [switch]$Service,
  [switch]$NoService,
  [switch]$NoNetwork,
  [switch]$DryRun,
  [switch]$NoPackages
)

$ErrorActionPreference = "Stop"
$WithPackages = -not $NoPackages
# Автозапуск и доступ из сети — по умолчанию: систему ставят для
# компании, а не для одного рабочего стола. Отказ: -NoService, -NoNetwork.
$WithService = $Service -or (-not $NoService)
$WithNetwork = -not $NoNetwork
$script:Step = 0
$script:Total = if ($Docker) { 5 } else { 8 }

function Say  { param($m) Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   { param($m) Write-Host "  [+] $m" -ForegroundColor Green }
function Warn { param($m) Write-Host "  [!] $m" -ForegroundColor Yellow }
function Fail { param($m) Write-Host "  [x] $m" -ForegroundColor Red }
function Step { param($m) $script:Step++; Write-Host "`n[$script:Step/$script:Total] $m" -ForegroundColor Cyan }
function Run  { param($cmd) if ($DryRun) { Write-Host "      would run: $cmd" } else { Invoke-Expression $cmd } }

function Show-Help {
  Write-Host @"

Установка прервана. Что обычно помогает:
  · запустите PowerShell от имени администратора;
  · если скрипты запрещены: Set-ExecutionPolicy -Scope Process Bypass;
  · при ошибках установки пакетов проверьте выход в интернет и прокси:
      `$env:HTTPS_PROXY = "http://адрес:порт"
  · повторный запуск безопасен.

"@ -ForegroundColor Yellow
}

trap {
  Fail "Ошибка: $($_.Exception.Message)"
  Fail "Строка: $($_.InvocationInfo.ScriptLineNumber)"
  Show-Help
  exit 1
}

Write-Host "`nУстановка ассистента корпоративной базы знаний" -ForegroundColor Cyan
Write-Host "Папка установки: $Dir"
if ($DryRun) { Warn "Пробный запуск: изменений не будет" }

$Source = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# ------------------------------------------------------------- Docker ------
if ($Docker) {
  Step "Проверяю Docker"
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "Docker Desktop не установлен: https://www.docker.com/products/docker-desktop"
    exit 1
  }
  docker info *> $null
  if ($LASTEXITCODE -ne 0) { Fail "Docker установлен, но не запущен. Запустите Docker Desktop."; exit 1 }
  Ok "Docker работает"

  Step "Копирую файлы"
  Run "New-Item -ItemType Directory -Force -Path '$Dir' | Out-Null"
  Run "Copy-Item -Recurse -Force '$Source\*' '$Dir'"
  Ok "Скопировано"

  Step "Готовлю настройки"
  if (-not (Test-Path "$Dir\.env")) {
    Run "Copy-Item '$Dir\.env.example' '$Dir\.env'"
    Warn "Создан .env — укажите KB_ROOT"
  } else { Ok ".env уже есть" }

  Step "Собираю образ"
  Run "docker compose -f '$Dir\docker-compose.yml' build"
  Ok "Собран"

  Step "Запускаю"
  Run "docker compose -f '$Dir\docker-compose.yml' up -d"
  Ok "Запущено"
  Write-Host "`nГотово. Админка: http://127.0.0.1:8800" -ForegroundColor Green
  exit 0
}

# ------------------------------------------------------------- Python ------
Step "Проверяю Python"
$py = $null
foreach ($c in @("python", "python3", "py")) {
  $cmd = Get-Command $c -ErrorAction SilentlyContinue
  if ($cmd) {
    $v = & $c -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
    if ($v -and [version]$v -ge [version]"3.10") { $py = $c; break }
  }
}
if (-not $py) {
  Fail "Нужен Python 3.10 или новее."
  Write-Host "     Скачайте: https://www.python.org/downloads/  (отметьте «Add to PATH»)"
  Write-Host "     Или: winget install Python.Python.3.12"
  exit 1
}
Ok "$py $(& $py -c 'import sys;print(\"%d.%d.%d\"%sys.version_info[:3])')"

# Ставим сами. Строчку «а теперь выполните вот это» пропускают, а потом
# неделю выясняют, почему не расшифровывается видео и не распознаются
# сканы. Сбой установки пакета не прерывает работу: без утилиты система
# ущербна, но работает, а прерванная установка не работает вовсе.
Step "Системные утилиты"
$tools = @(
  @{ Cmd = "ffmpeg";    Why = "видео и голосовые сообщения"; Pkg = "Gyan.FFmpeg" },
  @{ Cmd = "pdftotext"; Why = "разбор PDF, если не встанет PyMuPDF"; Pkg = "oschwartz10612.Poppler" },
  @{ Cmd = "git";       Why = "обновления"; Pkg = "Git.Git" },
  @{ Cmd = "tesseract"; Why = "распознавание сканов сертификатов"; Pkg = "UB-Mannheim.TesseractOCR" },
  @{ Cmd = "ollama";    Why = "запуск локальных моделей"; Pkg = "Ollama.Ollama" },
  @{ Cmd = "ODAFileConverter"; Why = "чтение чертежей DWG"; Pkg = "OpenDesignAlliance.ODAFileConverter" },
  @{ Cmd = "freecadcmd"; Why = "чтение моделей STEP и IGES"; Pkg = "FreeCAD.FreeCAD" }
)
$missing = @()
foreach ($t in $tools) {
  if (Get-Command $t.Cmd -ErrorAction SilentlyContinue) { Ok "$($t.Cmd) — есть" }
  else { Warn "$($t.Cmd) — нет ($($t.Why))"; $missing += $t }
}
if ($missing.Count -gt 0) {
  if (-not $WithPackages) {
    Warn "Установка пакетов выключена ключом -NoPackages. Без них часть возможностей не включится."
  } elseif (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Warn "Нет winget — поставьте вручную:"
    foreach ($t in $missing) { Write-Host "     winget install --id $($t.Pkg)" }
  } else {
    Say "Ставлю недостающие пакеты: $(($missing | ForEach-Object { $_.Cmd }) -join ', ')"
    $failed = @()
    foreach ($t in $missing) {
      $global:LASTEXITCODE = 0
      try {
        Run "winget install --id $($t.Pkg) -e --accept-package-agreements --accept-source-agreements --silent"
      } catch {
        Warn "$($t.Cmd): $($_.Exception.Message)"
      }
      # winget не бросает исключений: смотрим код возврата.
      if (-not $DryRun -and $LASTEXITCODE -ne 0) { $failed += $t.Cmd }
      else { Ok "$($t.Cmd) — поставлен" }
    }
    if ($failed.Count -gt 0) {
      Warn "Не установились: $($failed -join ', '). Поставьте вручную и запустите установку снова."
      foreach ($t in $missing | Where-Object { $failed -contains $_.Cmd }) {
        Write-Host "     winget install --id $($t.Pkg)"
      }
    }
    Warn "Новые утилиты появятся в PATH только после перезапуска терминала."
  }
}

Step "Размещаю файлы в $Dir"
Run "New-Item -ItemType Directory -Force -Path '$Dir','$Dir\data','$Dir\logs' | Out-Null"
if ($Source -ne $Dir) { Run "Copy-Item -Recurse -Force '$Source\*' '$Dir'"; Ok "Скопировано" }
else { Ok "Уже на месте" }

Step "Создаю окружение Python"
if (-not (Test-Path "$Dir\venv")) { Run "& $py -m venv '$Dir\venv'"; Ok "Создано" }
else { Ok "Уже существует" }
$venvPy = "$Dir\venv\Scripts\python.exe"
Run "& '$venvPy' -m pip install --quiet --upgrade pip"

Step "Ставлю зависимости"
Run "& '$venvPy' -m pip install --quiet -r '$Dir\requirements.txt'"
Ok "Готово"
if ($WithGpu) {
  Say "Ставлю библиотеки для локальных моделей — это долго"
  try { Run "& '$venvPy' -m pip install --quiet sentence-transformers faster-whisper" ; Ok "Установлено" }
  catch { Warn "Часть библиотек не поставилась, можно доставить позже" }
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Ok "Видеокарта найдена"
    Warn "vllm под Windows официально не поддерживается — используйте Ollama или WSL2"
  } else { Warn "Видеокарта не найдена: локальные модели будут медленными" }
}

Step "Готовлю настройки"
if (-not (Test-Path "$Dir\.env")) { Run "Copy-Item '$Dir\.env.example' '$Dir\.env'"; Ok "Создан .env" }
else { Ok ".env уже есть" }

if ($WithNetwork) {
  # Наружу без входа нельзя: заводится учётная запись admin/admin, если
  # записей ещё нет; проверки будут напоминать сменить пароль.
  $envText = if (Test-Path "$Dir\.env") { Get-Content "$Dir\.env" -Raw } else { "" }
  if ($envText -notmatch "(?m)^ADMIN_HOST=0\.0\.0\.0") {
    Run "Add-Content -Path '$Dir\.env' -Value 'ADMIN_HOST=0.0.0.0'"
  } else { Ok "Доступ из сети уже настроен" }
  if (-not $DryRun) {
    Push-Location $Dir
    try {
      & $venvPy -c "import security; print('  ' + security.ensure_default_admin()['message'])"
    } catch { Warn "Не удалось создать учётную запись — заведите её в разделе «Безопасность»" }
    Pop-Location
  } else { Write-Host "      would run: создание учётной записи admin/admin" }
  $lanIp = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -First 1).IPAddress
  if ($lanIp) { Ok "Из сети: http://${lanIp}:8800 (вход admin / admin)" }
  Warn "Windows спросит про брандмауэр при первом запуске — разрешите."
}

Step "Проверка и автозапуск"
if (-not $DryRun) {
  Push-Location $Dir
  & $venvPy -c "import config, db; db.init(); print('  проверка хранилища: ок')"
  & $venvPy -c "import llm_queue, config; llm_queue.ensure_tables(); print('  очередь к модели: не больше ' + str(config.LLM_MAX_CONCURRENT) + ' запросов одновременно')"
  Pop-Location
}
if ($WithService) {
  Say "Регистрирую задание автозапуска"
  try {
    $action  = New-ScheduledTaskAction -Execute $venvPy -Argument "$Dir\webui.py" -WorkingDirectory $Dir
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $set     = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    if (-not $DryRun) {
      Register-ScheduledTask -TaskName "KBAssistant" -Action $action -Trigger $trigger `
        -Settings $set -RunLevel Highest -Force | Out-Null
    }
    Ok "Задание KBAssistant создано"
    $envText = if (Test-Path "$Dir\.env") { Get-Content "$Dir\.env" -Raw } else { "" }
    if ($envText -match "(?m)^TELEGRAM_BOT_TOKEN=.+") {
      $botAction = New-ScheduledTaskAction -Execute $venvPy -Argument "$Dir\bot.py" -WorkingDirectory $Dir
      if (-not $DryRun) {
        Register-ScheduledTask -TaskName "KBAssistantBot" -Action $botAction -Trigger $trigger `
          -Settings $set -RunLevel Highest -Force | Out-Null
      }
      Ok "Задание KBAssistantBot создано"
    } else {
      Warn "Токен Telegram не задан — автозапуск бота появится после его"
      Write-Host "     указания в .env и повторного запуска установки."
    }
  } catch {
    # Автозапуск по умолчанию обязан уметь отступить: без прав
    # администратора это предупреждение, а не крах на последнем шаге.
    Warn "Не удалось создать автозапуск: $($_.Exception.Message)"
    Write-Host "     Запустите PowerShell от имени администратора и повторите,"
    Write-Host "     либо запускайте вручную: $venvPy $Dir\webui.py"
  }
}

Write-Host @"

УСТАНОВКА ЗАВЕРШЕНА

Веб-интерфейс: http://127.0.0.1:8800 (вход admin / admin — смените
пароль в разделе «Безопасность»).
Если автозапуск создан, он поднимется после перезагрузки; сейчас можно
запустить вручную: $venvPy $Dir\webui.py
Самый простой путь дальше — раздел «Быстрый старт» в веб-интерфейсе.

Те же шаги командами:
  1. Откройте настройки и укажите путь к базе знаний (KB_ROOT):
       notepad $Dir\.env
  2. Проиндексируйте базу:
       $venvPy $Dir\index.py build
  3. Включите смысловой поиск — без этого шага находятся только точные
     слова из документа, а вопросы своими словами остаются без ответа:
       $venvPy $Dir\index.py train-lsa
       $venvPy $Dir\index.py reembed
  4. Проверьте поиск:
       $venvPy $Dir\ask.py "какой напор у Водомет 55/50"
  5. Поставьте регулярные задания — копии, оповещения, очистку:
       $venvPy $Dir\schedule.py install
  6. Если модель локальная, посмотрите очередь запросов к ней. По умолчанию
     стоит один запрос одновременно: для одной видеокарты это правильно.
       $venvPy $Dir\llm_queue.py status
  7. Перед выставлением админки наружу проверьте настройку:
       $venvPy $Dir\preflight.py
  8. Запустите веб-интерфейс и откройте http://127.0.0.1:8800
       $venvPy $Dir\webui.py

Обновление: $Dir\install\update.ps1
Удаление:   $Dir\install\uninstall.ps1
Документация: $Dir\ДОКУМЕНТАЦИЯ.md

"@ -ForegroundColor Green
