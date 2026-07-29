#!/usr/bin/env bash
# Установка ассистента базы знаний на Linux и macOS.
#
#   ./install.sh                     — обычная установка
#   ./install.sh --docker            — через Docker Compose
#   ./install.sh --with-gpu          — плюс библиотеки для локальных моделей
#   ./install.sh --dir /opt/kb       — куда ставить
#   ./install.sh --service           — зарегистрировать автозапуск
#   ./install.sh --network           — открыть веб-интерфейс из сети (с паролем)
#   ./install.sh --no-packages       — не ставить системные пакеты самому
#   ./install.sh --dry-run           — показать, что будет сделано
#
# Скрипт идемпотентен: повторный запуск ничего не ломает и лишнего не ставит.

set -Eeuo pipefail

APP_NAME="kb-assistant"
TARGET="${TARGET:-$HOME/$APP_NAME}"
WITH_DOCKER=0
WITH_GPU=0
WITH_SERVICE=0
WITH_NETWORK=0
DRY_RUN=0
# Ставить недостающие системные пакеты самим. Включено: строчку «а теперь
# выполните вот эту команду» пропускают, и потом неделю выясняют, почему
# не распознаются сканы. Выключается там, где пакетами ведает
# конфигурация сервера.
WITH_PACKAGES=1
PYTHON_MIN="3.10"

RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; BLUE='\033[36m'; NC='\033[0m'
STEP=0

say()   { printf "${BLUE}==>${NC} %s\n" "$*"; }
ok()    { printf "  ${GREEN}✓${NC} %s\n" "$*"; }
warn()  { printf "  ${YELLOW}!${NC} %s\n" "$*"; }
fail()  { printf "  ${RED}✗${NC} %s\n" "$*" >&2; }
step()  { STEP=$((STEP+1)); printf "\n${BLUE}[%d/%d]${NC} %s\n" "$STEP" "$TOTAL_STEPS" "$*"; }
run()   { if [ "$DRY_RUN" = 1 ]; then printf "      would run: %s\n" "$*"; else eval "$@"; fi; }

on_error() {
  local code=$?
  local line=${BASH_LINENO[0]}
  fail "Установка прервана на строке $line, код $code."
  cat <<HINT

Что делать дальше:
  · посмотрите сообщение выше — обычно в нём прямо сказано, чего не хватает;
  · при нехватке прав запустите с sudo или выберите другую папку: --dir ~/kb;
  · если не ставятся пакеты Python, проверьте выход в интернет и наличие прокси;
  · повторный запуск скрипта безопасен: уже сделанные шаги пропускаются.
HINT
  exit "$code"
}
trap on_error ERR

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --docker) WITH_DOCKER=1 ;;
    --with-gpu) WITH_GPU=1 ;;
    --service) WITH_SERVICE=1 ;;
    --network) WITH_NETWORK=1 ;;
    --no-packages) WITH_PACKAGES=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --dir) shift; TARGET="${1:?после --dir нужен путь}" ;;
    -h|--help) usage ;;
    *) fail "Неизвестный ключ: $1"; usage ;;
  esac
  shift
done

TOTAL_STEPS=8
[ "$WITH_DOCKER" = 1 ] && TOTAL_STEPS=5

OS="$(uname -s)"
case "$OS" in
  Linux)  PLATFORM=linux ;;
  Darwin) PLATFORM=macos ;;
  *) fail "Поддерживаются Linux и macOS. Для Windows используйте install.ps1"; exit 1 ;;
esac

printf "\n${BLUE}Установка ассистента корпоративной базы знаний${NC}\n"
printf "Система: %s, папка установки: %s\n" "$PLATFORM" "$TARGET"
[ "$DRY_RUN" = 1 ] && warn "Пробный запуск: ничего меняться не будет"

# ---------------------------------------------------------------- Docker ----
if [ "$WITH_DOCKER" = 1 ]; then
  step "Проверяю Docker"
  command -v docker >/dev/null 2>&1 || { fail "Docker не установлен. https://docs.docker.com/get-docker/"; exit 1; }
  docker info >/dev/null 2>&1 || { fail "Docker установлен, но демон не запущен."; exit 1; }
  ok "Docker $(docker --version | cut -d' ' -f3 | tr -d ,)"
  if docker compose version >/dev/null 2>&1; then COMPOSE="docker compose";
  elif command -v docker-compose >/dev/null 2>&1; then COMPOSE="docker-compose";
  else fail "Не найден docker compose."; exit 1; fi
  ok "Compose: $COMPOSE"

  step "Копирую файлы в $TARGET"
  run "mkdir -p '$TARGET'"
  run "cp -r '$(cd "$(dirname "$0")/.." && pwd)/.' '$TARGET/'"
  ok "Скопировано"

  step "Готовлю настройки"
  if [ ! -f "$TARGET/.env" ]; then
    run "cp '$TARGET/.env.example' '$TARGET/.env'"
    warn "Создан .env из образца — укажите в нём KB_ROOT и токен бота"
  else ok ".env уже существует, не трогаю"; fi

  step "Собираю образ"
  run "cd '$TARGET' && $COMPOSE build"
  ok "Образ собран"

  step "Запускаю"
  run "cd '$TARGET' && $COMPOSE up -d"
  ok "Запущено"
  cat <<DONE

Готово. Что дальше:
  1. Откройте $TARGET/.env и укажите KB_ROOT — путь к папке базы знаний.
  2. Проиндексируйте: $COMPOSE exec app python index.py build
  3. Включите смысловой поиск (обязательно, иначе работает только
     поиск по точным словам):
       $COMPOSE exec app python index.py train-lsa
       $COMPOSE exec app python index.py reembed
  4. Поставьте резервное копирование:
       $COMPOSE exec app python backup.py create
  5. Админка: http://127.0.0.1:8800
  6. Журналы: $COMPOSE logs -f
DONE
  exit 0
fi

# --------------------------------------------------------------- Python -----
step "Проверяю Python"
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver=$("$candidate" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0)
    if [ "$(printf '%s\n%s\n' "$PYTHON_MIN" "$ver" | sort -V | head -1)" = "$PYTHON_MIN" ]; then
      PY="$candidate"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  fail "Нужен Python $PYTHON_MIN или новее."
  case "$PLATFORM" in
    linux) echo "     Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip";;
    macos) echo "     macOS: brew install python@3.12";;
  esac
  exit 1
fi
ok "$PY $($PY -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"

# ------------------------------------------------------- системные утилиты --
# Ставим сами. Раньше скрипт печатал команду и предлагал выполнить её
# руками — и это ровно та строчка, которую пропускают: установка ведь
# «прошла успешно». Потом выясняется, что видео не расшифровывается,
# сканы не распознаются, а архивы с сертификатами не распакованы, и
# связать это с пропущенной строкой в начале уже трудно.
#
# Правила здесь такие. Ставим только то, чего нет. Ошибка установки
# пакета не прерывает работу: без ffmpeg система ущербна, но работает, а
# упавшая на середине установка не работает вовсе. И всё это можно
# выключить ключом --no-packages: на сервере, где пакеты ставит
# конфигурация, самодеятельность вредна.
step "Системные утилиты"

# Что за что отвечает — чтобы предупреждение говорило о последствиях,
# а не об имени пакета.
TOOL_WHY_ffmpeg="видео и голосовые сообщения"
TOOL_WHY_pdftotext="разбор PDF, если не встанет PyMuPDF"
TOOL_WHY_bsdtar="распаковка архивов RAR с сертификатами"
TOOL_WHY_git="обновления"
TOOL_WHY_tesseract="распознавание сканов сертификатов"

MISSING=""
for tool in ffmpeg pdftotext bsdtar git tesseract; do
  if command -v "$tool" >/dev/null 2>&1; then ok "$tool — есть"
  else
    eval "why=\${TOOL_WHY_$tool}"
    warn "$tool — нет ($why)"
    MISSING="$MISSING $tool"
  fi
done

# Пакеты, в которых лежат эти утилиты. Имена у каждого менеджера свои.
packages_for() {
  local mgr="$1" out=""
  for tool in $MISSING; do
    case "$mgr:$tool" in
      apt:ffmpeg)     out="$out ffmpeg" ;;
      apt:pdftotext)  out="$out poppler-utils" ;;
      apt:bsdtar)     out="$out libarchive-tools" ;;
      apt:git)        out="$out git" ;;
      apt:tesseract)  out="$out tesseract-ocr tesseract-ocr-rus" ;;
      dnf:ffmpeg)     out="$out ffmpeg-free" ;;
      dnf:pdftotext)  out="$out poppler-utils" ;;
      dnf:bsdtar)     out="$out bsdtar" ;;
      dnf:git)        out="$out git" ;;
      dnf:tesseract)  out="$out tesseract tesseract-langpack-rus" ;;
      pacman:ffmpeg)    out="$out ffmpeg" ;;
      pacman:pdftotext) out="$out poppler" ;;
      pacman:bsdtar)    out="$out libarchive" ;;
      pacman:git)       out="$out git" ;;
      pacman:tesseract) out="$out tesseract tesseract-data-rus" ;;
      zypper:ffmpeg)    out="$out ffmpeg" ;;
      zypper:pdftotext) out="$out poppler-tools" ;;
      zypper:bsdtar)    out="$out libarchive" ;;
      zypper:git)       out="$out git" ;;
      zypper:tesseract) out="$out tesseract-ocr tesseract-ocr-traineddata-russian" ;;
      brew:ffmpeg)      out="$out ffmpeg" ;;
      brew:pdftotext)   out="$out poppler" ;;
      brew:bsdtar)      out="$out libarchive" ;;
      brew:git)         out="$out git" ;;
      brew:tesseract)   out="$out tesseract tesseract-lang" ;;
    esac
  done
  printf "%s" "$out"
}

# Python-окружение на Debian и Ubuntu лежит в отдельном пакете, и без
# него venv не создастся вовсе — а это уже не «часть возможностей».
if [ "$PLATFORM" = linux ] && command -v apt-get >/dev/null 2>&1; then
  if ! "$PY" -c "import venv, ensurepip" >/dev/null 2>&1; then
    warn "python3-venv — нет (без него не создать окружение Python)"
    MISSING="$MISSING venv"
  fi
fi

if [ -n "$MISSING" ] && [ "$WITH_PACKAGES" = 1 ]; then
  # Кем ставить. Под root sudo не нужен, без root и без sudo — только
  # показать команду: молча требовать пароль в середине установки хуже,
  # чем честно сказать, что прав не хватает.
  SUDO=""
  if [ "$(id -u)" != 0 ]; then
    if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
  fi

  MGR=""
  case "$PLATFORM" in
    linux)
      if   command -v apt-get >/dev/null 2>&1; then MGR=apt
      elif command -v dnf     >/dev/null 2>&1; then MGR=dnf
      elif command -v pacman  >/dev/null 2>&1; then MGR=pacman
      elif command -v zypper  >/dev/null 2>&1; then MGR=zypper
      fi ;;
    macos) command -v brew >/dev/null 2>&1 && MGR=brew ;;
  esac

  PKGS="$(packages_for "$MGR")"
  [ "$MGR" = apt ] && case " $MISSING " in *" venv "*) PKGS="$PKGS python3-venv" ;; esac

  if [ -z "$MGR" ]; then
    if [ "$PLATFORM" = macos ]; then
      warn "Не найден Homebrew — без него пакеты не поставить."
      echo "     Поставьте его одной командой и запустите установку снова:"
      echo '     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    else
      warn "Не понял, какой в системе менеджер пакетов — поставьте вручную:$MISSING"
    fi
  elif [ -z "$SUDO" ] && [ "$(id -u)" != 0 ] && [ "$MGR" != brew ]; then
    warn "Нет прав на установку пакетов. Выполните и запустите снова:"
    case "$MGR" in
      apt)    echo "     sudo apt-get update && sudo apt-get install -y$PKGS" ;;
      dnf)    echo "     sudo dnf install -y$PKGS" ;;
      pacman) echo "     sudo pacman -S --needed --noconfirm$PKGS" ;;
      zypper) echo "     sudo zypper install -y$PKGS" ;;
    esac
  elif [ -n "$PKGS" ]; then
    say "Ставлю недостающие пакеты:$PKGS"
    # Сбой не прерывает установку: часть пакетов может называться иначе
    # или отсутствовать в репозитории, и это повод сообщить, а не падать.
    case "$MGR" in
      apt)    run "$SUDO apt-get update -qq" || true
              run "$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq$PKGS" \
                || warn "часть пакетов не установилась" ;;
      dnf)    run "$SUDO dnf install -y -q$PKGS" || warn "часть пакетов не установилась" ;;
      pacman) run "$SUDO pacman -S --needed --noconfirm$PKGS" \
                || warn "часть пакетов не установилась" ;;
      zypper) run "$SUDO zypper --non-interactive install$PKGS" \
                || warn "часть пакетов не установилась" ;;
      brew)   run "brew install$PKGS" || warn "часть пакетов не установилась" ;;
    esac
    if [ "$DRY_RUN" = 0 ]; then
      LEFT=""
      for tool in $MISSING; do
        [ "$tool" = venv ] && continue
        command -v "$tool" >/dev/null 2>&1 && ok "$tool — установлен" || LEFT="$LEFT $tool"
      done
      [ -n "$LEFT" ] && warn "так и не появились:$LEFT — часть возможностей не включится"
    fi
  fi
elif [ -n "$MISSING" ]; then
  warn "Не хватает:$MISSING — установка пакетов выключена ключом --no-packages"
  echo "     Без них часть возможностей просто не включится."
fi

# ------------------------------------------------------------ размещение ----
step "Размещаю файлы в $TARGET"
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
run "mkdir -p '$TARGET'"
if [ "$SOURCE" != "$TARGET" ]; then
  run "cp -r '$SOURCE/.' '$TARGET/'"
  ok "Файлы скопированы"
else ok "Уже на месте"; fi
run "mkdir -p '$TARGET/data' '$TARGET/logs'"

# ------------------------------------------------------------- окружение ---
step "Создаю окружение Python"
if [ ! -d "$TARGET/venv" ]; then
  run "'$PY' -m venv '$TARGET/venv'"
  ok "Создано"
else ok "Уже существует"; fi
VENV_PY="$TARGET/venv/bin/python"
run "'$VENV_PY' -m pip install --quiet --upgrade pip"

step "Ставлю зависимости"
run "'$VENV_PY' -m pip install --quiet -r '$TARGET/requirements.txt'" \
  || { fail "Не установились зависимости."
       echo "     Проверьте интернет. За прокси: export HTTPS_PROXY=http://адрес:порт"
       exit 1; }
ok "Основные зависимости готовы"
if [ "$WITH_GPU" = 1 ]; then
  say "Ставлю библиотеки для локальных моделей — это долго и займёт несколько гигабайт"
  run "'$VENV_PY' -m pip install --quiet sentence-transformers faster-whisper" || \
    warn "Часть библиотек не поставилась — локальные модели можно доставить позже"
  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "Видеокарта найдена: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    run "'$VENV_PY' -m pip install --quiet vllm" || \
      warn "vllm не поставился — поставьте вручную по инструкции проекта"
  else
    warn "Видеокарта не найдена — локальные модели будут работать очень медленно"
  fi
fi

# -------------------------------------------------------------- настройки --
step "Готовлю настройки"
if [ ! -f "$TARGET/.env" ]; then
  run "cp '$TARGET/.env.example' '$TARGET/.env'"
  ok "Создан .env из образца"
else ok ".env уже есть, не перезаписываю"; fi
# В .env будет токен бота, в data/ — вопросы сотрудников: только владельцу.
run "chmod 600 '$TARGET/.env' 2>/dev/null || true"
run "chmod 700 '$TARGET/data' 2>/dev/null || true"

if [ "$WITH_NETWORK" = 1 ]; then
  # Доступ из сети. Наружу без пароля нельзя — админка управляет всей
  # системой, поэтому пароль генерируется, если его ещё нет.
  if grep -q "^ADMIN_HOST=0.0.0.0" "$TARGET/.env" 2>/dev/null; then
    ok "Доступ из сети уже настроен"
  else
    run "printf '\n# Доступ из сети (добавлено установщиком по --network)\nADMIN_HOST=0.0.0.0\n' >> '$TARGET/.env'"
  fi
  if ! grep -q "^ADMIN_TOKEN=." "$TARGET/.env" 2>/dev/null; then
    NEW_TOKEN="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    run "printf 'ADMIN_TOKEN=%s\n' '$NEW_TOKEN' >> '$TARGET/.env'"
    ok "Пароль администратора создан: $NEW_TOKEN — сохраните его"
  else
    ok "Пароль администратора уже задан (ADMIN_TOKEN в .env)"
  fi
  LAN_IP=""
  if [ "$PLATFORM" = macos ]; then LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
  else LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"; fi
  [ -n "$LAN_IP" ] && ok "Из сети: http://$LAN_IP:8800 (пароль — ADMIN_TOKEN)"
fi

# ------------------------------------------------------------- автозапуск --
step "Проверка и автозапуск"
if [ "$DRY_RUN" = 0 ]; then
  ( cd "$TARGET" && "$VENV_PY" -c "import config, db; db.init(); print('  проверка хранилища: ок')" )
  ( cd "$TARGET" && "$VENV_PY" -c "
import llm_queue, config
llm_queue.ensure_tables()
print(f'  очередь к модели: не больше {config.LLM_MAX_CONCURRENT} запросов одновременно')" )
  say "Проверяю настройку перед первым запуском"
  ( cd "$TARGET" && "$VENV_PY" preflight.py ) || \
    warn "Проверка нашла то, что нужно исправить до запуска (см. выше)"
fi
if [ "$WITH_SERVICE" = 1 ]; then
  if [ "$PLATFORM" = linux ] && command -v systemctl >/dev/null 2>&1; then
    # Имя пользователя берём надёжным способом. Переменная USER в
    # неинтерактивной оболочке — при установке по SSH одной командой, из
    # cron или из скрипта — может быть не задана вовсе, и при set -u
    # установка падала на «USER: unbound variable» ровно в тот момент,
    # когда всё остальное уже сделано.
    RUN_AS="${SUDO_USER:-${USER:-$(id -un)}}"
    UNIT=/etc/systemd/system/$APP_NAME.service
    say "Создаю службу systemd (нужны права root)"
    run "sudo tee $UNIT >/dev/null <<UNITEOF
[Unit]
Description=Ассистент корпоративной базы знаний
After=network-online.target

[Service]
Type=simple
User=$RUN_AS
WorkingDirectory=$TARGET
ExecStart=$VENV_PY $TARGET/webui.py
Restart=on-failure
RestartSec=10
# Даём дописать индекс: остановка посреди записи оставляет обрезанный
# файл векторов, и смысловой поиск после этого молча исчезает.
KillSignal=SIGTERM
TimeoutStopSec=40
StandardOutput=append:$TARGET/logs/service.log
StandardError=append:$TARGET/logs/service.log

[Install]
WantedBy=multi-user.target
UNITEOF"
    run "sudo systemctl daemon-reload && sudo systemctl enable --now $APP_NAME"
    ok "Служба $APP_NAME запущена"
    # Бот — отдельной службой с безусловным перезапуском. Убитый ночью
    # OOM-киллером бот без Restart=always остаётся мёртвым до тех пор,
    # пока сотрудники не пожалуются, — а админка при этом зелёная.
    if grep -q "^TELEGRAM_BOT_TOKEN=." "$TARGET/.env" 2>/dev/null; then
      BOT_UNIT=/etc/systemd/system/$APP_NAME-bot.service
      run "sudo tee $BOT_UNIT >/dev/null <<BOTEOF
[Unit]
Description=Telegram-бот ассистента базы знаний
After=network-online.target

[Service]
Type=simple
User=$RUN_AS
WorkingDirectory=$TARGET
ExecStart=$VENV_PY $TARGET/bot.py
Restart=always
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
BOTEOF"
      run "sudo systemctl daemon-reload && sudo systemctl enable --now $APP_NAME-bot"
      ok "Служба $APP_NAME-bot запущена"
    else
      warn "Токен Telegram не задан — служба бота не создана. Задайте"
      echo "     TELEGRAM_BOT_TOKEN в .env и запустите установку с --service снова."
    fi
  elif [ "$PLATFORM" = macos ]; then
    PLIST="$HOME/Library/LaunchAgents/ru.company.$APP_NAME.plist"
    run "mkdir -p '$HOME/Library/LaunchAgents'"
    run "cat > '$PLIST' <<PLISTEOF
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
  <key>Label</key><string>ru.company.$APP_NAME</string>
  <key>ProgramArguments</key><array>
    <string>$VENV_PY</string><string>$TARGET/webui.py</string></array>
  <key>WorkingDirectory</key><string>$TARGET</string>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$TARGET/logs/service.log</string>
  <key>StandardErrorPath</key><string>$TARGET/logs/service.log</string>
</dict></plist>
PLISTEOF"
    run "launchctl unload '$PLIST' 2>/dev/null || true"
    run "launchctl load '$PLIST'"
    ok "Автозапуск настроен"
    if grep -q "^TELEGRAM_BOT_TOKEN=." "$TARGET/.env" 2>/dev/null; then
      BOT_PLIST="$HOME/Library/LaunchAgents/ru.company.$APP_NAME-bot.plist"
      run "cat > '$BOT_PLIST' <<BOTPLISTEOF
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
  <key>Label</key><string>ru.company.$APP_NAME-bot</string>
  <key>ProgramArguments</key><array>
    <string>$VENV_PY</string><string>$TARGET/bot.py</string></array>
  <key>WorkingDirectory</key><string>$TARGET</string>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$TARGET/logs/bot-service.log</string>
  <key>StandardErrorPath</key><string>$TARGET/logs/bot-service.log</string>
</dict></plist>
BOTPLISTEOF"
      run "launchctl unload '$BOT_PLIST' 2>/dev/null || true"
      run "launchctl load '$BOT_PLIST'"
      ok "Автозапуск бота настроен"
    else
      warn "Токен Telegram не задан — автозапуск бота не настроен. Задайте"
      echo "     TELEGRAM_BOT_TOKEN в .env и запустите установку с --service снова."
    fi
  else
    warn "Автозапуск на этой системе не настроен — запускайте вручную"
  fi
fi

cat <<DONE

УСТАНОВКА ЗАВЕРШЕНА

Самый простой путь дальше — открыть веб-интерфейс и пройти по шагам там:
там же видно, какие из них уже сделаны.

       $TARGET/venv/bin/python $TARGET/webui.py
   и открыть http://127.0.0.1:8800 — раздел «Быстрый старт»

Те же шаги командами, если интерфейс недоступен:
  1. Откройте настройки и укажите путь к базе знаний:
       nano $TARGET/.env          (параметр KB_ROOT)
  2. Проиндексируйте базу:
       $TARGET/venv/bin/python $TARGET/index.py build
  3. Включите смысловой поиск — без этого шага находятся только точные
     слова из документа, а вопросы своими словами остаются без ответа:
       $TARGET/venv/bin/python $TARGET/index.py train-lsa
       $TARGET/venv/bin/python $TARGET/index.py reembed
  4. Проверьте поиск:
       $TARGET/venv/bin/python $TARGET/ask.py "какой напор у Водомет 55/50"
  5. Поставьте регулярное резервное копирование индекса:
       $TARGET/venv/bin/python $TARGET/backup.py create
       $TARGET/venv/bin/python $TARGET/backup.py schedule
  6. Если в базе есть сканы сертификатов — распознайте их:
       $TARGET/venv/bin/python $TARGET/ocr.py providers
       $TARGET/venv/bin/python $TARGET/ocr.py run
  7. Если модель локальная, проверьте очередь запросов к ней. По умолчанию
     стоит один запрос одновременно: для одной видеокарты это правильно —
     иначе десять одновременных вопросов делят её память и все отвечают
     медленно. Посмотреть, как очередь ведёт себя под нагрузкой:
       $TARGET/venv/bin/python $TARGET/llm_queue.py status
       $TARGET/venv/bin/python $TARGET/llm_queue.py stats
  8. Запустите веб-интерфейс:
       $TARGET/venv/bin/python $TARGET/webui.py
     и откройте http://127.0.0.1:8800
  9. Перед выставлением админки наружу проверьте настройку:
       $TARGET/venv/bin/python $TARGET/preflight.py
     Она не даст запуститься с открытым наружу интерфейсом без пароля и
     с ролью по умолчанию, которой нет в разграничении доступа.
 10. Для Telegram укажите в .env токен бота и запустите:
       $TARGET/venv/bin/python $TARGET/bot.py

Обновление: $TARGET/install/update.sh
Удаление:   $TARGET/install/uninstall.sh
Документация: $TARGET/ДОКУМЕНТАЦИЯ.md
DONE
