#!/usr/bin/env bash
# Установка ассистента базы знаний на Linux и macOS.
#
#   ./install.sh                     — обычная установка
#   ./install.sh --docker            — через Docker Compose
#   ./install.sh --with-gpu          — плюс библиотеки для локальных моделей
#   ./install.sh --dir /opt/kb       — куда ставить
#   ./install.sh --service           — зарегистрировать автозапуск
#   ./install.sh --dry-run           — показать, что будет сделано
#
# Скрипт идемпотентен: повторный запуск ничего не ломает и лишнего не ставит.

set -Eeuo pipefail

APP_NAME="kb-assistant"
TARGET="${TARGET:-$HOME/$APP_NAME}"
WITH_DOCKER=0
WITH_GPU=0
WITH_SERVICE=0
DRY_RUN=0
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
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --docker) WITH_DOCKER=1 ;;
    --with-gpu) WITH_GPU=1 ;;
    --service) WITH_SERVICE=1 ;;
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
step "Проверяю системные утилиты"
MISSING=""
check_tool() {
  if command -v "$1" >/dev/null 2>&1; then ok "$1 — есть"
  else warn "$1 — нет ($2)"; MISSING="$MISSING $1"; fi
}
check_tool ffmpeg    "видео, голосовые сообщения"
check_tool pdftotext "разбор PDF без PyMuPDF"
check_tool bsdtar    "распаковка архивов RAR"
check_tool git       "обновления"
if [ -n "$MISSING" ]; then
  echo
  warn "Не хватает:$MISSING"
  case "$PLATFORM" in
    linux)
      if command -v apt-get >/dev/null 2>&1; then
        echo "     sudo apt-get install -y ffmpeg poppler-utils libarchive-tools git \\"
        echo "                          tesseract-ocr tesseract-ocr-rus"
      elif command -v dnf >/dev/null 2>&1; then
        echo "     sudo dnf install -y ffmpeg poppler-utils bsdtar git tesseract tesseract-langpack-rus"
      fi ;;
    macos) echo "     brew install ffmpeg poppler libarchive git tesseract tesseract-lang" ;;
  esac
  echo "     Без них часть возможностей просто не включится — установка продолжится."
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
    UNIT=/etc/systemd/system/$APP_NAME.service
    say "Создаю службу systemd (нужны права root)"
    run "sudo tee $UNIT >/dev/null <<UNITEOF
[Unit]
Description=Ассистент корпоративной базы знаний
After=network-online.target

[Service]
Type=simple
User=$USER
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
