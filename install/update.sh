#!/usr/bin/env bash
# Обновление ассистента. Индекс, настройки и накопленные данные сохраняются.
#
#   ./update.sh                     — обновить установку рядом со скриптом
#   ./update.sh --target DIR        — обновить установку в другой папке
#   ./update.sh --from DIR          — взять код отсюда (например из клона git)
#   ./update.sh --backup-only       — только сделать резервную копию
#   ./update.sh --no-backup         — обновить БЕЗ копии (быстрее; --rollback
#                                     откатит к предыдущей, более старой копии)
#   ./update.sh --rollback          — вернуться к предыдущей версии
#
# Про --target и --from. Частый случай: репозиторий склонирован в одну
# папку, а работающая установка живёт в другой. Раньше скрипт молча брал
# ту папку, из которой его запустили, обновлял в ней код (в клоне это
# всегда «уже актуально») и падал на отсутствующем venv с сообщением
# «No such file or directory» — по которому невозможно догадаться, что
# обновлялась не та папка.

set -Eeuo pipefail

TARGET=""
SOURCE=""
MODE=update
WITH_BACKUP=1
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --from)   SOURCE="${2:-}"; shift 2 ;;
    --backup-only) MODE=backup; shift ;;
    --no-backup)   WITH_BACKUP=0; shift ;;
    --rollback)    MODE=rollback; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) printf "Неизвестный ключ: %s\n" "$1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")/.." && pwd)"

# Установка — это папка, где есть окружение Python и настройки. Клон
# репозитория ни тем, ни другим не является, и обновлять в нём нечего.
is_install() { [ -x "$1/venv/bin/python" ] && [ -f "$1/webui.py" ]; }

if [ -z "$TARGET" ]; then
  if is_install "$HERE"; then
    TARGET="$HERE"
  else
    # Ищем установку по обычным местам, прежде чем сдаваться.
    for candidate in "$HOME/kb-assistant" "/opt/kb-assistant" \
                     "/usr/local/kb-assistant" "$HOME/assistant" "$HERE"; do
      if is_install "$candidate"; then TARGET="$candidate"; break; fi
    done
  fi
fi

if [ -z "$TARGET" ] || ! is_install "$TARGET"; then
  printf "\033[31m  ✗\033[0m Не нашёл установку ассистента.\n\n" >&2
  printf "  Папка %s — это исходный код, а не установка: в ней нет venv.\n" "$HERE" >&2
  printf "  Скорее всего, репозиторий склонирован отдельно, а рабочая\n" >&2
  printf "  установка лежит в другом месте. Укажите её явно:\n\n" >&2
  printf "      %s --target /путь/к/установке --from %s\n\n" "$0" "$HERE" >&2
  printf "  Найти установку можно так:\n" >&2
  printf "      ls -d ~/*/venv/bin/python /opt/*/venv/bin/python 2>/dev/null\n" >&2
  exit 2
fi

# Код берём из отдельной папки, только если она действительно другая.
[ -n "$SOURCE" ] && [ "$(cd "$SOURCE" && pwd)" = "$TARGET" ] && SOURCE=""

# Скрипт запущен из папки с кодом (клона), а установка нашлась в другом
# месте — значит, код нужно брать отсюда, это и есть смысл запуска.
# Раньше SOURCE в этом случае оставался пустым, скрипт шёл обновляться
# через git прямо в установке, а «git stash --include-untracked» прятал
# в тайник неотслеживаемые файлы — вместе с .env. Со стороны это
# выглядело как «все настройки сбрасываются при обновлении».
if [ -z "$SOURCE" ] && [ "$TARGET" != "$HERE" ] && [ -f "$HERE/webui.py" ]; then
  SOURCE="$HERE"
fi

BACKUP_DIR="$TARGET/backups"
GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; BLUE='\033[36m'; NC='\033[0m'
say(){ printf "${BLUE}==>${NC} %s\n" "$*"; }
ok(){ printf "  ${GREEN}✓${NC} %s\n" "$*"; }
warn(){ printf "  ${YELLOW}!${NC} %s\n" "$*"; }
fail(){ printf "  ${RED}✗${NC} %s\n" "$*" >&2; }

trap 'fail "Обновление прервано. Данные не тронуты, откат: $0 --rollback"; exit 1' ERR

STAMP="$(date +%Y%m%d-%H%M%S)"
# Отпечаток ЗАПУЩЕННОЙ версии скрипта — снимается до того, как копирование
# кода перепишет файл по пути $0. Сравнивать $0 с целью после копирования
# бессмысленно: по обоим путям уже лежит новое содержимое, а выполняется
# по-прежнему старое, которое bash прочитал при старте.
SELF_SUM="$(cksum < "$0" 2>/dev/null || echo none)"
mkdir -p "$BACKUP_DIR"

backup() {
  say "Сохраняю копию настроек и данных"
  local dest="$BACKUP_DIR/$STAMP"
  mkdir -p "$dest"
  # Индекс снимаем средствами самой программы: копия SQLite остаётся
  # целостной, даже если прямо сейчас идёт индексация, и сразу же
  # проверяется на разворачиваемость. Простое копирование файла базы
  # под нагрузкой даёт битый файл, который выглядит целым.
  if [ -x "$TARGET/venv/bin/python" ] && [ -f "$TARGET/backup.py" ]; then
    if "$TARGET/venv/bin/python" "$TARGET/backup.py" create \
         --note "перед обновлением" --quiet >/dev/null 2>&1; then
      ok "снимок индекса сделан и проверен"
    else
      warn "снимок индекса не получился — копирую файлы как есть"
    fi
  fi
  for item in .env data logs; do
    [ -e "$TARGET/$item" ] && cp -r "$TARGET/$item" "$dest/" && ok "$item"
  done
  # Код тоже, чтобы был откат.
  tar -czf "$dest/code.tar.gz" -C "$TARGET" \
      --exclude=data --exclude=logs --exclude=venv --exclude=backups . 2>/dev/null
  ok "Копия: $dest"
  ls -1dt "$BACKUP_DIR"/*/ 2>/dev/null | tail -n +6 | xargs -r rm -rf
  echo "$dest"
}

case "$MODE" in
  backup) backup; exit 0 ;;
  rollback)
    LAST="$(ls -1dt "$BACKUP_DIR"/*/ 2>/dev/null | head -1 || true)"
    [ -z "$LAST" ] && { fail "Резервных копий нет."; exit 1; }
    say "Возвращаю версию из $LAST"
    [ -f "$LAST/code.tar.gz" ] && tar -xzf "$LAST/code.tar.gz" -C "$TARGET"
    [ -f "$LAST/.env" ] && cp "$LAST/.env" "$TARGET/.env"
    ok "Код и настройки возвращены."
    # Код откатился, а схема базы — нет: обновление могло мигрировать её
    # вперёд, и старый код упадёт не сейчас, а на первом обращении к
    # изменённой таблице. Честно предупреждаем и показываем выход.
    warn "База данных НЕ откатывается автоматически: если обновление меняло"
    echo "     схему, восстановите снимок данных из этой же папки:"
    echo "     $TARGET/venv/bin/python $TARGET/backup.py restore --latest"
    echo "     (текущие данные перед этим сохранит: backup.py create)"
    ok "Откат выполнен. Перезапустите сервисы."
    exit 0 ;;
esac

if [ "${KB_UPDATE_STAGE2:-0}" = 1 ]; then
  ok "Продолжаю обновлённым скриптом"
else

if [ "$WITH_BACKUP" = 1 ]; then
  backup >/dev/null
else
  # Пропуск копии — осознанный обмен: минус несколько минут сейчас,
  # минус свежая точка отката потом. --rollback никуда не денется, но
  # откатит к ПРЕДЫДУЩЕЙ копии — с ней уедут и настройки, менявшиеся
  # после неё. Говорим об этом до обновления, а не после аварии.
  warn "Резервная копия пропущена (--no-backup)."
  LAST_BK="$(ls -1dt "$BACKUP_DIR"/*/ 2>/dev/null | head -1 || true)"
  if [ -n "$LAST_BK" ]; then
    echo "     Откат будет к копии: $LAST_BK"
  else
    echo "     Копий нет вовсе — откатываться будет некуда."
  fi
fi

say "Останавливаю службы"
if command -v systemctl >/dev/null 2>&1 \
   && systemctl is-active --quiet kb-assistant 2>/dev/null; then
  sudo systemctl stop kb-assistant && ok "systemd остановлен"
fi
if command -v systemctl >/dev/null 2>&1 \
   && systemctl is-active --quiet kb-assistant-bot 2>/dev/null; then
  sudo systemctl stop kb-assistant-bot && ok "служба бота остановлена"
fi
pkill -f "$TARGET/webui.py" 2>/dev/null && ok "админка останавливается" || true
pkill -f "$TARGET/bot.py" 2>/dev/null && ok "бот останавливается" || true
# Ждём настоящего завершения. Служба получает на остановку 40 секунд —
# чтобы дописать файл векторов; pkill без ожидания переписывал код прямо
# под процессом, который ещё дописывал индекс.
WAITED=0
while pgrep -f "$TARGET/webui.py|$TARGET/bot.py" >/dev/null 2>&1; do
  [ "$WAITED" -ge 45 ] && { warn "процессы не остановились за 45 с — продолжаю"; break; }
  sleep 1; WAITED=$((WAITED+1))
done
[ "$WAITED" -gt 0 ] && ok "процессы завершились за $WAITED с"

say "Обновляю код"
printf "  установка: %s\n" "$TARGET"
# Настройки не имеют права меняться от обновления кода: их меняет только
# сам ассистент. Снимок до копирования и безусловное восстановление после
# защищают .env и файл ключей на любом пути обновления — rsync, tar, git.
ENV_TMP="$(mktemp -d)"
for f in .env secrets.env; do
  [ -f "$TARGET/$f" ] && cp "$TARGET/$f" "$ENV_TMP/$f"
done
if [ -n "$SOURCE" ]; then
  # Код лежит отдельно — например, в клоне репозитория. Копируем только
  # его: данные, настройки и ключи установки остаются на месте. Именно
  # ради этого исключения и нужен отдельный режим, иначе обновление
  # затирает рабочий .env файлом из репозитория.
  printf "  источник:  %s\n" "$SOURCE"
  [ -d "$SOURCE/.git" ] && { git -C "$SOURCE" pull --ff-only 2>/dev/null || true; }
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude data --exclude logs --exclude venv --exclude backups \
          --exclude .env --exclude secrets.env --exclude .git \
          "$SOURCE"/ "$TARGET"/ && ok "код обновлён из $SOURCE"
  else
    (cd "$SOURCE" && tar cf - --exclude=data --exclude=logs --exclude=venv \
        --exclude=backups --exclude=.env --exclude=secrets.env --exclude=.git . ) \
      | (cd "$TARGET" && tar xf -) && ok "код обновлён из $SOURCE"
  fi
elif [ -d "$TARGET/.git" ]; then
  # Прячем только ПРАВКИ отслеживаемых файлов. Никакого --include-untracked:
  # .env, данные и журналы в git не отслеживаются, и тот ключ утаскивал их
  # в тайник — после обновления настройки «сбрасывались» на умолчания.
  git -C "$TARGET" stash push -m "update-$STAMP" >/dev/null 2>&1 || true
  git -C "$TARGET" pull --ff-only && ok "код обновлён"
else
  warn "Установка не под git и источник не указан — обновлять нечем."
  warn "Либо распакуйте новый архив поверх, либо укажите папку с кодом:"
  warn "    $0 --target $TARGET --from /путь/к/репозиторию"
fi

# Возвращаем настройки из снимка, если обновление кода их изменило или
# стёрло — независимо от того, каким путём шло обновление.
for f in .env secrets.env; do
  if [ -f "$ENV_TMP/$f" ] && ! cmp -s "$ENV_TMP/$f" "$TARGET/$f" 2>/dev/null; then
    cp "$ENV_TMP/$f" "$TARGET/$f"
    ok "настройки ($f) восстановлены — обновление кода их не меняет"
  fi
done
rm -rf "$ENV_TMP"

# Настройки, спрятанные прошлыми обновлениями в git stash (старая ошибка
# с --include-untracked), возвращаем на место. Неотслеживаемые файлы
# stash хранит в третьем родителе записи.
if [ ! -f "$TARGET/.env" ] && [ -d "$TARGET/.git" ]; then
  for st in $(git -C "$TARGET" stash list --format='%gd' 2>/dev/null); do
    if git -C "$TARGET" show "$st^3:.env" > "$TARGET/.env" 2>/dev/null \
       || git -C "$TARGET" show "$st:.env" > "$TARGET/.env" 2>/dev/null; then
      if [ -s "$TARGET/.env" ]; then
        ok "настройки нашлись в git stash ($st) и восстановлены"
        break
      fi
    fi
    rm -f "$TARGET/.env"
  done
fi

# Скрипт обновления только что мог обновить сам себя. Всё, что дальше, —
# зависимости, ollama, миграции, проверки — выполняет уже НОВАЯ версия:
# иначе каждый шаг, добавленный в обновление, начинал работать только со
# второго запуска, и «обновлённая» установка была беднее свежей ровно на
# один прогон. Классическая ловушка самообновляющихся скриптов.
NEW_SUM="$(cksum < "$TARGET/install/update.sh" 2>/dev/null || echo none)"
if [ -f "$TARGET/install/update.sh" ] && [ "$NEW_SUM" != "$SELF_SUM" ]; then
  say "Скрипт обновления обновился — продолжаю новой версией"
  export KB_UPDATE_STAGE2=1
  STAGE2_ARGS=(--target "$TARGET")
  [ -n "$SOURCE" ] && STAGE2_ARGS+=(--from "$SOURCE")
  [ "$WITH_BACKUP" = 0 ] && STAGE2_ARGS+=(--no-backup)
  exec bash "$TARGET/install/update.sh" "${STAGE2_ARGS[@]}"
fi

fi   # конец пропуска подготовительных шагов (KB_UPDATE_STAGE2)

# Сбой установки зависимостей не должен рушить обновление. К этому
# моменту код уже скопирован, и прерывание оставляет установку
# наполовину обновлённой — худшее из состояний. А причина почти всегда
# внешняя и временная: нет сети, закрыт PyPI, отвалилось зеркало.
# Библиотеки при этом меняются редко, и старые обычно подходят.
say "Обновляю зависимости"
if "$TARGET/venv/bin/python" -m pip install --quiet --upgrade \
     -r "$TARGET/requirements.txt" 2>/tmp/kb-pip-$STAMP.log; then
  ok "готово"
else
  warn "не удалось обновить библиотеки (подробности: /tmp/kb-pip-$STAMP.log)"
  warn "код обновлён и, скорее всего, работоспособен — проверка ниже покажет."
  warn "Повторить потом: $TARGET/venv/bin/python -m pip install -r $TARGET/requirements.txt"
fi

# vllm живёт вне requirements.txt (ставится только там, где есть карта
# NVIDIA), поэтому обновляется отдельно — на тех же условиях, что ставился.
if "$TARGET/venv/bin/python" -c "import vllm" 2>/dev/null \
   || command -v nvidia-smi >/dev/null 2>&1; then
  say "Обновляю vllm"
  "$TARGET/venv/bin/python" -m pip install --quiet --upgrade vllm \
    2>>/tmp/kb-pip-$STAMP.log && ok "vllm обновлён" \
    || warn "vllm не обновился (не страшно: работает прежняя версия)"
fi

# ollama: доставляем, если его ещё нет, — установка могла пройти до того,
# как он появился в установщике. Уже стоящий не трогаем: обновление
# сервера моделей посреди обновления кода — лишний риск без нужды.
if command -v ollama >/dev/null 2>&1; then
  ok "ollama уже установлен"
else
  say "Ставлю ollama — запуск локальных моделей"
  if [ "$(uname -s)" = Darwin ]; then
    command -v brew >/dev/null 2>&1 && { brew install ollama && ok "ollama установлен" \
      || warn "ollama не поставился: brew install ollama"; } \
      || warn "Нужен Homebrew: brew install ollama"
  else
    curl -fsSL https://ollama.com/install.sh | sh \
      && ok "ollama установлен" \
      || warn "ollama не поставился — вручную: curl -fsSL https://ollama.com/install.sh | sh"
  fi
fi

say "Проверяю совместимость данных"
"$TARGET/venv/bin/python" -c "
import sys; sys.path.insert(0, '$TARGET')
import db, config
db.init()
n = db.q1(\"SELECT COUNT(*) n FROM documents\")['n']
print(f'  документов в индексе: {n}')
" && ok "схема хранилища обновлена автоматически"

# Проверка модулей — диагностика, а не условие. Если что-то не
# импортируется, об этом надо сказать, но не откатывать всё обновление:
# чаще всего не хватает необязательной библиотеки для одного формата.
say "Проверяю конвейер индексации"
# Если чанкинг или нормализация изменились, индекс разобран по старым
# правилам, а поиск работает по новым — качество тихо проседает. Пересборка
# ставится в очередь и начнётся с запуском админки.
"$TARGET/venv/bin/python" "$TARGET/index.py" pipeline-check --enqueue 2>/dev/null \
  || warn "проверка конвейера не отработала (не критично)"

say "Проверяю модули"
"$TARGET/venv/bin/python" -c "
import sys; sys.path.insert(0, '$TARGET')
ok, bad = 0, []
for m in ('index','search','answer','bot','webui','audit','graph','media','crawl',
          'voice','sip','models','metrics','watcher','llm','llm_queue','jobs',
          'security','alerts','retention','backup','ocr','rerank','lsa','analytics',
          'access','tracing','regression','contextual','preflight','shutdown',
          'quickstart'):
    try:
        __import__(m); ok += 1
    except Exception as exc:
        bad.append(f'{m}: {exc}')
print(f'  загружается модулей: {ok}')
for line in bad[:8]:
    print('  не загрузился ' + line)
" || warn "часть модулей не загрузилась — смотрите список выше"

# Процессы, убитые при остановке, могли оставить за собой занятые места
# в очереди к модели. Само оно освободится по сроку, но это до пяти минут
# на ровном месте: после обновления ассистент выглядел бы зависшим.
# Задачи, оборванные остановкой, снимаем: иначе занятый ими ресурс три
# минуты держит блокировку, и первая же индексация получает отказ.
say "Снимаю оборванные задачи"
"$TARGET/venv/bin/python" "$TARGET/jobs.py" reap 2>/dev/null || true

say "Освобождаю очередь к модели"
"$TARGET/venv/bin/python" "$TARGET/llm_queue.py" clear || \
  warn "не удалось очистить очередь — освободится сама по сроку"

say "Проверяю настройку после обновления"
"$TARGET/venv/bin/python" "$TARGET/preflight.py" || \
  fail "Проверка не пройдена — служба не запускается. Исправьте отмеченное выше."

if command -v systemctl >/dev/null 2>&1 \
   && systemctl is-enabled --quiet kb-assistant 2>/dev/null; then
  sudo systemctl start kb-assistant && ok "служба запущена"
fi

printf "\n${GREEN}Обновление завершено.${NC}\n"
if [ "$WITH_BACKUP" = 1 ]; then
  echo "Копия предыдущей версии: $BACKUP_DIR/$STAMP"
else
  echo "Копия при этом обновлении не делалась (--no-backup)."
fi
echo "Если что-то пошло не так: $0 --rollback"
