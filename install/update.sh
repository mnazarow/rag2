#!/usr/bin/env bash
# Обновление ассистента. Индекс, настройки и накопленные данные сохраняются.
#
#   ./update.sh                — обновить код и зависимости
#   ./update.sh --backup-only  — только сделать резервную копию
#   ./update.sh --rollback     — вернуться к предыдущей версии

set -Eeuo pipefail
TARGET="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$TARGET/backups"
GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; BLUE='\033[36m'; NC='\033[0m'
say(){ printf "${BLUE}==>${NC} %s\n" "$*"; }
ok(){ printf "  ${GREEN}✓${NC} %s\n" "$*"; }
warn(){ printf "  ${YELLOW}!${NC} %s\n" "$*"; }
fail(){ printf "  ${RED}✗${NC} %s\n" "$*" >&2; }

trap 'fail "Обновление прервано. Данные не тронуты, откат: $0 --rollback"; exit 1' ERR

MODE=update
[ "${1:-}" = "--backup-only" ] && MODE=backup
[ "${1:-}" = "--rollback" ] && MODE=rollback

STAMP="$(date +%Y%m%d-%H%M%S)"
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
    ok "Откат выполнен. Перезапустите сервисы."
    exit 0 ;;
esac

backup >/dev/null

say "Останавливаю службы"
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet kb-assistant; then
  sudo systemctl stop kb-assistant && ok "systemd остановлен"
fi
pkill -f "$TARGET/webui.py" 2>/dev/null && ok "админка остановлена" || true
pkill -f "$TARGET/bot.py" 2>/dev/null && ok "бот остановлен" || true

if [ -d "$TARGET/.git" ]; then
  say "Обновляю код из репозитория"
  git -C "$TARGET" stash push --include-untracked -m "update-$STAMP" >/dev/null 2>&1 || true
  git -C "$TARGET" pull --ff-only && ok "код обновлён"
else
  warn "Это не git-репозиторий: распакуйте новый архив поверх и запустите снова"
fi

say "Обновляю зависимости"
"$TARGET/venv/bin/python" -m pip install --quiet --upgrade -r "$TARGET/requirements.txt"
ok "готово"

say "Проверяю совместимость данных"
"$TARGET/venv/bin/python" -c "
import sys; sys.path.insert(0, '$TARGET')
import db, config
db.init()
n = db.q1(\"SELECT COUNT(*) n FROM documents\")['n']
print(f'  документов в индексе: {n}')
" && ok "схема хранилища обновлена автоматически"

say "Проверяю модули"
"$TARGET/venv/bin/python" -c "
import sys; sys.path.insert(0, '$TARGET')
for m in ('index','search','answer','bot','webui','audit','graph','media','crawl','voice','sip','models','metrics','watcher','llm','llm_queue','jobs','security','alerts','retention','backup','ocr','rerank','lsa','analytics','access','tracing','regression','contextual'):
    __import__(m)
print('  все модули загружаются')
"

# Процессы, убитые при остановке, могли оставить за собой занятые места
# в очереди к модели. Само оно освободится по сроку, но это до пяти минут
# на ровном месте: после обновления ассистент выглядел бы зависшим.
say "Освобождаю очередь к модели"
"$TARGET/venv/bin/python" "$TARGET/llm_queue.py" clear || \
  warn "не удалось очистить очередь — освободится сама по сроку"

if command -v systemctl >/dev/null 2>&1 && systemctl is-enabled --quiet kb-assistant 2>/dev/null; then
  sudo systemctl start kb-assistant && ok "служба запущена"
fi

printf "\n${GREEN}Обновление завершено.${NC}\n"
echo "Копия предыдущей версии: $BACKUP_DIR/$STAMP"
echo "Если что-то пошло не так: $0 --rollback"
