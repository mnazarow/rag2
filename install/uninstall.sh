#!/usr/bin/env bash
# Удаление ассистента.
#
#   ./uninstall.sh            — удалить программу, данные сохранить
#   ./uninstall.sh --all      — удалить вместе с индексом и журналами
#   ./uninstall.sh --keep-env — сохранить файл настроек отдельно
#
# Папка базы знаний (KB_ROOT) НЕ ТРОГАЕТСЯ никогда.

set -Eeuo pipefail
TARGET="$(cd "$(dirname "$0")/.." && pwd)"
GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; BLUE='\033[36m'; NC='\033[0m'
say(){ printf "${BLUE}==>${NC} %s\n" "$*"; }
ok(){ printf "  ${GREEN}✓${NC} %s\n" "$*"; }
warn(){ printf "  ${YELLOW}!${NC} %s\n" "$*"; }

REMOVE_DATA=0; KEEP_ENV=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all) REMOVE_DATA=1 ;;
    --keep-env) KEEP_ENV=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac; shift
done

KB_ROOT="$(grep -E '^KB_ROOT=' "$TARGET/.env" 2>/dev/null | cut -d= -f2- || echo '')"
printf "\n${YELLOW}Удаление ассистента${NC}\n"
echo "Папка программы: $TARGET"
[ -n "$KB_ROOT" ] && echo "Папка базы знаний: $KB_ROOT — НЕ БУДЕТ ТРОНУТА"
[ "$REMOVE_DATA" = 1 ] && printf "${RED}Индекс и журналы будут удалены.${NC}\n" \
                       || echo "Индекс и журналы сохранятся."
printf "\nПродолжить? Введите: удалить\n> "
read -r ANSWER
[ "$ANSWER" != "удалить" ] && { echo "Отменено."; exit 0; }

say "Останавливаю процессы"
pkill -f "$TARGET/webui.py" 2>/dev/null && ok "админка" || true
pkill -f "$TARGET/bot.py" 2>/dev/null && ok "бот" || true
pkill -f "$TARGET/watcher.py" 2>/dev/null && ok "слежение" || true
"$TARGET/venv/bin/python" -c "
import sys; sys.path.insert(0,'$TARGET')
import models; models.stop()" 2>/dev/null && ok "сервер модели" || true

say "Убираю автозапуск"
if command -v systemctl >/dev/null 2>&1 && [ -f /etc/systemd/system/kb-assistant.service ]; then
  sudo systemctl disable --now kb-assistant 2>/dev/null || true
  sudo rm -f /etc/systemd/system/kb-assistant.service
  sudo systemctl daemon-reload
  ok "служба systemd удалена"
fi
PLIST="$HOME/Library/LaunchAgents/ru.company.kb-assistant.plist"
if [ -f "$PLIST" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true; rm -f "$PLIST"; ok "автозапуск macOS удалён"
fi

if command -v docker >/dev/null 2>&1 && [ -f "$TARGET/docker-compose.yml" ]; then
  say "Убираю контейнеры"
  (cd "$TARGET" && docker compose down 2>/dev/null) && ok "контейнеры остановлены" || true
fi

if [ "$KEEP_ENV" = 1 ] && [ -f "$TARGET/.env" ]; then
  cp "$TARGET/.env" "$HOME/kb-assistant.env.saved"
  ok "настройки сохранены: $HOME/kb-assistant.env.saved"
fi

say "Удаляю файлы"
if [ "$REMOVE_DATA" = 1 ]; then
  rm -rf "$TARGET"
  ok "удалено полностью"
else
  find "$TARGET" -maxdepth 1 -mindepth 1 \
       ! -name data ! -name logs ! -name .env ! -name backups -exec rm -rf {} +
  ok "программа удалена, сохранены: data, logs, .env, backups"
fi

printf "\n${GREEN}Готово.${NC} Папка базы знаний не изменялась.\n"
