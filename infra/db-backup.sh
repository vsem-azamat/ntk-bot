#!/bin/sh
# Daily off-site logical backup of the ntk-bot database, kept on the bot host
# (the database server has its own backups; this copy survives losing it).
set -eu

BACKUP_DIR=/backups
KEEP_DAYS="${BACKUP_KEEP_DAYS:-30}"
RUN_AT_UTC="${BACKUP_AT_UTC:-03:40}"

backup() {
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  tmp="$BACKUP_DIR/.ntk-$stamp.dump.tmp"
  if pg_dump --format=custom --no-owner --file="$tmp" "$DATABASE_URL"; then
    chmod 600 "$tmp"
    mv "$tmp" "$BACKUP_DIR/ntk-$stamp.dump"
    echo "[db-backup] wrote ntk-$stamp.dump ($(du -h "$BACKUP_DIR/ntk-$stamp.dump" | cut -f1))"
    find "$BACKUP_DIR" -name 'ntk-*.dump' -mtime +"$KEEP_DAYS" -delete
  else
    rm -f "$tmp"
    echo "[db-backup] pg_dump failed" >&2
  fi
}

seconds_until_next_run() {
  now="$(date -u +%s)"
  next="$(date -u -d "$(date -u +%Y-%m-%d) $RUN_AT_UTC" +%s)"
  [ "$next" -le "$now" ] && next=$((next + 86400))
  echo $((next - now))
}

mkdir -p "$BACKUP_DIR"
# A dump right after (re)deploy proves credentials and TLS work.
backup
while true; do
  sleep "$(seconds_until_next_run)"
  backup
done
