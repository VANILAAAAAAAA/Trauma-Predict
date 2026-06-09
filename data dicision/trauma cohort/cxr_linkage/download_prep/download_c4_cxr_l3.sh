#!/usr/bin/env bash
set -euo pipefail

LIST="/home/vanila/code/EHR-Predict/data dicision/trauma cohort/cxr_linkage/download_prep/c4_cxr_download_list.txt"
CRED="/home/vanila/code/Physionet Oath.txt"
OUT="/mnt/d/Data/mimic-cxr-jpg-c4"
LOG="/home/vanila/code/EHR-Predict/data dicision/trauma cohort/cxr_linkage/download_prep/c4_cxr_download.log"

if [[ ! -f "$LIST" ]]; then
  echo "Missing download list: $LIST" >&2
  exit 1
fi
if [[ ! -f "$CRED" ]]; then
  echo "Missing PhysioNet credential file: $CRED" >&2
  exit 1
fi

USER_NAME=$(grep -i '^username:' "$CRED" | head -1 | cut -d':' -f2- | tr -d '\r' | xargs)
PASSWORD=$(grep -i '^password:' "$CRED" | head -1 | cut -d':' -f2- | tr -d '\r' | xargs)

if [[ -z "$USER_NAME" || -z "$PASSWORD" ]]; then
  echo "Credential file must contain username: and password:" >&2
  exit 1
fi

COUNT=$(wc -l < "$LIST" | tr -d ' ')
AVAIL_GB=$(df -BG /mnt/d/Data | awk 'NR==2 {gsub("G", "", $4); print $4}')

cat <<EOF
C4 CXR JPG download
  list: $LIST
  files: $COUNT
  output: $OUT
  available /mnt/d/Data: ${AVAIL_GB} GB
  log: $LOG
EOF

mkdir -p "$OUT"

# Direct file URLs; no recursive crawl. -c resumes partial files.
wget -c \
  --user "$USER_NAME" \
  --password "$PASSWORD" \
  -i "$LIST" \
  -P "$OUT" \
  -o "$LOG"

echo "Download finished. See log: $LOG"
