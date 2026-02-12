#!/bin/bash
# 删除 ~/.cache/huggingface 下所有 root 属主的文件和目录，便于用当前用户重新下载。
# 需要 sudo 执行: sudo bash scripts/remove_root_owned_hf_cache.sh

set -e
CACHE="/home/hku/.cache/huggingface"
if [[ ! -d "$CACHE" ]]; then
  echo "No such dir: $CACHE"
  exit 0
fi
echo "Removing root-owned files under $CACHE ..."
find "$CACHE" -user root -delete
echo "Done. You can re-run your download now."
echo "If download still fails, consider: sudo rm -rf $CACHE  (then re-download all models)."
