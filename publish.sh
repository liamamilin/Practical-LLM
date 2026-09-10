#!/bin/bash
# 发布脚本：提交全书并推送到 GitHub（触发 Pages 部署）
# 用法: ./publish.sh "可选的提交说明"
set -e
cd "$(dirname "$0")"

REMOTE="git@github.com:liamamilin/Practical-LLM.git"
BRANCH="main"

# 1. 配置 remote（幂等）
if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "$REMOTE"
    echo "→ 已添加 remote: $REMOTE"
fi

# 2. 暂存所有变更（.gitignore 会自动排除大文件）
git add -A

# 3. 没有变更则直接退出
if git diff --cached --quiet; then
    echo "✓ 没有需要提交的变更"
    exit 0
fi

# 4. 提交
MSG="${1:-更新书稿 $(date +%F)}"
git commit -m "$MSG"
echo "→ 已提交: $MSG"

# 5. 推送（首次推送带 -u 建立跟踪）
if git rev-parse --verify origin/"$BRANCH" >/dev/null 2>&1; then
    git push
else
    git push -u origin "$BRANCH"
fi

echo "✓ 已推送。GitHub Actions 正在构建，稍后见: https://liamamilin.github.io/Practical-LLM/"
