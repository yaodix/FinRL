二、日常同步 + 开发（反复使用）
每次想跟进原仓库并继续自己开发，执行：

cd /你的本地/FinRL
git checkout main
git fetch upstream
git rebase upstream/main
git push origin main
git checkout -B feature/my-work
git rebase main
git push -u origin feature/my-work

你在 feature/my-work 上改完后：

git add .
git commit -m "feat: your changes"
git push

如果 feature 分支做过 rebase 后推送被拒绝，用：

git push --force-with-lease

三、推荐固定工作流（避免冲突）

1. main 只做同步，不直接写代码。
2. 所有新增文件和实验都放 feature 分支。
3. 提 PR 时用 feature 分支，不要用 main。
