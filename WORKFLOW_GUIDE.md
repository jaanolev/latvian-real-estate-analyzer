# 🚀 Webapp Update Workflow Guide

## Quick Reference

### Every Time You Want to Update Your Webapp:

1. **Edit your code** (app.py or other files)
2. **Test locally** (optional): `streamlit run app.py`
3. **Commit changes**:
   ```powershell
   git add .
   git commit -m "Describe your changes here"
   git push origin main
   ```
4. **Wait 2-3 minutes** - Streamlit Cloud auto-deploys!

---

## Example Workflow

### Scenario: You want to add a new feature

```powershell
# 1. Make your changes in VS Code/Cursor
# (edit app.py, add new functions, etc.)

# 2. Test locally (optional)
streamlit run app.py
# Check that everything works at http://localhost:8501

# 3. If happy with changes, commit them
git add .
git commit -m "Added property age filter and improved charts"

# 4. Push to GitHub
git push origin main

# 5. Check Streamlit Cloud
# Go to: https://share.streamlit.io/
# Watch the logs - your app will redeploy automatically!
```

---

## Important Notes

### ✅ What Triggers Auto-Deployment?
- ANY push to GitHub's `main` branch
- Streamlit Cloud watches your repo 24/7

### ⏱️ How Long Does Redeployment Take?
- **First deployment**: 5-20 minutes (installing everything)
- **Updates**: 1-3 minutes (much faster, packages cached!)

### 🔗 Does the URL Change?
- **NO!** Your URL stays the same forever
- Example: `https://latvian-real-estate-analyzer-xxx.streamlit.app`
- Users can bookmark it and it never changes

### 💾 What About Data Files?
- Your CSV files are in the repo, so they deploy too
- If you update the CSV files, just push them like any other file
- **Warning**: GitHub has a 100MB file size limit

### 🛑 How to Stop Auto-Deployment?
- Just don't push to GitHub! Work locally all you want
- Only pushes to GitHub trigger deployment

---

## Common Commands Cheat Sheet

```powershell
# See what files changed
git status

# See what you modified
git diff

# Stage all changes
git add .

# Stage specific file
git add app.py

# Commit with message
git commit -m "Your message here"

# Push to GitHub (triggers deployment)
git push origin main

# Pull latest changes (if working on multiple computers)
git pull origin main

# View commit history
git log --oneline
```

---

## Troubleshooting

### My changes aren't showing up!
1. Did you push to GitHub? Check: `git status`
2. Did Streamlit Cloud redeploy? Check the logs
3. Clear browser cache (Ctrl+F5)

### Deployment failed!
1. Check Streamlit Cloud logs for error messages
2. Test locally first: `streamlit run app.py`
3. Common issues:
   - Syntax errors in Python code
   - Missing packages in requirements.txt
   - File path issues (use relative paths!)

### I broke everything! How to roll back?
```powershell
# See recent commits
git log --oneline

# Revert to previous commit
git revert HEAD

# Or hard reset (careful!)
git reset --hard HEAD~1
git push origin main --force
```

---

## Pro Tips

### 🎯 Tip 1: Test Locally First!
Always run `streamlit run app.py` locally before pushing. Catch errors early!

### 🎯 Tip 2: Use Good Commit Messages
- ❌ Bad: "updated stuff"
- ✅ Good: "Fixed date filter bug and improved performance"

### 🎯 Tip 3: Small, Frequent Updates
Push small changes often rather than huge updates. Easier to debug!

### 🎯 Tip 4: Branch for Big Changes (Advanced)
For major features, create a branch:
```powershell
git checkout -b new-feature
# Make changes
git add .
git commit -m "Work in progress"
# When ready, merge to main
git checkout main
git merge new-feature
git push origin main
```

---

## Your Webapp URLs

### Local (Testing):
- `http://localhost:8501`
- Only accessible on your computer
- Stops when you close terminal

### Production (Live):
- `https://latvian-real-estate-analyzer-6qspebyzqhkhns.streamlit.app/`
- Accessible worldwide 24/7
- Updates automatically when you push to GitHub

---

## Questions?

- **How often can I update?** As many times as you want!
- **Will users see downtime?** Brief (10-30 seconds) during redeployment
- **Can I have multiple versions?** Yes! Deploy different branches
- **Is it really free?** Yes, for public apps!


