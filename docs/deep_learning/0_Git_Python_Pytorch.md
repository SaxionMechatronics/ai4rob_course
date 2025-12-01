# **Introduction to Git & Version Control Systems**

## **1. Introduction to Git**
Git is a free, open-source, **distributed Version Control System (VCS)** designed to efficiently manage projects of all sizes—from small scripts to large-scale software systems.  
It enables developers to track changes, collaborate effectively, and maintain a complete history of project evolution.

---

## **2. What is a Version Control System (VCS)?**
A Version Control System is a software tool that helps developers:

- **Manage and track changes** to their code over time  
- **Collaborate** without overwriting each other’s work  
- **Preserve a complete history** of project modifications

As developers, we continuously create, save, modify, and refine our work. A VCS ensures these iterative changes are captured, organized, and recoverable.

---

## **3. Why We Need a Version Control System**

### **a. Track and Manage Code Changes**
A VCS records:
- What changed  
- Who made the change  
- When it was made  
- Why the change was necessary (via commit messages)

This clarity is essential for debugging, audits, and understanding long-term development decisions.

### **b. Enable Collaboration**
Multiple developers can work on the same project without the risk of:
- Overwriting each other’s code  
- Losing progress  
- Introducing conflicting changes

### **c. Roll Back When Needed**
Mistakes happen.  
A VCS allows developers to:
- Restore previous versions  
- Recover deleted or broken code  
- Experiment freely, knowing they can revert at any time

### **d. Maintain a Complete Project History**
Every change becomes part of a structured timeline of development.  
This allows teams to:
- Analyze how the codebase evolved  
- Compare versions  
- Understand feature progression

### **In summary, a VCS helps answer:**
- **When** was a change made?  
- **Why** was it made (commit message)?  
- **What** exactly changed?  
- **Who** changed it?  
- **Can we revert to an earlier version?**  
- **Can we branch, experiment, and merge safely?**

---

## **4. Essential Git Commands**

### **4.1 `git init`**
Initializes a new Git repository in the current directory.  
This directory becomes your **working directory** where Git tracks changes.

### **4.2 `git add`**
Stages files for commit.  
The staging area acts as a **temporary holding space** before you record changes.

### **4.3 `git commit`**
Creates a snapshot of the project at a specific moment.  
Each commit includes:
- The staged changes  
- A descriptive message explaining *what* and *why*

### **4.4 `git push`**
Uploads your local commits to a remote repository so **others can access your updates**.

### **4.5 `git pull`**
Downloads and integrates changes from a remote repository so **you can stay up to date** with the latest updates from collaborators.

### **4.6 `git checkout`**
Switches between branches or different versions of the project.  
This allows developers to:
- Try new features  
- Test ideas  
- Work safely without affecting the main branch

---
## **5 Git Cheat Sheet for quick reference**

### **5.1 Create a Repository**

| Action | Command |
|--------|---------|
| Initialize a new local repository | `git init [project name]` |
| Clone/download an existing repository | `git clone my_url` |


### **5.2 Observe Your Repository**

| Action | Command |
|--------|---------|
| List new/modified files not yet committed | `git status` |
| Show changes in unstaged files | `git diff` |
| Show changes in staged files | `git diff --cached` |
| Show all staged & unstaged changes | `git diff HEAD` |
| Show differences between two commits | `git diff commit1 commit2` |
| Show who changed each line of a file | `git blame [file]` |
| Show file changes for a commit/file | `git show [commit]:[file]` |
| Show full commit history | `git log` |
| Show commit history for file/dir with diffs | `git log -p [file/directory]` |


### **5.3 Working with Branches**

| Action | Command |
|--------|---------|
| List all local branches | `git branch` |
| List all branches (local + remote) | `git branch -av` |
| Switch to a branch | `git checkout my_branch` |
| Create a new branch | `git branch new_branch` |
| Delete a branch | `git branch -d my_branch` |
| Merge branch_a into branch_b | `git checkout branch_b` → `git merge branch_a` |
| Tag the current commit | `git tag my_tag` |


### **5.4 Make a Change**

| Action | Command |
|--------|---------|
| Stage a file | `git add [file]` |
| Stage all modified files | `git add .` |
| Commit staged changes | `git commit -m "message"` |
| Commit all tracked files | `git commit -am "message"` |
| Unstage a file (keep changes) | `git reset [file]` |
| Revert everything to last commit | `git reset --hard` |


### **5.5. Synchronize (Remote Operations)**

| Action | Command |
|--------|---------|
| Fetch latest changes (no merge) | `git fetch` |
| Fetch + merge latest changes | `git pull` |
| Fetch + rebase (clean history) | `git pull --rebase` |
| Push local commits to remote | `git push` |


### **5.6 Help**

| Action | Command |
|--------|---------|
| Show help for a Git command | `git command --help` |
| Official GitHub training | https://training.github.com/ |

---



## **End of Module Summary**
By understanding the purpose of version control and mastering basic Git commands, developers can work more confidently, collaborate more effectively, and maintain an organized, reliable codebase.

---
---

## Link to the Google Colab for Python 101

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/Python_101.ipynb)

---
## Link to the Google Colab for Pytorch 101

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaxionMechatronics/ai4rob_course/blob/main/docs/notebooks/deep_learning/Pytorch_101.ipynb)
---

[← Back to Index](index.md){ .md-button }
[Start with Introduction to Deep Learning →](12_Introduction.md){ .md-button .md-button--primary }