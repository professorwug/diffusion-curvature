#!/bin/bash

# Git Worktree script
# Creates a new git worktree with a branch and opens it in VS Code Insider

# Function to display help
show_help() {
    echo "Usage: $(basename $0) [OPTIONS] [branch-suffix]"
    echo ""
    echo "Creates a new git worktree with a branch and opens it with ccv."
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help      Show this help message"
    echo "  -m, --mcp       Select and activate an MCP configuration template"
    echo "  --mcp PURPOSE   Activate a specific MCP template (e.g., --mcp dwh)"
    echo ""
    echo "ARGUMENTS:"
    echo "  branch-suffix   Optional branch name suffix (default: patch-{timestamp})"
    echo ""
    echo "Available MCP templates:"
    
    # Dynamically list MCP templates from the directory
    local templates_dir="$MAIN_WORKTREE/mcp-json-templates"
    if [ -d "$templates_dir" ]; then
        for template in "$templates_dir"/.mcp.*.json; do
            if [ -f "$template" ]; then
                local template_name=$(basename "$template" | sed 's/^\.mcp\.//' | sed 's/\.json$//')
                echo -n "  - $template_name"
                
                # Extract description from README.md if available
                local readme="$templates_dir/README.md"
                if [ -f "$readme" ]; then
                    local desc=$(grep -A1 "\.mcp\.$template_name\.json" "$readme" | tail -n1 | sed 's/^[[:space:]]*//')
                    if [ -n "$desc" ] && [ "$desc" != "$(grep "\.mcp\.$template_name\.json" "$readme")" ]; then
                        echo ": $desc"
                    else
                        echo ""
                    fi
                else
                    echo ""
                fi
            fi
        done
    fi
    echo ""
    echo "Examples:"
    echo "  $(basename $0)                    # Create worktree with auto-generated branch name"
    echo "  $(basename $0) fix-bug            # Create worktree with branch wug/fix-bug"
    echo "  $(basename $0) --mcp dwh fix-bug  # Create worktree and activate dwh MCP template"
    echo "  $(basename $0) -m                 # Interactive MCP template selection"
}

# Function to select MCP template interactively
select_mcp_template() {
    local templates_dir="$1"
    local templates=()
    local descriptions=()
    
    # Collect available templates
    for template in "$templates_dir"/.mcp.*.json; do
        if [ -f "$template" ]; then
            local template_name=$(basename "$template" | sed 's/^\.mcp\.//' | sed 's/\.json$//')
            templates+=("$template_name")
            
            # Get description from README if available
            local readme="$templates_dir/README.md"
            local desc=""
            if [ -f "$readme" ]; then
                desc=$(grep -A1 "\.mcp\.$template_name\.json" "$readme" | tail -n1 | sed 's/^[[:space:]]*//')
                if [ -z "$desc" ] || [ "$desc" = "$(grep "\.mcp\.$template_name\.json" "$readme")" ]; then
                    desc="No description available"
                fi
            else
                desc="No description available"
            fi
            descriptions+=("$desc")
        fi
    done
    
    if [ ${#templates[@]} -eq 0 ]; then
        echo "No MCP templates found in $templates_dir"
        return 1
    fi
    
    # Always use numbered selection for simplicity and reliability
    echo "Available MCP templates:" >&2
    echo "" >&2
    for i in "${!templates[@]}"; do
        echo "  $((i+1)). ${templates[$i]}" >&2
        echo "     ${descriptions[$i]}" >&2
        echo "" >&2
    done
    
    read -p "Select template (1-${#templates[@]}): " selection
    
    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#templates[@]} ]; then
        echo "${templates[$((selection-1))]}"
    else
        echo "Invalid selection" >&2
        return 1
    fi
}

# Function to activate MCP template
activate_mcp_template() {
    local template_name="$1"
    local worktree_path="$2"
    local templates_dir="$MAIN_WORKTREE/mcp-json-templates"
    
    local template_file="$templates_dir/.mcp.$template_name.json"
    if [ ! -f "$template_file" ]; then
        echo "Error: MCP template '$template_name' not found"
        return 1
    fi
    
    echo "Activating MCP template: $template_name"
    cp "$template_file" "$worktree_path/.mcp.json"
    echo "  ✓ Copied .mcp.$template_name.json to .mcp.json"
}

# Function to detect the default branch (main or master)
get_default_branch() {
    # First try to get it from git config
    local default_branch=$(git config --get init.defaultBranch 2>/dev/null)
    
    if [ -z "$default_branch" ]; then
        # Check if main branch exists
        if git show-ref --verify --quiet refs/heads/main; then
            default_branch="main"
        elif git show-ref --verify --quiet refs/heads/master; then
            default_branch="master"
        else
            # Try to get from remote
            default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')
            if [ -z "$default_branch" ]; then
                # Default to main if nothing else works
                default_branch="main"
            fi
        fi
    fi
    
    echo "$default_branch"
}

# Get the git repository root (works in both main repo and worktrees)
CURRENT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$CURRENT_ROOT" ]; then
  echo "Error: Not in a git repository"
  exit 1
fi

# Find the main worktree (the actual git directory)
GIT_COMMON_DIR="$(git rev-parse --git-common-dir 2>/dev/null)"
if [ "$GIT_COMMON_DIR" = ".git" ] || [ "$GIT_COMMON_DIR" = "$(pwd)/.git" ]; then
  # We're in the main repository
  MAIN_WORKTREE="$CURRENT_ROOT"
else
  # We're in a worktree, find the main worktree
  MAIN_WORKTREE="$(dirname "$GIT_COMMON_DIR")"
fi

# Parse command line arguments
MCP_TEMPLATE=""
BRANCH_SUFFIX=""
INTERACTIVE_MCP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m)
            INTERACTIVE_MCP=true
            shift
            ;;
        --mcp)
            if [ -n "$2" ] && [[ ! "$2" =~ ^- ]]; then
                MCP_TEMPLATE="$2"
                shift 2
            else
                echo "Error: --mcp requires a template name"
                exit 1
            fi
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            BRANCH_SUFFIX="$1"
            shift
            ;;
    esac
done

# Handle interactive MCP selection
if [ "$INTERACTIVE_MCP" = true ]; then
    MCP_TEMPLATE=$(select_mcp_template "$MAIN_WORKTREE/mcp-json-templates")
    if [ $? -ne 0 ] || [ -z "$MCP_TEMPLATE" ]; then
        echo "MCP template selection cancelled or failed"
        exit 1
    fi
fi

# If no argument provided, use patch-{datetime}
if [ -z "$BRANCH_SUFFIX" ]; then
  BRANCH_SUFFIX="patch-$(date +%s)"
fi

# Create the full branch name with github username prefix
BRANCH_NAME="wug/$BRANCH_SUFFIX"

# Create worktree directory name (replace / with -)
WORKTREE_DIR=$(echo "$BRANCH_NAME" | tr '/' '-')

# Handle uncommitted changes
echo "Checking for uncommitted changes..."
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo ""
  echo "You have uncommitted changes:"
  echo ""
  git status --short
  echo ""
  echo "How would you like to handle these changes?"
  echo "1) Commit them now (interactive)"
  echo "2) Stash them (will be reapplied after worktree creation)"
  echo "3) Cancel worktree creation"
  echo ""
  read -p "Choose an option (1-3): " choice
  
  case $choice in
    1)
      echo ""
      echo "Staging all changes..."
      git add -A
      echo ""
      echo "Please enter a commit message:"
      read -p "> " commit_message
      if [ -z "$commit_message" ]; then
        commit_message="WIP: Changes before creating worktree $BRANCH_NAME"
      fi
      git commit -m "$commit_message"
      if [ $? -eq 0 ]; then
        echo "Changes committed successfully!"
        STASHED=false
      else
        echo "Commit failed. Exiting."
        exit 1
      fi
      ;;
    2)
      echo "Stashing uncommitted changes..."
      git stash push -m "Auto-stash before creating worktree $BRANCH_NAME"
      STASHED=true
      ;;
    3)
      echo "Worktree creation cancelled."
      exit 0
      ;;
    *)
      echo "Invalid option. Cancelling."
      exit 1
      ;;
  esac
else
  STASHED=false
fi

# Get the default branch
DEFAULT_BRANCH=$(get_default_branch)

# Ensure we're on default branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$DEFAULT_BRANCH" ]; then
  echo "Switching to $DEFAULT_BRANCH branch..."
  git checkout "$DEFAULT_BRANCH"
fi

# Pull latest default branch
echo "Pulling latest $DEFAULT_BRANCH branch..."
git pull origin "$DEFAULT_BRANCH"

# Create the worktree (at the same level as other worktrees)
echo "Creating worktree: $WORKTREE_DIR"
echo "With branch: $BRANCH_NAME"

# Get the parent directory of the main worktree
WORKTREES_PARENT="$(dirname "$MAIN_WORKTREE")"
NEW_WORKTREE_PATH="$WORKTREES_PARENT/$WORKTREE_DIR"

git worktree add "$NEW_WORKTREE_PATH" -b "$BRANCH_NAME"

if [ $? -eq 0 ]; then
  echo "Worktree created successfully!"
  
  # Copy Rails credential key files and VS Code tasks if they exist
  echo "Copying Rails credential keys and VS Code settings..."
  
  # Define the files to copy
  FILES_TO_COPY=(
    "web-app/config/master.key"
    "web-app/config/credentials/development.key"
    "web-app/config/credentials/production.key"
    "web-app/config/credentials/staging.key"
    ".vscode/tasks.json"
    "mcp-json-templates/.secrets"
  )
  
  # Copy each file if it exists (from current worktree)
  for FILE in "${FILES_TO_COPY[@]}"; do
    SOURCE_FILE="$CURRENT_ROOT/$FILE"
    if [ -f "$SOURCE_FILE" ]; then
      TARGET_FILE="$NEW_WORKTREE_PATH/$FILE"
      TARGET_DIR=$(dirname "$TARGET_FILE")
      
      # Create target directory if it doesn't exist
      mkdir -p "$TARGET_DIR"
      
      # Copy the file
      cp "$SOURCE_FILE" "$TARGET_FILE"
      echo "  ✓ Copied $FILE"
    else
      echo "  ⚠ $FILE not found in source directory"
    fi
  done
  
  # Handle MCP template activation if requested
  if [ -n "$MCP_TEMPLATE" ]; then
    activate_mcp_template "$MCP_TEMPLATE" "$NEW_WORKTREE_PATH"
  fi
  
  echo "Opening with ccv..."
  cd "$NEW_WORKTREE_PATH"
  claude --dangerously-skip-permissions
  
  # If we stashed changes, apply them back
  if [ "$STASHED" = true ]; then
    echo ""
    echo "Returning to original location and applying stashed changes..."
    cd "$CURRENT_ROOT"
    git stash pop
    if [ $? -eq 0 ]; then
      echo "Stashed changes reapplied successfully!"
    else
      echo "Failed to reapply stashed changes. They remain in your stash."
      echo "You can manually apply them later with: git stash pop"
    fi
  fi
else
  echo "Failed to create worktree"
  exit 1
fi