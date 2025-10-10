#!/bin/bash

# Directory containing our custom git hooks
HOOKS_DIR="$(git rev-parse --show-toplevel)/.githooks"
# Directory where git looks for hooks
GIT_HOOKS_DIR="$(git rev-parse --show-toplevel)/.git/hooks"

echo "Installing Git hooks..."

# Ensure the git hooks directory exists
mkdir -p "$GIT_HOOKS_DIR"

# Loop through all files in the hooks directory
for hook in "$HOOKS_DIR"/*; do
    # Skip if no files are found
    [ -e "$hook" ] || continue
    
    # Get just the filename without the path
    hook_name=$(basename "$hook")
    
    # Make the original hook file executable
    chmod +x "$HOOKS_DIR/$hook_name"
    
    # Create symbolic link in git hooks directory
    ln -sf "$HOOKS_DIR/$hook_name" "$GIT_HOOKS_DIR/$hook_name"
    
    echo "Installed $hook_name"
done

echo "Git hooks installation complete!"
