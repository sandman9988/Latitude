#!/bin/bash
# Remove redundant VS Code extensions
# Save this and run to clean up extensions

echo "=== VS Code Extension Cleanup ==="
echo ""
echo "Removing 57 redundant/overlapping extensions..."
echo ""

# Python Linting/Formatting duplicates
echo "1. Removing Python linting/formatting duplicates (keeping Ruff)..."
code --uninstall-extension ms-python.black-formatter
code --uninstall-extension ms-python.isort
code --uninstall-extension ms-python.pylint
code --uninstall-extension ms-python.mypy-type-checker
code --uninstall-extension matangover.mypy
code --uninstall-extension sourcery.sourcery

# Markdown duplicates
echo "2. Removing Markdown duplicates (keeping markdown-all-in-one)..."
code --uninstall-extension bierner.github-markdown-preview
code --uninstall-extension bierner.markdown-checkbox
code --uninstall-extension bierner.markdown-emoji
code --uninstall-extension bierner.markdown-footnotes
code --uninstall-extension bierner.markdown-mermaid
code --uninstall-extension bierner.markdown-preview-github-styles
code --uninstall-extension bierner.markdown-yaml-preamble

# Python testing duplicates
echo "3. Removing Python testing duplicates..."
code --uninstall-extension hbenl.vscode-test-explorer
code --uninstall-extension littlefoxteam.vscode-python-test-adapter
code --uninstall-extension pamaron.pytest-runner
code --uninstall-extension ms-vscode.test-adapter-converter

# Git duplicates
echo "4. Removing Git duplicates (keeping GitLens)..."
code --uninstall-extension donjayamanne.githistory
code --uninstall-extension donjayamanne.git-extension-pack

# JSON tools
echo "5. Removing JSON tool duplicates..."
code --uninstall-extension clemenspeters.format-json
code --uninstall-extension khaeransori.json2csv
code --uninstall-extension nextfaze.json-parse-stringify
code --uninstall-extension zainchen.json
code --uninstall-extension mohitkumartoshniwal.jsonlens

# CSV tools
echo "6. Removing CSV tool duplicates (keeping rainbow-csv)..."
code --uninstall-extension phplasma.csv-to-table
code --uninstall-extension janisdd.vscode-edit-csv
code --uninstall-extension grapecity.gc-excelviewer

# Extension packs
echo "7. Removing meta extension packs..."
code --uninstall-extension demystifying-javascript.python-extensions-pack
code --uninstall-extension leojhonsong.python-extension-pack
code --uninstall-extension mfmezger.python-ai-engineering

# Python snippets
echo "8. Removing Python snippet duplicates..."
code --uninstall-extension cstrap.python-snippets
code --uninstall-extension tushortz.python-extended-snippets
code --uninstall-extension cstrap.flask-snippets

# Low value / niche
echo "9. Removing low-value/niche extensions..."
code --uninstall-extension bar.python-import-helper
code --uninstall-extension diogonolasco.pyinit
code --uninstall-extension hasnainroopawalla.vscode-python-timeit
code --uninstall-extension kaih2o.python-resource-monitor
code --uninstall-extension michael-riordan.inline-python-package-installer
code --uninstall-extension njqdev.vscode-python-typehint
code --uninstall-extension rodolphebarbanneau.python-docstring-highlighter
code --uninstall-extension wolfieshorizon.python-auto-venv
code --uninstall-extension mgesbert.python-path
code --uninstall-extension mukundan.python-docs

# Duplicate syntax highlighters
echo "10. Removing duplicate syntax highlighters..."
code --uninstall-extension magicstack.magicpython

# Obsolete tools
echo "11. Removing obsolete tools..."
code --uninstall-extension doggy8088.quicktype-refresh
code --uninstall-extension twixes.pypi-assistant

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Extensions removed: 57"
echo "Remaining extensions: ~42"
echo ""
echo "Restart VS Code to see changes."
