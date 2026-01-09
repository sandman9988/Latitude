# Contributing Guidelines - Incremental Development

## Workflow

### For Each Update:

```bash
# 1. Create feature branch
git checkout -b update-X.Y-feature-name

# 2. Make your changes (small, focused)
# Edit files...

# 3. Test immediately
./tests/test_runner.sh

# 4. If tests pass, test with live bot
# Kill existing bot if running
pkill -f ctrader_ddqn_paper.py

# Start fresh
export CTRADER_USERNAME="5179095"
export CTRADER_PASSWORD_QUOTE="your_password"
export CTRADER_PASSWORD_TRADE="your_password"
./run.sh

# 5. Watch logs for 5 minutes
tail -f logs/python/ctrader_*.log

# 6. If working correctly, commit
git add .
git commit -m "Update X.Y: Brief description

- What was added
- What was changed
- How to test it
"

# 7. Merge to main
git checkout main
git merge update-X.Y-feature-name

# 8. Tag release
git tag v0.X.Y
```

## Testing Checklist

Before committing ANY change:

- [ ] Code runs without syntax errors
- [ ] No new warnings in logs
- [ ] Bot connects to both FIX sessions
- [ ] Market data still streaming
- [ ] New feature logs visible
- [ ] No regression in existing features
- [ ] Tested for at least 5 minutes live

## Rollback Procedure

If something breaks:

```bash
# Quick rollback
git checkout main
git reset --hard v0.X.Y  # Last known good version

# Restart bot
pkill -f ctrader_ddqn_paper.py
./run.sh
```

## Code Standards

### Python Style
- Use type hints where possible
- Docstrings for all classes/functions
- Keep functions under 50 lines
- Single responsibility principle

### Logging
```python
# Use structured logging
logger.info("[FEATURE] Description: value=%s", value)

# Levels:
# DEBUG - Verbose details
# INFO - Normal operations
# WARNING - Recoverable issues
# ERROR - Failures requiring attention
# CRITICAL - System-threatening issues
```

### Error Handling
```python
# Always handle errors gracefully
try:
    risky_operation()
except SpecificException as e:
    logger.error("Operation failed: %s", e)
    # Fallback or default behavior
```

## Feature Size Guidelines

**Good update (2-4 hours):**
- Add MFE/MAE tracking
- Add one new feature
- Add one metric
- Fix one bug

**Too big (don't do this):**
- Refactor entire architecture
- Add dual agents + reward shaping
- Add 20 features at once

Break big features into small pieces!

## Documentation

Update these files when relevant:
- `DEVELOPMENT_ROADMAP.md` - Mark items complete
- `README.md` - Add new features to description
- `CHANGELOG.md` - Log all changes
- `KNOWN_ISSUES.md` - Document any bugs

## Questions?

Check:
1. DEVELOPMENT_ROADMAP.md - Overall plan
2. This file - How to contribute
3. README.md - How to use
4. Logs - What's happening
