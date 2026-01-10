# Error Triage & Remediation Plan
## Total: ~745 IDE Problems

### ✅ COMPLETED (Session 1)
- [x] Removed 15 unused `# type: ignore` comments (ctrader_ddqn_paper.py)
- [x] Fixed type incompatibility: `last_disconnect_reason` now uses `str(reason)`
- [x] Marked unused `frame` parameter with `# noqa: ARG002`

### 🔧 P1 - Must Fix (Will Fix Next)

#### 1. Implicit Optional Type Hints (~25 occurrences)
**Issue:** PEP 484 prohibits `def func(x=None)` without `Optional[T]`  
**Files:** activity_monitor.py, agent_arena.py, trade_analyzer.py  
**Fix:** `def func(x: Optional[int] = None)` or `x: int | None = None`

#### 2. datetime.utcnow() Deprecation (~20 occurrences)  
**Issue:** `datetime.datetime.utcnow()` is deprecated in Python 3.12+  
**Files:** hud_tabbed.py, performance_tracker.py  
**Fix:** Replace with `datetime.now(UTC)`

#### 3. Type Annotation Missing (~40 occurrences)
**Issue:** Lists/dicts need type hints  
**Example:** `recent_rewards = []` → `recent_rewards: list[float] = []`  
**Files:** activity_monitor.py, ensemble_tracker.py, feature_tournament.py

#### 4. Incompatible Return Types (~15 occurrences)
**Issue:** Functions return `Any` instead of declared types  
**Files:** feature_tournament.py, ring_buffer.py  
**Fix:** Add explicit type casts or fix return type declaration

### ⚠️ P2 - Should Consider

#### 5. Line Length Violations (~120 occurrences)
**Range:** 101-158 characters (limit: 100)  
**Decision:** Most are 101-112 (acceptable), but 120+ should be split  
**Action:** Fix lines >115 chars, add `# noqa: E501` for acceptable cases

#### 6. Lazy Logging (~80 occurrences)
**Issue:** `LOG.info(f"Value: {x}")` is slower than `LOG.info("Value: %s", x)`  
**Impact:** Performance (minimal in this codebase)  
**Action:** Fix in hot paths (main loop, tick processing)

#### 7. Missing Docstrings (~60 occurrences)
**Files:** test files, utility functions  
**Action:** Add docstrings to public APIs only

### 📋 P3 - Suppress/Document (Low Priority)

#### 8. Spelling Warnings (~150 occurrences)  
**Issue:** Domain-specific terms flagged  
**Terms:** ctrader, BTCUSD, XAUUSD, vpin, VPIN, lookback, Satchell, Sortino, clord, USEC  
**Action:** Create `.vscode/settings.json` with custom dictionary

#### 9. Catching General Exception (~50 occurrences)
**Rationale:** Trading bot must stay alive during FIX protocol errors  
**Action:** Add `# noqa: BLE001` comments with justification

#### 10. Complexity Warnings (~40 occurrences)
- "Too many statements" (50+ statements)
- "Too many branches" (12+ branches)  
- "Cognitive complexity" (15+ score)

**Files:** ctrader_ddqn_paper.py (main loop is inherently complex)  
**Action:** Document as intentional, consider refactoring only if logic changes

#### 11. Sourcery Suggestions (~30 occurrences)
**Type:** "Use named expression", "Extract into method", etc.  
**Decision:** Cosmetic improvements only, not errors  
**Action:** Apply selectively where it improves readability

#### 12. Redefining Names from Outer Scope (~40 occurrences)
**Context:** Test code using common variable names (state, action, metrics)  
**Action:** Rename test variables or suppress with `# noqa: PLW2901`

#### 13. Import Outside Toplevel (~20 occurrences)
**Rationale:** Lazy imports to avoid circular dependencies or optional deps  
**Action:** Add `# noqa: PLC0415` with reason

### 📊 Next Steps

**Immediate (This Session):**
1. Fix all "PEP 484 implicit Optional" issues (25 fixes)
2. Fix all datetime.utcnow() calls (20 fixes)
3. Add type annotations for critical paths (40 fixes)
4. Fix incompatible return types (15 fixes)

**Short Term:**
5. Add spelling dictionary for domain terms
6. Fix lines >115 chars
7. Add lazy logging to hot paths

**Long Term:**
8. Add docstrings to public APIs
9. Consider refactoring high-complexity functions
10. Apply useful Sourcery suggestions

**Estimation:**
- P1 fixes: ~100 issues → reduce from 745 to ~645 (13% reduction)
- P2 fixes: ~260 issues → reduce to ~385 (48% total reduction)
- P3 suppression: ~250 issues → final ~135 legitimate issues (82% total reduction)

### 🎯 Target: <150 IDE Problems
Focus on **real type safety, deprecation warnings, and missing annotations**.  
Suppress **style preferences and inherent complexity** with documented reasons.
