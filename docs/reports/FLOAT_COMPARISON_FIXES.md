# Floating Point Comparison Fixes

## Summary
Fixed all instances of direct floating point equality (`==`) and inequality (`!=`) comparisons in the trading bot codebase. These comparisons are unreliable due to floating point precision issues.

## Changes Made

### 1. Enhanced `safe_math.py`
Added three new utility methods for safe float comparisons:

- **`is_zero(x, eps=SAFE_EPSILON)`**: Checks if a float is effectively zero within epsilon tolerance
- **`is_not_zero(x, eps=SAFE_EPSILON)`**: Checks if a float is effectively non-zero
- **`is_close(a, b, rel_tol=1e-9, abs_tol=SAFE_EPSILON)`**: Checks if two floats are approximately equal using both relative and absolute tolerance (similar to `math.isclose()`)

### 2. Fixed Files

#### `dual_policy.py`
- **Line 343**: `if self.entry_price == 0:` → `if SafeMath.is_zero(self.entry_price):`
- **Line 702**: `assert policy.mfe == 0.0` → `assert SafeMath.is_zero(policy.mfe)`

#### `trade_exporter.py`
- **Line 104**: `if entry_price == 0:` → `if SafeMath.is_zero(entry_price):`
- **Line 109**: `entry_price = exit_price if exit_price != 0 else 1.0` → `entry_price = exit_price if SafeMath.is_not_zero(exit_price) else 1.0`

#### `ensemble_tracker.py`
- **Line 378**: `if bonus_low == 0.0:` → `if SafeMath.is_zero(bonus_low):`

#### `feature_tournament.py`
- **Line 224**: `if mean_corr == 0:` → `if SafeMath.is_zero(mean_corr):`

## Why This Matters

Floating point numbers cannot precisely represent many decimal values due to binary representation limitations:

```python
>>> format(0.1, ".17g")
'0.10000000000000001'

>>> my_float = 0.1
>>> numerator, denominator = my_float.as_integer_ratio()
>>> f"{numerator} / {denominator}"
'3602879701896397 / 36028797018963968'
```

### Problems with Direct Comparison:
1. **Imprecision**: Base-2 representation cannot exactly store many base-10 fractions
2. **Non-associativity**: The order of operations affects results due to rounding at each step
3. **Accumulation**: Errors accumulate through calculation chains

### Trading Bot Impact:
In a trading context, these issues could lead to:
- Incorrect position management (thinking a position is open when it's actually closed)
- Faulty risk calculations
- Unexpected trade triggers or exits
- Division by values that should be treated as zero

## Best Practices

✅ **DO:**
```python
if SafeMath.is_zero(price):
    ...

if SafeMath.is_close(calculated_value, expected_value):
    ...
```

❌ **DON'T:**
```python
if price == 0:
    ...

if calculated_value == expected_value:
    ...
```

## Testing Recommendations

After these changes, consider adding unit tests to verify:
1. Edge cases around zero values in position calculations
2. Price comparison logic in entry/exit decisions
3. Feature correlation calculations with near-zero values

## Files Modified
- `/home/renierdejager/Documents/ctrader_trading_bot/safe_math.py`
- `/home/renierdejager/Documents/ctrader_trading_bot/dual_policy.py`
- `/home/renierdejager/Documents/ctrader_trading_bot/trade_exporter.py`
- `/home/renierdejager/Documents/ctrader_trading_bot/ensemble_tracker.py`
- `/home/renierdejager/Documents/ctrader_trading_bot/feature_tournament.py`
