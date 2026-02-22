# Database Migration Analysis

## Current Persistence Architecture

**System**: JSON files with `AtomicPersistence` (CRC32 validation)
**Files**:
- `data/state/trade_integration_BTCUSD.json` - Position/tracker state
- `data/bot_config.json` - Bot configuration
- `data/current_position.json` - Position data
- `data/decision_log.json` - DDQN decisions
- `data/training_stats.json` - Learning metrics
- `logs/audit/decisions.jsonl` - JSONL audit log
- `logs/audit/trade_audit.jsonl` - Trade audit trail

**Strengths**:
- ✅ Simple, human-readable
- ✅ Easy to inspect/debug
- ✅ CRC32 validation prevents corruption
- ✅ Atomic writes with backup/rollback
- ✅ No external dependencies

**Weaknesses**:
- ❌ Floating point precision artifacts (NOW FIXED)
- ❌ Limited query capabilities
- ❌ No transaction support across multiple files
- ❌ Concurrent access issues (file locks)
- ❌ No schema versioning
- ❌ Manual data migrations
- ❌ Limited data integrity constraints

---

## Database Options

### Option 1: SQLite (Recommended)

**Pros**:
- ✅ Zero-configuration, embedded database
- ✅ ACID transactions (atomic consistency)
- ✅ Proper float/decimal storage
- ✅ SQL queries for analysis
- ✅ Schema migrations (via Alembic)
- ✅ Single file database (easy backup)
- ✅ No external server required
- ✅ Built into Python stdlib

**Cons**:
- ❌ Learning curve (SQL/ORM)
- ❌ More complex backup strategy
- ❌ Potential file corruption (less likely than JSON)

**Use Case**: Perfect for single-bot deployment, local development

### Option 2: PostgreSQL

**Pros**:
- ✅ Production-grade reliability
- ✅ Advanced features (JSONB, time-series extensions)
- ✅ Concurrent connections
- ✅ Replication/HA options
- ✅ Point-in-time recovery

**Cons**:
- ❌ Requires server installation/management
- ❌ More complex setup
- ❌ Overkill for single bot

**Use Case**: Multi-bot production environment, cloud deployment

---

## Migration Impact Analysis

### What Would Improve

1. **Data Integrity**
   - FOREIGN KEY constraints between positions/trades/decisions
   - CHECK constraints for valid states
   - UNIQUE constraints prevent duplicate broker tickets
   - Transaction rollback on error

2. **Querying & Analytics**
   - SQL queries for performance analysis
   - Aggregate functions (AVG, SUM, COUNT)
   - JOIN operations between related data
   - Time-series analysis without loading all data

3. **Float Precision**
   - DECIMAL type for exact monetary values
   - REAL/DOUBLE PRECISION for prices
   - No JSON serialization artifacts

4. **State Recovery**
   - Single source of truth in DB
   - No orphaned position tickets (FOREIGN KEY cascade)
   - Atomic multi-table updates
   - Consistent state across restarts

5. **Schema Evolution**
   - Alembic migrations for schema changes
   - Version tracking
   - Rollback support

### What Would Stay the Same

- Core trading logic unchanged
- FIX protocol interaction unchanged
- DDQN model training unchanged
- Risk management logic unchanged

### What Would Break

- `AtomicPersistence` class (replace with SQLAlchemy)
- All `save_json()`/`load_json()` calls
- Direct JSON file reads in analysis scripts
- File-based backup/restore scripts

---

## Recommendation

### Immediate Fix (DONE ✅)
**Round floats to 8 decimals** before JSON serialization - fixes cosmetic issue without migration overhead.

### Short-Term (3-6 months)
**Stick with JSON files** for now because:
1. Bot is in development/testing phase
2. Single-bot deployment
3. Current persistence is stable (with float fix)
4. Focus should be on trading strategy, not infrastructure

### Long-Term (6-12 months)
**Migrate to SQLite** when:
1. Bot enters production with real money
2. Need better analytics/reporting
3. Want to track multiple symbols/strategies
4. Team grows beyond 1 developer
5. Need audit trail compliance

**Schema Design**:
```sql
-- Positions table
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    broker_ticket TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    direction INTEGER NOT NULL,  -- 1=LONG, -1=SHORT
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_price REAL,
    exit_time TIMESTAMP,
    pnl REAL,
    mfe REAL,
    mae REAL,
    status TEXT CHECK(status IN ('OPEN', 'CLOSED'))
);

-- Decisions table (DDQN)
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    state_hash TEXT NOT NULL,
    action INTEGER NOT NULL,
    q_values TEXT NOT NULL,  -- JSON array
    epsilon REAL NOT NULL,
    explore BOOLEAN NOT NULL
);

-- Trades table (execution)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    position_id INTEGER REFERENCES positions(id),
    order_id TEXT UNIQUE NOT NULL,
    side TEXT CHECK(side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL,
    timestamp TIMESTAMP NOT NULL
);

-- Circuit breaker events
CREATE TABLE circuit_breaker_events (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    breaker_type TEXT NOT NULL,
    metric_value REAL NOT NULL,
    threshold REAL NOT NULL,
    positions_closed INTEGER DEFAULT 0
);
```

---

## Migration Steps (Future)

1. **Design Schema** (1 week)
   - ERD diagram
   - Normalization
   - Index strategy

2. **Setup SQLAlchemy** (1 week)
   - Models (Position, Trade, Decision)
   - Session management
   - Connection pooling

3. **Data Migration** (3 days)
   - Export JSON to CSV
   - Import to SQLite
   - Validate data integrity

4. **Update Persistence Layer** (1 week)
   - Replace `AtomicPersistence` with `DatabasePersistence`
   - Update all save/load calls
   - Add transaction context managers

5. **Testing** (1 week)
   - Unit tests for DB layer
   - Integration tests
   - Crash recovery tests
   - Performance benchmarks

6. **Alembic Setup** (2 days)
   - Initialize migrations
   - Create baseline migration
   - CI/CD integration

**Total Effort**: ~4 weeks (100-120 hours)

---

## Decision

**NOW**: ✅ Fixed float precision with rounding (15 min effort)

**LATER**: Consider SQLite migration when bot reaches production maturity and requires better analytics/auditability.

**Current JSON approach is sufficient** for development/testing phase.

---

## References

- [SQLite When to Use](https://www.sqlite.org/whentouse.html)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [Alembic Migrations](https://alembic.sqlalchemy.org/en/latest/)
