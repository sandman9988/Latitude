"""
Event-Relative Time Features
Convert absolute time to event-relative coordinates
Handbook Section 9.4 - Event-Relative Time

Instead of "Tuesday 14:30", use "30 mins after London open, 6 hours before rollover"
"""

import logging
from datetime import datetime, timezone, time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math

from safe_math import SafeMath

LOG = logging.getLogger(__name__)


@dataclass
class SessionTimes:
    """Trading session times in UTC"""
    name: str
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int
    
    def get_open_minutes(self) -> int:
        """Get session open in minutes from UTC midnight"""
        return self.open_hour * 60 + self.open_minute
    
    def get_close_minutes(self) -> int:
        """Get session close in minutes from UTC midnight"""
        return self.close_hour * 60 + self.close_minute


class EventTimeFeatureEngine:
    """
    Calculate time features relative to market events
    
    Provides normalized features for:
    - Session proximity (London open/close, NY open/close)
    - Rollover time (swap charges)
    - Session overlaps (high liquidity periods)
    - Week/month position
    """
    
    # Major trading sessions (UTC)
    SESSIONS = {
        'SYDNEY': SessionTimes('Sydney', 21, 0, 6, 0),  # 21:00-06:00 UTC
        'TOKYO': SessionTimes('Tokyo', 23, 0, 8, 0),   # 23:00-08:00 UTC
        'LONDON': SessionTimes('London', 7, 0, 16, 0),  # 07:00-16:00 UTC
        'NEW_YORK': SessionTimes('New York', 12, 0, 21, 0),  # 12:00-21:00 UTC
    }
    
    # Forex rollover time (17:00 ET = 22:00 UTC typically)
    ROLLOVER_UTC_HOUR = 22
    ROLLOVER_UTC_MINUTE = 0
    
    def __init__(self):
        """Initialize event time feature engine"""
        self.cache: Dict[str, Dict] = {}
        self.last_update_minute = -1
    
    def calculate_features(self, dt: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate all event-relative time features
        
        Args:
            dt: Datetime to calculate features for (UTC), defaults to now
            
        Returns:
            Dictionary of normalized features
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Cache features per minute (expensive to recalculate every tick)
        current_minute = dt.hour * 60 + dt.minute
        cache_key = f"{dt.date()}_{current_minute}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        features = {}
        
        # Session proximity features
        for session_name, session in self.SESSIONS.items():
            prefix = session_name.lower()
            
            # Minutes to/from open
            mins_to_open, mins_from_open = self._calc_session_proximity(
                dt, session.get_open_minutes()
            )
            features[f'{prefix}_mins_to_open'] = self._normalize_minutes(mins_to_open)
            features[f'{prefix}_mins_from_open'] = self._normalize_minutes(mins_from_open)
            
            # Minutes to/from close
            mins_to_close, mins_from_close = self._calc_session_proximity(
                dt, session.get_close_minutes()
            )
            features[f'{prefix}_mins_to_close'] = self._normalize_minutes(mins_to_close)
            features[f'{prefix}_mins_from_close'] = self._normalize_minutes(mins_from_close)
            
            # Is session active (binary)
            features[f'{prefix}_is_active'] = float(self._is_session_active(dt, session))
        
        # Rollover proximity
        rollover_mins = self.ROLLOVER_UTC_HOUR * 60 + self.ROLLOVER_UTC_MINUTE
        mins_to_rollover, mins_from_rollover = self._calc_session_proximity(dt, rollover_mins)
        features['mins_to_rollover'] = self._normalize_minutes(mins_to_rollover)
        features['mins_from_rollover'] = self._normalize_minutes(mins_from_rollover)
        
        # Session overlaps (high liquidity)
        features['london_ny_overlap'] = float(
            self._is_session_active(dt, self.SESSIONS['LONDON']) and
            self._is_session_active(dt, self.SESSIONS['NEW_YORK'])
        )
        features['tokyo_london_overlap'] = float(
            self._is_session_active(dt, self.SESSIONS['TOKYO']) and
            self._is_session_active(dt, self.SESSIONS['LONDON'])
        )
        features['sydney_tokyo_overlap'] = float(
            self._is_session_active(dt, self.SESSIONS['SYDNEY']) and
            self._is_session_active(dt, self.SESSIONS['TOKYO'])
        )
        
        # Week position (0 = Sunday open, 1 = Friday close)
        features['week_progress'] = self._calc_week_progress(dt)
        
        # Month position (0 = month start, 1 = month end)
        features['month_progress'] = self._calc_month_progress(dt)
        
        # Day of week (0-6, where 0=Monday in ISO, but we use 0=Sunday for FX)
        features['day_of_week'] = float((dt.weekday() + 1) % 7)  # Sunday=0
        
        # Hour of day (0-23)
        features['hour_of_day'] = float(dt.hour)
        
        # Is weekend
        features['is_weekend'] = float(dt.weekday() >= 5)  # Saturday=5, Sunday=6
        
        # Store in cache
        self.cache[cache_key] = features
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:500]
            for key in oldest_keys:
                del self.cache[key]
        
        return features
    
    def _calc_session_proximity(self, dt: datetime, event_minutes: int) -> Tuple[float, float]:
        """
        Calculate minutes to and from an event
        
        Args:
            dt: Current datetime (UTC)
            event_minutes: Event time in minutes from UTC midnight (0-1439)
            
        Returns:
            (minutes_to_event, minutes_from_event)
            - minutes_to_event: positive if event is ahead, negative if passed
            - minutes_from_event: positive if event has passed, negative if upcoming
        """
        current_minutes = dt.hour * 60 + dt.minute
        
        # Simple case: same day
        diff = event_minutes - current_minutes
        
        # Handle day wrap
        if diff < -720:  # More than 12 hours in past = upcoming tomorrow
            diff += 1440
        elif diff > 720:  # More than 12 hours in future = was yesterday
            diff -= 1440
        
        minutes_to = diff
        minutes_from = -diff
        
        return minutes_to, minutes_from
    
    def _is_session_active(self, dt: datetime, session: SessionTimes) -> bool:
        """Check if a session is currently active"""
        current_minutes = dt.hour * 60 + dt.minute
        open_minutes = session.get_open_minutes()
        close_minutes = session.get_close_minutes()
        
        # Handle overnight sessions (e.g., Sydney: 21:00-06:00)
        if open_minutes > close_minutes:
            return current_minutes >= open_minutes or current_minutes < close_minutes
        else:
            return open_minutes <= current_minutes < close_minutes
    
    def _normalize_minutes(self, minutes: float, max_range: float = 720.0) -> float:
        """
        Normalize minutes to [-1, 1] range
        
        Args:
            minutes: Minutes value
            max_range: Maximum expected range (default 12 hours = 720 mins)
            
        Returns:
            Normalized value in [-1, 1]
        """
        return SafeMath.clamp(minutes / max_range, -1.0, 1.0)
    
    def _calc_week_progress(self, dt: datetime) -> float:
        """
        Calculate progress through trading week (0-1)
        
        FX week: Sunday 21:00 UTC to Friday 21:00 UTC
        """
        # Convert to minutes since week start (Sunday 21:00 UTC)
        weekday = (dt.weekday() + 1) % 7  # Sunday=0
        hour = dt.hour
        minute = dt.minute
        
        # Minutes since Sunday 00:00
        minutes_since_sunday = weekday * 1440 + hour * 60 + minute
        
        # Adjust for FX week starting at Sunday 21:00
        fx_week_start = 21 * 60  # Sunday 21:00
        minutes_into_week = minutes_since_sunday - fx_week_start
        
        if minutes_into_week < 0:
            # Before Sunday 21:00 = end of previous week
            minutes_into_week += 7 * 1440
        
        # FX week length: Sunday 21:00 to Friday 21:00 = 5 days
        fx_week_length = 5 * 1440
        
        progress = SafeMath.safe_div(minutes_into_week, fx_week_length, 0.0)
        return SafeMath.clamp(progress, 0.0, 1.0)
    
    def _calc_month_progress(self, dt: datetime) -> float:
        """Calculate progress through month (0-1)"""
        import calendar
        
        day_of_month = dt.day
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        
        # Include intraday progress
        hour_fraction = dt.hour / 24.0
        progress = (day_of_month - 1 + hour_fraction) / days_in_month
        
        return SafeMath.clamp(progress, 0.0, 1.0)
    
    def get_active_sessions(self, dt: Optional[datetime] = None) -> list:
        """Get list of currently active session names"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        active = []
        for name, session in self.SESSIONS.items():
            if self._is_session_active(dt, session):
                active.append(name)
        
        return active
    
    def get_next_major_event(self, dt: Optional[datetime] = None) -> Tuple[str, int]:
        """
        Get the next major market event and minutes until it
        
        Returns:
            (event_name, minutes_until)
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        current_minutes = dt.hour * 60 + dt.minute
        events = []
        
        # Check all session opens/closes
        for session_name, session in self.SESSIONS.items():
            open_mins = session.get_open_minutes()
            close_mins = session.get_close_minutes()
            
            mins_to_open, _ = self._calc_session_proximity(dt, open_mins)
            mins_to_close, _ = self._calc_session_proximity(dt, close_mins)
            
            if mins_to_open > 0:
                events.append((f"{session_name} Open", mins_to_open))
            if mins_to_close > 0:
                events.append((f"{session_name} Close", mins_to_close))
        
        # Check rollover
        rollover_mins = self.ROLLOVER_UTC_HOUR * 60 + self.ROLLOVER_UTC_MINUTE
        mins_to_rollover, _ = self._calc_session_proximity(dt, rollover_mins)
        if mins_to_rollover > 0:
            events.append(("Rollover", mins_to_rollover))
        
        if not events:
            return ("None", 0)
        
        # Return nearest event
        events.sort(key=lambda x: x[1])
        return events[0]
    
    def is_high_liquidity_period(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if current time is a high liquidity period
        
        High liquidity = session overlap or major session active
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        
        features = self.calculate_features(dt)
        
        # Session overlaps are high liquidity
        if features['london_ny_overlap'] > 0:
            return True
        if features['tokyo_london_overlap'] > 0:
            return True
        
        # Individual major sessions
        if features['london_is_active'] > 0:
            return True
        if features['new_york_is_active'] > 0:
            return True
        
        return False


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("EVENT-RELATIVE TIME FEATURES - TEST SUITE")
    print("=" * 80)
    
    engine = EventTimeFeatureEngine()
    
    # Test 1: Current time features
    print("\n[Test 1] Current Time Features")
    print("-" * 80)
    
    now = datetime.now(timezone.utc)
    print(f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    features = engine.calculate_features(now)
    
    print(f"\nActive sessions: {', '.join(engine.get_active_sessions(now))}")
    
    # Session features
    print("\nSession Proximity:")
    for session in ['london', 'new_york', 'tokyo', 'sydney']:
        is_active = features.get(f'{session}_is_active', 0)
        mins_to_open = features.get(f'{session}_mins_to_open', 0)
        mins_to_close = features.get(f'{session}_mins_to_close', 0)
        
        status = "ACTIVE" if is_active else "CLOSED"
        print(f"  {session.title():12s}: {status:8s} | "
              f"Open in: {mins_to_open:+.3f} | Close in: {mins_to_close:+.3f}")
    
    # Rollover
    print(f"\nRollover:")
    print(f"  Minutes to: {features['mins_to_rollover']:+.3f}")
    print(f"  Minutes from: {features['mins_from_rollover']:+.3f}")
    
    # Overlaps
    print(f"\nSession Overlaps:")
    print(f"  London-NY: {features['london_ny_overlap']:.0f}")
    print(f"  Tokyo-London: {features['tokyo_london_overlap']:.0f}")
    print(f"  Sydney-Tokyo: {features['sydney_tokyo_overlap']:.0f}")
    
    # Time position
    print(f"\nTime Position:")
    print(f"  Week progress: {features['week_progress']:.3f}")
    print(f"  Month progress: {features['month_progress']:.3f}")
    print(f"  Day of week: {features['day_of_week']:.0f}")
    print(f"  Hour of day: {features['hour_of_day']:.0f}")
    print(f"  Is weekend: {features['is_weekend']:.0f}")
    
    # Test 2: Specific times
    print("\n[Test 2] Specific Times")
    print("-" * 80)
    
    test_times = [
        datetime(2026, 1, 10, 7, 0, tzinfo=timezone.utc),   # London open
        datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc),  # NY open
        datetime(2026, 1, 10, 22, 0, tzinfo=timezone.utc),  # Rollover
        datetime(2026, 1, 11, 3, 0, tzinfo=timezone.utc),   # Asian session
    ]
    
    for test_dt in test_times:
        features = engine.calculate_features(test_dt)
        active = engine.get_active_sessions(test_dt)
        high_liq = engine.is_high_liquidity_period(test_dt)
        next_event, mins = engine.get_next_major_event(test_dt)
        
        print(f"\n{test_dt.strftime('%Y-%m-%d %H:%M UTC')}:")
        print(f"  Active: {', '.join(active) if active else 'None'}")
        print(f"  High liquidity: {high_liq}")
        print(f"  Next event: {next_event} in {mins:.0f} mins")
    
    # Test 3: Week progress calculation
    print("\n[Test 3] Week Progress")
    print("-" * 80)
    
    # Sunday 21:00 UTC (week start)
    sun_start = datetime(2026, 1, 11, 21, 0, tzinfo=timezone.utc)
    # Wednesday 12:00 UTC (mid-week)
    wed_mid = datetime(2026, 1, 14, 12, 0, tzinfo=timezone.utc)
    # Friday 21:00 UTC (week end)
    fri_end = datetime(2026, 1, 16, 21, 0, tzinfo=timezone.utc)
    
    for label, dt in [("Sunday 21:00", sun_start), ("Wednesday 12:00", wed_mid), ("Friday 21:00", fri_end)]:
        features = engine.calculate_features(dt)
        print(f"{label:20s}: progress = {features['week_progress']:.3f}")
    
    # Test 4: Feature count
    print("\n[Test 4] Feature Summary")
    print("-" * 80)
    
    features = engine.calculate_features()
    print(f"Total features: {len(features)}")
    print(f"\nFeature list:")
    for name, value in sorted(features.items()):
        print(f"  {name:30s}: {value:+.4f}")
    
    print("\n" + "=" * 80)
    print("✅ EVENT-RELATIVE TIME FEATURES READY")
    print("=" * 80)
    print("\nKey benefits:")
    print("  ✓ Session-aware features (better than raw time)")
    print("  ✓ Normalized [-1, 1] for neural networks")
    print("  ✓ Captures market microstructure")
    print("  ✓ Liquidity period detection")
    print("  ✓ Event proximity (rollover, opens, closes)")
