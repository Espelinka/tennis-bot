import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from dotenv import load_dotenv
from scipy.stats import binom
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
PB_URL = os.getenv("PB_URL", "http://127.0.0.1:8095")
PB_EMAIL = os.getenv("PB_ADMIN_EMAIL")
PB_PASSWORD = os.getenv("PB_ADMIN_PASSWORD")

# Stats and Logging
stats_funnel = {
    "total_matches": 0,
    "skipped_surface": 0,
    "skipped_hold": 0,
    "skipped_weather": 0,
    "alerts_sent": 0
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
SACKMANN_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
TOUR_AVG_RETURN = 0.22  # Rough approximation for ATP Tour Average Return On Service

# --- Data Loading & Management ---
_stats_cache = {"df": pd.DataFrame(), "expiry": datetime.min}

async def download_csv(year: int) -> pd.DataFrame:
    url = f"{SACKMANN_BASE_URL}atp_matches_{year}.csv"
    logger.info(f"Downloading data for year {year}...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                text = await response.text()
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                return df
            else:
                logger.error(f"Failed to download {url}: {response.status}")
                return pd.DataFrame()

async def get_combined_stats() -> pd.DataFrame:
    global _stats_cache
    if datetime.now() < _stats_cache["expiry"] and not _stats_cache["df"].empty:
        return _stats_cache["df"]

    # Try current and previous years, fallback to 2024/2023 if 404
    years_to_try = [2026, 2025, 2024, 2023]
    dfs = []
    
    for year in years_to_try:
        df = await download_csv(year)
        if not df.empty:
            dfs.append(df)
        
        # We need at least 2 years of data for 52-week coverage
        if len(dfs) >= 2:
            break
            
    if not dfs:
        logger.error("No data could be downloaded from Sackmann repo!")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    _stats_cache["df"] = combined
    _stats_cache["expiry"] = datetime.now() + timedelta(hours=24)
    return combined

# Mapping of major ATP tournament locations to coordinates
# This is a simplified version, in production it should be more extensive or use a Geocoding API
TOURNAMENT_LOCATIONS = {
    "Australian Open": (37.8136, 144.9631),
    "Wimbledon": (51.4343, 0.2145),
    "French Open": (48.8475, 2.2492),
    "US Open": (40.7500, -73.8467),
    "Indian Wells": (33.7233, -116.3750),
    "Miami": (25.9581, -80.2389),
    "Monte Carlo": (43.7483, 7.4331),
    "Madrid": (40.4168, -3.7038),
    "Rome": (41.9028, 12.4964),
    "Canada": (43.6532, -79.3832),
    "Cincinnati": (39.1031, -84.5120),
    "Shanghai": (31.2304, 121.4737),
    "Paris": (48.8566, 2.3522),
    "Dubai": (25.2048, 55.2708),
    "Doha": (25.2854, 51.5310),
}

# Mapping of tournament names to surfaces (simplified)
# Sackmann data has 'Hard', 'Grass', 'Clay', 'Carpet'
SURFACE_MAPPING = {
    "Australian Open": "Hard",
    "Wimbledon": "Grass",
    "French Open": "Clay",
    "US Open": "Hard",
    "Indian Wells": "Hard",
    "Miami": "Hard",
    "Monte Carlo": "Clay",
    "Madrid": "Clay",
    "Rome": "Clay",
    "Paris": "Indoor Hard",
}

def get_tournament_surface(sport_title: str) -> str:
    """Guess surface based on tournament name."""
    for name, surface in SURFACE_MAPPING.items():
        if name.lower() in sport_title.lower():
            return surface
    # Default to Hard for ATP if unknown (common on tour)
    return "Hard"

def get_tournament_coords(sport_title: str) -> Tuple[float, float]:
    """Get coords based on tournament name."""
    for name, coords in TOURNAMENT_LOCATIONS.items():
        if name.lower() in sport_title.lower():
            return coords
    return (0.0, 0.0) # Default unknown

def normalize_name(name: str) -> str:
    """Normalize player name for better matching."""
    # Odds API: "Novak Djokovic"
    # Sackmann: "Novak Djokovic"
    # Usually they match, but we can add simple stripping/casing
    return name.strip().title()


def calculate_weighted_stats(player_name: str, surface: str, df: pd.DataFrame) -> Tuple[float, float, int]:
    """
    Calculates Weighted Moving Average (WMA) for Hold% and Return%
    Matches within last 60 days get x2 weight.
    """
    # Filter by player and surface
    # Note: Jeff Sackmann CSV uses winner_name and loser_name
    p_matches = df[((df['winner_name'] == player_name) | (df['loser_name'] == player_name)) & (df['surface'] == surface)].copy()
    
    if len(p_matches) < 15:
        return 0.0, 0.0, len(p_matches)

    # Convert tourney_date to datetime
    p_matches['date'] = pd.to_datetime(p_matches['tourney_date'], format='%Y%m%d')
    cutoff_date = datetime.now() - timedelta(days=60)
    
    holds = []
    returns = []
    weights = []

    for _, row in p_matches.iterrows():
        weight = 2 if row['date'] >= cutoff_date else 1
        
        # Calculate stats for this match
        try:
            if row['winner_name'] == player_name:
                # Player won
                # Hold% = (w_1stWon + w_2ndWon) / w_svpt
                if row['w_svpt'] > 0:
                    hold_pct = (row['w_1stWon'] + row['w_2ndWon']) / row['w_svpt']
                    # Return% = (l_svpt - (l_1stWon + l_2ndWon)) / l_svpt
                    ret_pct = (row['l_svpt'] - (row['l_1stWon'] + row['l_2ndWon'])) / row['l_svpt']
                else: continue
            else:
                # Player lost
                if row['l_svpt'] > 0:
                    hold_pct = (row['l_1stWon'] + row['l_2ndWon']) / row['l_svpt']
                    ret_pct = (row['w_svpt'] - (row['w_1stWon'] + row['w_2ndWon'])) / row['w_svpt']
                else: continue
            
            holds.append(hold_pct)
            returns.append(ret_pct)
            weights.append(weight)
        except Exception:
            continue

    if not holds:
        return 0.0, 0.0, 0

    w_hold = np.average(holds, weights=weights)
    w_ret = np.average(returns, weights=weights)
    
    return w_hold, w_ret, len(holds)

def calculate_match_probability(hold_a: float, hold_b: float) -> float:
    """
    Calculates probability of 1st set 5:5 (TB reach or 10+ games)
    Using binomial distribution for 5 service games each.
    """
    # Prob Player A holds 5/5
    p_a_5_5 = binom.pmf(5, 5, hold_a)
    # Prob Player A holds 4/5
    p_a_4_5 = binom.pmf(4, 5, hold_a)
    
    # Prob Player B holds 5/5
    p_b_5_5 = binom.pmf(5, 5, hold_b)
    # Prob Player B holds 4/5
    p_b_4_5 = binom.pmf(4, 5, hold_b)
    
    # Scenario 1: Both hold all (0 breaks)
    sc1 = p_a_5_5 * p_b_5_5
    # Scenario 2: One break each (1-1 in breaks)
    sc2 = p_a_4_5 * p_b_4_5
    
    return sc1 + sc2

# --- API Integrations ---

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def get_weather(city: str, lat: float, lon: float, start_time: datetime) -> Optional[float]:
    """Returns wind speed in km/h for the given time."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m",
        "timezone": "auto",
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": start_time.strftime("%Y-%m-%d")
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                hour_idx = start_time.hour
                if "hourly" in data and "wind_speed_10m" in data["hourly"]:
                    return data["hourly"]["wind_speed_10m"][hour_idx]
    return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def get_odds() -> List[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/tennis_atp/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "alternate_total_games_1st_set", # Specific market for 1st set totals
        "oddsFormat": "decimal"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if not data:
                    # Fallback to main totals if 1st set not available (though not perfect for the strategy)
                    params["markets"] = "totals"
                    async with session.get(url, params=params) as resp_fallback:
                        if resp_fallback.status == 200:
                            return await resp_fallback.json()
                return data
            return []

# --- PocketBase Integration ---

class PocketBaseClient:
    def __init__(self, url, email, password):
        self.url = url
        self.email = email
        self.password = password
        self.token = None

    async def authenticate(self):
        async with aiohttp.ClientSession() as session:
            payload = {"identity": self.email, "password": self.password}
            async with session.post(f"{self.url}/api/admins/auth-with-password", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.token = data["token"]
                    logger.info("PB Authenticated successfully")
                else:
                    logger.error(f"PB Auth failed: {await resp.text()}")

    async def save_signal(self, data: dict):
        if not self.token:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.token}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/api/collections/signals/records", json=data, headers=headers) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to save signal to PB: {await resp.text()}")

pb_client = PocketBaseClient(PB_URL, PB_EMAIL, PB_PASSWORD)

# --- Bot Handlers ---

dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("🎾 Теннисный бот (ATP Value Betting) запущен.\nИспользуйте /status для статистики.")

@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    conn_status = "✅ OK" if pb_client.token else "❌ No Auth"
    text = (
        f"📊 <b>Статистика воронки:</b>\n"
        f"- Проверено матчей: {stats_funnel['total_matches']}\n"
        f"- Отсеяно по покрытию: {stats_funnel['skipped_surface']}\n"
        f"- Отсеяно по холду: {stats_funnel['skipped_hold']}\n"
        f"- Отсеяно по погоде: {stats_funnel['skipped_weather']}\n"
        f"- Выдано алертов: {stats_funnel['alerts_sent']}\n\n"
        f"🔗 <b>PocketBase:</b> {conn_status}"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)

# --- Main Engine ---

async def scan_matches(bot: Bot):
    logger.info("Starting scan cycle...")
    df_stats = await get_combined_stats()
    odds_data = await get_odds()
    
    if not isinstance(odds_data, list):
        logger.warning(f"Unexpected Odds API response: {odds_data}")
        return

    for match in odds_data:
        stats_funnel["total_matches"] += 1
        
        sport_title = match.get('sport_title', 'Unknown')
        surface = get_tournament_surface(sport_title)
        
        # Filter 1: Surface
        if surface not in ["Hard", "Grass", "Carpet", "Indoor Hard"]:
            stats_funnel["skipped_surface"] += 1
            continue
            
        p1 = normalize_name(match['home_team'])
        p2 = normalize_name(match['away_team'])
        
        # Filter 2: WMA Stats
        h1, r1, count1 = calculate_weighted_stats(p1, surface, df_stats)
        h2, r2, count2 = calculate_weighted_stats(p2, surface, df_stats)
        
        if count1 < 15 or count2 < 15:
            stats_funnel["skipped_hold"] += 1
            continue
            
        # Filter 3: Dynamic Hold Adjustment
        exp_hold_1 = h1 - (r2 - TOUR_AVG_RETURN)
        exp_hold_2 = h2 - (r1 - TOUR_AVG_RETURN)
        
        if exp_hold_1 < 0.85 or exp_hold_2 < 0.85:
            stats_funnel["skipped_hold"] += 1
            continue
            
        # Filter 4: Weather
        wind_speed = 0
        if surface in ["Hard", "Grass"]:
            lat, lon = get_tournament_coords(sport_title)
            match_time = datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00'))
            
            if lat != 0.0:
                wind_speed = await get_weather(sport_title, lat, lon, match_time)
                if wind_speed and wind_speed > 20:
                    stats_funnel["skipped_weather"] += 1
                    continue

        # Filter 5: Probability Math
        p_total = calculate_match_probability(exp_hold_1, exp_hold_2)
        fair_odds = 1 / p_total if p_total > 0 else 999
        min_target_odds = fair_odds * 1.05
        
        # Filter 6: Odds Value
        best_odds = 0
        bookie_name = ""
        # The-Odds-API structure: match['bookmakers'] -> market
        # Trying both alternate and main total keys
        possible_keys = ['alternate_total_games_1st_set', 'total_games_1st_set', 'totals']
        
        for bookmaker in match.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] in possible_keys:
                    for outcome in market['outcomes']:
                        # Point is 9.5 for alternate, or we check point explicitly
                        if outcome['name'] == 'Over' and outcome.get('point') == 9.5:
                            if outcome['price'] > best_odds:
                                best_odds = outcome['price']
                                bookie_name = bookmaker['title']

        if best_odds >= min_target_odds:
            # Generate Alert
            alert_text = (
                f"🎾 <b>ATP {sport_title} ({surface})</b>\n"
                f"👤 <b>{p1} vs {p2}</b>\n\n"
                f"🎯 <b>Событие:</b> Тотал 1-го сета БОЛЬШЕ 9.5\n"
                f"📊 <b>Наша вероятность:</b> {p_total:.1%} (Справедливый кэф: {fair_odds:.2f})\n"
                f"🌪 <b>Погода:</b> {'Крытый корт 🏟' if surface in ['Indoor Hard', 'Carpet'] else (f'Ветер {wind_speed} км/ч' if wind_speed else 'Нет данных по ветру')}\n\n"
                f"🟢 <b>ИНСТРУКЦИЯ:</b>\n"
                f"Заходить строго от коэффициента <b>{min_target_odds:.2f}</b> и выше.\n"
                f"🔥 Текущий лучший кэф: <b>{best_odds:.2f}</b> ({bookie_name})"
            )
            
            # Send to Telegram (Broadcast to all subscribers or just ADMIN_ID)
            admin_id = os.getenv("ADMIN_ID")
            if admin_id:
                try:
                    await bot.send_message(admin_id, alert_text, parse_mode=ParseMode.HTML)
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
            
            # Save to PB
            try:
                await pb_client.save_signal({
                    "match": f"{p1} vs {p2}",
                    "predicted_prob": round(float(p_total), 4),
                    "fair_odds": round(float(fair_odds), 2),
                    "bookmaker_odds": float(best_odds),
                    "bookmaker_name": bookie_name,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"PB Save error: {e}")
            
            stats_funnel["alerts_sent"] += 1
            logger.info(f"ALERT SENT: {p1} vs {p2}")
    
    logger.info(f"Cycle complete. Total processed: {stats_funnel['total_matches']}")

async def main():
    bot = Bot(token=BOT_TOKEN)
    
    # Initialize PB
    await pb_client.authenticate()
    
    # Background task for scanning
    async def loop_scanner():
        while True:
            try:
                await scan_matches(bot)
            except Exception as e:
                logger.exception(f"Error in scanner loop: {e}")
            await asyncio.sleep(3600) # Scan every hour
    
    asyncio.create_task(loop_scanner())
    
    # Start polling
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
