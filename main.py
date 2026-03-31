import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from bs4 import BeautifulSoup
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
TOUR_AVG_RETURN = 0.22 

# --- Parsing Logic (TennisAbstract On-Demand) ---

def player_to_url(name: str) -> str:
    """
    Normalizes player name from Odds API to TennisAbstract URL format.
    Example: 'C. Alcaraz Garfia' -> 'https://www.tennisabstract.com/cgi-bin/player.cgi?p=CAlcaraz'
    or 'Daniil Medvedev' -> 'https://www.tennisabstract.com/cgi-bin/player.cgi?p=DaniilMedvedev'
    """
    # 1. Убираем точки и лишние пробелы
    clean_name = name.replace('.', '').strip()
    
    # 2. Разбиваем имя на части
    parts = clean_name.split()
    
    # 3. Берем первую часть (имя/инициал) и последнюю (основная фамилия)
    if len(parts) >= 2:
        first_name = parts[0].title()
        last_name = parts[-1].title()
        camel_case = "".join(filter(str.isalnum, f"{first_name}{last_name}"))
    else:
        camel_case = "".join(filter(str.isalnum, clean_name.title()))
        
    return f"https://www.tennisabstract.com/cgi-bin/player.cgi?p={camel_case}"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def fetch_player_stats(player_name: str, target_surface: str) -> Tuple[float, float, int]:
    url = player_to_url(player_name)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
    
    logger.info(f"Scraping stats for {player_name}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200: return 0.0, 0.0, 0
                html = await resp.text()
    except Exception as e:
        logger.error(f"Network error for {player_name}: {e}")
        return 0.0, 0.0, 0

    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', {'id': 'matches'})
    if not table: return 0.0, 0.0, 0

    holds, returns, weights = [], [], []
    now = datetime.now()
    cutoff_52_weeks = now - timedelta(days=365)
    
    rows = table.find_all('tr')[1:]
    
    # Сначала найдем дату последнего матча для динамического WMA
    all_match_dates = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 15: continue
        try:
            m_date = datetime.strptime(cells[0].text.strip(), '%Y-%m-%d')
            all_match_dates.append(m_date)
        except: continue
    
    if not all_match_dates: return 0.0, 0.0, 0
    latest_match_date = max(all_match_dates)
    cutoff_60_days = latest_match_date - timedelta(days=60)

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 15: continue
        try:
            match_date = datetime.strptime(cells[0].text.strip(), '%Y-%m-%d')
            if match_date < cutoff_52_weeks: break 

            surface = cells[4].text.strip()
            norm_surface = 'Indoor Hard' if surface == 'I.Hard' else surface
            
            if norm_surface != target_surface:
                if not (target_surface == 'Indoor Hard' and surface == 'I.Hard'): continue

            svw_str = cells[13].text.strip().replace('%', '')
            rtw_str = cells[14].text.strip().replace('%', '')
            if not svw_str or not rtw_str: continue
            
            svw, rtw = float(svw_str) / 100.0, float(rtw_str) / 100.0
            weight = 2 if match_date >= cutoff_60_days else 1
            
            holds.append(svw); returns.append(rtw); weights.append(weight)
        except: continue

    if len(holds) < 15: return 0.0, 0.0, len(holds)
    return np.average(holds, weights=weights), np.average(returns, weights=weights), len(holds)

# --- API Integrations ---

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def get_weather(lat: float, lon: float, start_time: datetime) -> Optional[float]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m", "timezone": "auto", "start_date": start_time.strftime("%Y-%m-%d"), "end_date": start_time.strftime("%Y-%m-%d")}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                hour_idx = start_time.hour
                if "hourly" in data and "wind_speed_10m" in data["hourly"]:
                    return data["hourly"]["wind_speed_10m"][hour_idx]
    return None

async def get_active_atp_sports() -> List[str]:
    url = "https://api.the-odds-api.com/v4/sports"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params={"apiKey": ODDS_API_KEY}) as resp:
            if resp.status == 200:
                all_sports = await resp.json()
                return [s['key'] for s in all_sports if s['key'].startswith('tennis_atp') and s['active']]
    return []

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def get_odds(sport_slug: str) -> List[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_slug}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "alternate_total_games_1st_set,totals_1st_set,totals,h2h", "oddsFormat": "decimal"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200: return await resp.json()
            logger.error(f"Odds API Error {sport_slug}: {resp.status}")
    return []

def calculate_match_probability(hold_a: float, hold_b: float) -> float:
    p_a_5_5 = binom.pmf(5, 5, hold_a); p_a_4_5 = binom.pmf(4, 5, hold_a)
    p_b_5_5 = binom.pmf(5, 5, hold_b); p_b_4_5 = binom.pmf(4, 5, hold_b)
    return (p_a_5_5 * p_b_5_5) + (p_a_4_5 * p_b_4_5)

# --- PocketBase ---

class PocketBaseClient:
    def __init__(self, url, email, password):
        self.url, self.email, self.password, self.token = url, email, password, None

    async def authenticate(self):
        async with aiohttp.ClientSession() as session:
            payload = {"identity": self.email, "password": self.password}
            async with session.post(f"{self.url}/api/admins/auth-with-password", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json(); self.token = data["token"]
                    logger.info("PB Authenticated successfully")

    async def save_signal(self, data: dict):
        if not self.token: await self.authenticate()
        headers = {"Authorization": f"Bearer {self.token}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/api/collections/signals/records", json=data, headers=headers) as resp:
                if resp.status != 200: logger.error(f"PB Save fail: {await resp.text()}")

pb_client = PocketBaseClient(PB_URL, PB_EMAIL, PB_PASSWORD)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("🎾 Теннисный бот (Scraping mode) запущен.\nИспользуйте /status.")

@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    conn_status = "✅ OK" if pb_client.token else "❌ No Auth"
    text = (f"📊 <b>Статистика воронки:</b>\n- Проверено матчей: {stats_funnel['total_matches']}\n- Отсеяно по покрытию: {stats_funnel['skipped_surface']}\n"
            f"- Отсеяно по холду: {stats_funnel['skipped_hold']}\n- Отсеяно по погоде: {stats_funnel['skipped_weather']}\n- Выдано алертов: {stats_funnel['alerts_sent']}\n\n🔗 <b>PB Connection:</b> {conn_status}")
    await message.answer(text, parse_mode=ParseMode.HTML)

# --- Main Engine ---

TOURNAMENT_LOCATIONS = {"Australian Open": (37.8136, 144.9631), "Wimbledon": (51.4343, 0.2145), "French Open": (48.8475, 2.2492), "US Open": (40.7500, -73.8467), "Indian Wells": (33.7233, -116.3750), "Miami": (25.9581, -80.2389), "Monte Carlo": (43.7483, 7.4331), "Madrid": (40.4168, -3.7038), "Rome": (41.9028, 12.4964)}
SURFACE_MAPPING = {"Australian Open": "Hard", "Wimbledon": "Grass", "French Open": "Clay", "US Open": "Hard", "Indian Wells": "Hard", "Miami": "Hard", "Monte Carlo": "Clay", "Madrid": "Clay", "Rome": "Clay"}

async def scan_matches(bot: Bot):
    logger.info("Starting scan cycle...")
    active_sports = await get_active_atp_sports()
    if not active_sports: return

    for sport_slug in active_sports:
        odds_data = await get_odds(sport_slug)
        if not odds_data: continue

        for match in odds_data:
            stats_funnel["total_matches"] += 1
            sport_title = match.get('sport_title', 'Unknown')
            surface = "Hard"
            for name, s in SURFACE_MAPPING.items():
                if name.lower() in sport_title.lower(): surface = s; break
            
            if surface not in ["Hard", "Grass", "Indoor Hard", "Carpet"]:
                stats_funnel["skipped_surface"] += 1; continue
                
            p1, p2 = match['home_team'], match['away_team']
            h1, r1, count1 = await fetch_player_stats(p1, surface)
            h2, r2, count2 = await fetch_player_stats(p2, surface)
            
            if count1 < 15 or count2 < 15:
                stats_funnel["skipped_hold"] += 1; continue
                
            exp_hold_1 = h1 - (r2 - TOUR_AVG_RETURN)
            exp_hold_2 = h2 - (r1 - TOUR_AVG_RETURN)
            
            if exp_hold_1 < 0.85 or exp_hold_2 < 0.85:
                stats_funnel["skipped_hold"] += 1; continue
                
            wind_speed = 0
            if surface in ["Hard", "Grass"]:
                lat, lon = 0.0, 0.0
                for name, coords in TOURNAMENT_LOCATIONS.items():
                    if name.lower() in sport_title.lower(): lat, lon = coords; break
                if lat != 0.0:
                    match_time = datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00'))
                    wind_speed = await get_weather(lat, lon, match_time)
                    if wind_speed and wind_speed > 20:
                        stats_funnel["skipped_weather"] += 1; continue

            p_total = calculate_match_probability(exp_hold_1, exp_hold_2)
            fair_odds = 1 / p_total if p_total > 0 else 999
            min_target_odds = fair_odds * 1.05
            
            best_odds, bookie_name = 0, ""
            possible_keys = ['alternate_total_games_1st_set', 'total_games_1st_set', 'totals_1st_set', 'totals']
            for bookmaker in match.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] in possible_keys:
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over' and outcome.get('point') == 9.5:
                                if outcome['price'] > best_odds:
                                    best_odds = outcome['price']; bookie_name = bookmaker['title']

            if best_odds >= min_target_odds:
                alert_text = (f"🎾 <b>{sport_title} ({surface})</b>\n👤 <b>{p1} vs {p2}</b>\n\n🎯 <b>Событие:</b> Тотал 1-го сета БОЛЬШЕ 9.5\n"
                            f"📊 <b>Наша вероятность:</b> {p_total:.1%} (Справедливый кэф: {fair_odds:.2f})\n"
                            f"🌪 <b>Погода:</b> {'Крытый корт 🏟' if surface in ['Indoor Hard', 'Carpet'] else (f'Ветер {wind_speed} км/ч' if wind_speed else 'Нет данных по ветру')}\n\n"
                            f"🟢 <b>ИНСТРУКЦИЯ:</b>\nЗаходить строго от коэффициента <b>{min_target_odds:.2f}</b> и выше.\n🔥 Текущий лучший кэф: <b>{best_odds:.2f}</b> ({bookie_name})")
                
                admin_id = os.getenv("ADMIN_ID")
                if admin_id:
                    try: await bot.send_message(admin_id, alert_text, parse_mode=ParseMode.HTML)
                    except: pass
                
                await pb_client.save_signal({"match": f"{p1} vs {p2}", "predicted_prob": round(float(p_total), 4), "fair_odds": round(float(fair_odds), 2), "bookmaker_odds": float(best_odds), "bookmaker_name": bookie_name, "timestamp": datetime.now().isoformat()})
                stats_funnel["alerts_sent"] += 1

async def main():
    bot = Bot(token=BOT_TOKEN)
    await pb_client.authenticate()
    async def loop_scanner():
        while True:
            try: await scan_matches(bot)
            except Exception as e: logger.exception(f"Error: {e}")
            await asyncio.sleep(3600)
    asyncio.create_task(loop_scanner())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
