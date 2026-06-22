#!/usr/bin/env python3
"""Lords of the Realm II seasonal strategy simulation.

A text adaptation of the original game's county-management loop. Time advances
only when a season ends. Optional automation can manage the player's realm, and
offline progress resolves a capped number of automated seasons.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

SAVE_FILE = "lords_manual_save.json"
SEASONS = ("Spring", "Summer", "Autumn", "Winter")
RATIONS = ("none", "quarter", "half", "normal", "double", "triple")
TAXES = ("none", "low", "normal", "high", "excessive")
INDUSTRIES = ("wood", "stone", "iron", "weapons")
TROOPS = ("peasants", "archers", "macemen", "swordsmen", "knights", "pikemen")


@dataclass
class County:
    name: str
    owner: str = "player"
    population: int = 1200
    happiness: float = 70.0
    health: float = 75.0
    fields: Dict[str, int] = field(default_factory=lambda: {"cattle": 4, "grain": 0, "fallow": 2})
    cattle: int = 220
    grain: int = 0
    labor_agriculture: int = 80
    active_industries: List[str] = field(default_factory=lambda: ["wood"])
    stocks: Dict[str, int] = field(default_factory=lambda: {"wood": 80, "stone": 40, "iron": 20, "weapons": 20})
    ration: str = "normal"
    beef_share: int = 100
    tax: str = "normal"
    castle: str = "none"
    castle_progress: int = 0
    garrison: Dict[str, int] = field(default_factory=lambda: {name: 0 for name in TROOPS})


@dataclass
class Noble:
    name: str
    temperament: str
    counties: int = 1
    strength: int = 250
    relation: int = 0
    alive: bool = True


@dataclass
class GameState:
    ruler: str = "MattKoRn"
    year: int = 1268
    season_index: int = 0
    treasury: int = 500
    counties: Dict[str, County] = field(default_factory=dict)
    army: Dict[str, int] = field(default_factory=lambda: {name: 0 for name in TROOPS})
    opponents: Dict[str, Noble] = field(default_factory=dict)
    automation: bool = False
    difficulty: str = "normal"
    last_saved: float = field(default_factory=time.time)
    log: List[str] = field(default_factory=list)

    @property
    def season(self) -> str:
        return SEASONS[self.season_index]

    @property
    def home(self) -> County:
        return next(county for county in self.counties.values() if county.owner == "player")


def new_game() -> GameState:
    game = GameState()
    game.counties = {
        "Bedfordshire": County("Bedfordshire"),
        "Gwynedd": County("Gwynedd", owner="Bishop", population=950, happiness=62),
        "Kent": County("Kent", owner="Baron", population=1100, happiness=58),
        "Yorkshire": County("Yorkshire", owner="Knight", population=1000, happiness=66),
        "Cornwall": County("Cornwall", owner="Countess", population=900, happiness=72),
    }
    game.opponents = {
        "Bishop": Noble("Bishop", "scheming", strength=260),
        "Baron": Noble("Baron", "aggressive", strength=310),
        "Knight": Noble("Knight", "honourable", strength=280),
        "Countess": Noble("Countess", "cautious", strength=240),
    }
    game.log.append("The throne is empty. Unite the realm and defeat every rival noble.")
    return game


def county_from_dict(data: dict) -> County:
    defaults = asdict(County(data.get("name", "County")))
    defaults.update(data)
    return County(**defaults)


def load_game() -> GameState:
    if not os.path.exists(SAVE_FILE):
        return new_game()
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        game = GameState(
            ruler=data.get("ruler", "MattKoRn"),
            year=int(data.get("year", 1268)),
            season_index=int(data.get("season_index", 0)) % 4,
            treasury=int(data.get("treasury", 500)),
            army={name: int(data.get("army", {}).get(name, 0)) for name in TROOPS},
            automation=bool(data.get("automation", False)),
            difficulty=data.get("difficulty", "normal"),
            last_saved=float(data.get("last_saved", time.time())),
            log=list(data.get("log", []))[-100:],
        )
        game.counties = {name: county_from_dict(value) for name, value in data.get("counties", {}).items()}
        if not game.counties:
            game.counties = new_game().counties
        base_opponents = new_game().opponents
        game.opponents = {}
        for name, default in base_opponents.items():
            values = asdict(default)
            values.update(data.get("opponents", {}).get(name, {}))
            game.opponents[name] = Noble(**values)
        apply_offline_progress(game)
        return game
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return new_game()


def save_game(game: GameState) -> None:
    game.last_saved = time.time()
    data = asdict(game)
    with open(SAVE_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def apply_offline_progress(game: GameState) -> None:
    elapsed = max(0, time.time() - game.last_saved)
    # One simulated season per six offline hours, capped at four seasons.
    seasons = min(4, int(elapsed // 21600))
    if seasons <= 0:
        return
    old_mode = game.automation
    game.automation = True
    for _ in range(seasons):
        automated_management(game)
        resolve_season(game, offline=True)
    game.automation = old_mode
    game.log.append(f"Offline stewardship resolved {seasons} season(s).")


def production_preview(county: County) -> Dict[str, int]:
    workers = max(0, int(county.population * 0.55))
    agriculture_workers = int(workers * county.labor_agriculture / 100)
    industry_workers = workers - agriculture_workers
    total_fields = max(1, sum(county.fields.values()))
    cattle_fields = county.fields.get("cattle", 0)
    grain_fields = county.fields.get("grain", 0)
    cattle_growth = int(cattle_fields * agriculture_workers / total_fields / 90)
    grain_gain = int(grain_fields * agriculture_workers / total_fields / 12)
    output = {"cattle": cattle_growth, "grain": grain_gain, "wood": 0, "stone": 0, "iron": 0, "weapons": 0}
    active = [name for name in county.active_industries if name in INDUSTRIES]
    if active:
        per_site = industry_workers / len(active)
        rates = {"wood": 0.18, "stone": 0.10, "iron": 0.07, "weapons": 0.035}
        for name in active:
            output[name] = int(per_site * rates[name])
    if SEASONS[0] == "Spring":
        pass
    return output


def food_required(county: County) -> int:
    multiplier = {"none": 0.0, "quarter": 0.25, "half": 0.5, "normal": 1.0, "double": 2.0, "triple": 3.0}[county.ration]
    return int(county.population * multiplier)


def resolve_county(county: County, game: GameState) -> None:
    output = production_preview(county)
    county.cattle = max(0, county.cattle + output["cattle"])
    county.grain += output["grain"]
    for material in INDUSTRIES:
        county.stocks[material] = max(0, county.stocks.get(material, 0) + output[material])

    demand = food_required(county)
    dairy = min(demand, county.cattle * 3)
    remaining = demand - dairy
    beef_target = int(remaining * county.beef_share / 100)
    cows_slaughtered = min(county.cattle, (beef_target + 7) // 8)
    beef = cows_slaughtered * 8
    county.cattle -= cows_slaughtered
    grain_used = min(county.grain, max(0, remaining - beef))
    county.grain -= grain_used
    fed = dairy + beef + grain_used
    food_ratio = 1.0 if demand == 0 else min(1.0, fed / demand)

    tax_delta = {"none": 4, "low": 2, "normal": 0, "high": -4, "excessive": -9}[county.tax]
    ration_delta = {"none": -12, "quarter": -8, "half": -4, "normal": 1, "double": 4, "triple": 6}[county.ration]
    shortage_delta = int((food_ratio - 1.0) * 30)
    county.happiness = max(0, min(100, county.happiness + tax_delta + ration_delta + shortage_delta))
    county.health = max(0, min(100, county.health + (food_ratio - 0.75) * 12))

    growth_rate = (county.happiness - 50) / 1000 + (county.health - 50) / 1500
    if food_ratio < 0.75:
        growth_rate -= 0.04
    county.population = max(100, int(county.population * (1 + growth_rate)))

    tax_rate = {"none": 0, "low": 0.03, "normal": 0.06, "high": 0.10, "excessive": 0.16}[county.tax]
    game.treasury += int(county.population * tax_rate)

    if county.happiness < 15 and random.random() < 0.25:
        loss = min(county.population // 8, 150)
        county.population -= loss
        game.log.append(f"Peasants revolted in {county.name}; {loss} people were lost.")


def troop_strength(army: Dict[str, int]) -> int:
    values = {"peasants": 1, "archers": 3, "macemen": 3, "swordsmen": 4, "knights": 8, "pikemen": 4}
    return sum(army.get(name, 0) * values[name] for name in TROOPS)


def resolve_rivals(game: GameState) -> None:
    player_counties = [c for c in game.counties.values() if c.owner == "player"]
    for noble in game.opponents.values():
        if not noble.alive:
            continue
        temperament_bonus = {"aggressive": 30, "scheming": 15, "honourable": 10, "cautious": 0}[noble.temperament]
        noble.strength += random.randint(8, 24) + temperament_bonus // 5
        if random.random() < 0.10 + temperament_bonus / 300 and player_counties:
            target = random.choice(player_counties)
            defence = troop_strength(target.garrison) + (200 if target.castle != "none" else 0)
            attack = int(noble.strength * random.uniform(0.7, 1.2))
            if attack > defence and len(player_counties) > 1:
                target.owner = noble.name
                noble.counties += 1
                game.log.append(f"{noble.name} captured {target.name}.")
            else:
                noble.strength = max(20, noble.strength - defence // 4)
                game.log.append(f"{target.name} resisted an attack by {noble.name}.")


def resolve_season(game: GameState, offline: bool = False) -> None:
    for county in game.counties.values():
        if county.owner == "player":
            resolve_county(county, game)
    wages = sum(game.army.values()) * 2
    game.treasury -= wages
    if game.treasury < 0:
        game.log.append("The treasury cannot meet army wages; desertions follow.")
        for troop in TROOPS:
            game.army[troop] = int(game.army[troop] * 0.9)
        game.treasury = 0
    resolve_rivals(game)
    old = game.season
    game.season_index = (game.season_index + 1) % 4
    if game.season_index == 0:
        game.year += 1
    game.log.append(f"{old} ended. It is now {game.season}, {game.year}.")
    if not offline:
        save_game(game)


def automated_management(game: GameState) -> None:
    for county in game.counties.values():
        if county.owner != "player":
            continue
        projected_food = county.cattle * 3 + county.grain
        need = max(1, county.population)
        if projected_food < need * 2:
            county.labor_agriculture = min(95, county.labor_agriculture + 10)
            county.ration = "half" if projected_food < need else "normal"
            county.tax = "low" if county.happiness < 55 else "normal"
        else:
            county.labor_agriculture = max(45, county.labor_agriculture - 5)
            county.ration = "normal"
            county.tax = "normal" if county.happiness < 75 else "high"
        industry_workers = 100 - county.labor_agriculture
        if industry_workers < 15:
            county.active_industries = ["wood"]
        elif county.stocks["iron"] < 80:
            county.active_industries = ["wood", "iron"]
        else:
            county.active_industries = ["wood", "stone", "weapons"]


def recruit(game: GameState, troop: str, amount: int) -> str:
    costs = {"peasants": 3, "archers": 12, "macemen": 10, "swordsmen": 16, "knights": 40, "pikemen": 14}
    weapon_cost = {"peasants": 0, "archers": 1, "macemen": 1, "swordsmen": 1, "knights": 2, "pikemen": 1}
    county = game.home
    amount = max(0, min(amount, county.population // 10))
    gold = costs[troop] * amount
    weapons = weapon_cost[troop] * amount
    if game.treasury < gold or county.stocks["weapons"] < weapons:
        return "Not enough gold or weapons."
    game.treasury -= gold
    county.stocks["weapons"] -= weapons
    county.population -= amount
    game.army[troop] += amount
    county.happiness = max(0, county.happiness - amount / 100)
    return f"Recruited {amount} {troop}."


def attack(game: GameState, target_name: str) -> str:
    target = game.counties.get(target_name)
    if not target or target.owner == "player":
        return "Invalid enemy county."
    noble = game.opponents.get(target.owner)
    player_power = troop_strength(game.army) * random.uniform(0.75, 1.25)
    enemy_power = (noble.strength if noble else 200) * random.uniform(0.75, 1.25)
    if target.castle != "none":
        enemy_power *= 1.5
    if player_power <= 0:
        return "You have no army."
    if player_power > enemy_power:
        casualties = min(0.55, enemy_power / player_power * 0.45)
        for troop in TROOPS:
            game.army[troop] = int(game.army[troop] * (1 - casualties))
        target.owner = "player"
        target.happiness = 40
        if noble:
            noble.counties -= 1
            noble.strength = max(0, noble.strength // 2)
            if noble.counties <= 0:
                noble.alive = False
                game.log.append(f"{noble.name} has been eliminated.")
        return f"Victory! {target.name} is now yours."
    casualties = min(0.8, enemy_power / max(1, player_power) * 0.35)
    for troop in TROOPS:
        game.army[troop] = int(game.army[troop] * (1 - casualties))
    return f"Defeat at {target.name}. Your surviving army withdrew."


def status(game: GameState) -> None:
    print("\n" + "=" * 72)
    print(f"{game.ruler}'s Realm | {game.season} {game.year} | Treasury: {game.treasury}")
    print(f"Automation: {'ON' if game.automation else 'OFF'} | Army strength: {troop_strength(game.army)}")
    print("-" * 72)
    for county in game.counties.values():
        marker = "*" if county.owner == "player" else " "
        print(f"{marker} {county.name:16} owner={county.owner:9} pop={county.population:5} "
              f"happy={county.happiness:5.1f} health={county.health:5.1f}")
    print("-" * 72)
    county = game.home
    preview = production_preview(county)
    print(f"Home: {county.name} | cattle={county.cattle} grain={county.grain} fields={county.fields}")
    print(f"Labor: {county.labor_agriculture}% agriculture / {100-county.labor_agriculture}% industry")
    print(f"Industry: {', '.join(county.active_industries) or 'none'} | Stocks: {county.stocks}")
    print(f"Rations: {county.ration} | Tax: {county.tax} | Next-season output: {preview}")
    print(f"Army: {game.army}")
    if game.log:
        print("Recent events:")
        for entry in game.log[-5:]:
            print("  -", entry)
    print("=" * 72)


def help_text() -> None:
    print("""
Commands
  status                         Show the realm and next-season forecast
  labor <0-100>                  Set agriculture share; remainder is industry
  fields <cattle> <grain> <fallow>
  industry <wood,stone,iron,weapons|none>
  ration <none|quarter|half|normal|double|triple>
  tax <none|low|normal|high|excessive>
  recruit <troop> <amount>       Draft and equip soldiers
  attack <county name>           Resolve a field battle/county assault
  automate                       Toggle AI stewardship
  end                            End the season and resolve all consequences
  save                           Save immediately
  help                           Show commands
  quit                           Save and leave
""")


def command_loop(game: GameState) -> None:
    help_text()
    while True:
        status(game)
        if all(not noble.alive for noble in game.opponents.values()):
            print("\nYou have defeated every challenger. The crown is yours!")
            save_game(game)
            return
        try:
            parts = input("\nYour command> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            parts = ["quit"]
        if not parts:
            continue
        command = parts[0].lower()
        county = game.home
        try:
            if command == "labor" and len(parts) == 2:
                county.labor_agriculture = max(0, min(100, int(parts[1])))
            elif command == "fields" and len(parts) == 4:
                values = [max(0, int(value)) for value in parts[1:]]
                if sum(values) != sum(county.fields.values()):
                    print("Field total must remain", sum(county.fields.values()))
                else:
                    county.fields = dict(zip(("cattle", "grain", "fallow"), values))
            elif command == "industry" and len(parts) == 2:
                names = [] if parts[1] == "none" else parts[1].split(",")
                if any(name not in INDUSTRIES for name in names):
                    print("Unknown industry.")
                else:
                    county.active_industries = names
            elif command == "ration" and len(parts) == 2 and parts[1] in RATIONS:
                county.ration = parts[1]
            elif command == "tax" and len(parts) == 2 and parts[1] in TAXES:
                county.tax = parts[1]
            elif command == "recruit" and len(parts) == 3 and parts[1] in TROOPS:
                print(recruit(game, parts[1], int(parts[2])))
            elif command == "attack" and len(parts) >= 2:
                print(attack(game, " ".join(parts[1:])))
            elif command == "automate":
                game.automation = not game.automation
                print("Automation", "enabled." if game.automation else "disabled.")
            elif command == "end":
                if game.automation:
                    automated_management(game)
                resolve_season(game)
            elif command == "save":
                save_game(game)
                print("Game saved.")
            elif command == "status":
                pass
            elif command == "help":
                help_text()
            elif command in ("quit", "exit"):
                save_game(game)
                return
            else:
                print("Unknown or incomplete command. Type help.")
        except ValueError:
            print("That command requires a valid number.")


def main() -> None:
    print("Lords of the Realm II - Manual Simulation")
    print("Seasonal strategy mode: no real-time idle resource generation.")
    game = load_game()
    command_loop(game)


if __name__ == "__main__":
    main()
