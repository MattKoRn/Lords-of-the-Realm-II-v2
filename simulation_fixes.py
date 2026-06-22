"""Runtime stability and save-migration fixes for manual_simulation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict

import manual_simulation as sim


_original_load_game = sim.load_game
_original_resolve_rivals = sim.resolve_rivals
_original_attack = sim.attack
_original_recruit = sim.recruit


def _safe_int(value, default=0, minimum=None, maximum=None):
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _safe_float(value, default=0.0, minimum=None, maximum=None):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _normalise_county(name, county):
    if not isinstance(county, sim.County):
        data = county if isinstance(county, dict) else {}
        allowed = set(asdict(sim.County(str(name))))
        data = {key: value for key, value in data.items() if key in allowed}
        data["name"] = str(data.get("name") or name)
        county = sim.county_from_dict(data)

    county.name = str(county.name or name)
    county.owner = str(county.owner or "player")
    county.population = _safe_int(county.population, 1200, 100)
    county.happiness = _safe_float(county.happiness, 70.0, 0.0, 100.0)
    county.health = _safe_float(county.health, 75.0, 0.0, 100.0)
    county.cattle = _safe_int(county.cattle, 0, 0)
    county.grain = _safe_int(county.grain, 0, 0)
    county.labor_agriculture = _safe_int(county.labor_agriculture, 80, 0, 100)
    county.beef_share = _safe_int(county.beef_share, 100, 0, 100)
    county.castle = str(county.castle or "none")
    county.castle_progress = _safe_int(county.castle_progress, 0, 0, 100)

    if county.ration not in sim.RATIONS:
        county.ration = "normal"
    if county.tax not in sim.TAXES:
        county.tax = "normal"

    fields = county.fields if isinstance(county.fields, dict) else {}
    county.fields = {
        "cattle": _safe_int(fields.get("cattle"), 4, 0),
        "grain": _safe_int(fields.get("grain"), 0, 0),
        "fallow": _safe_int(fields.get("fallow"), 2, 0),
    }
    if sum(county.fields.values()) == 0:
        county.fields["fallow"] = 6

    stocks = county.stocks if isinstance(county.stocks, dict) else {}
    county.stocks = {
        material: _safe_int(stocks.get(material), 0, 0)
        for material in sim.INDUSTRIES
    }

    industries = county.active_industries
    if not isinstance(industries, list):
        industries = []
    county.active_industries = list(dict.fromkeys(
        str(item).lower() for item in industries if str(item).lower() in sim.INDUSTRIES
    ))

    garrison = county.garrison if isinstance(county.garrison, dict) else {}
    county.garrison = {
        troop: _safe_int(garrison.get(troop), 0, 0)
        for troop in sim.TROOPS
    }
    return county


def normalise_game(game):
    game.year = _safe_int(getattr(game, "year", 1268), 1268, 1)
    game.season_index = _safe_int(getattr(game, "season_index", 0), 0) % len(sim.SEASONS)
    game.treasury = _safe_int(getattr(game, "treasury", 500), 500, 0)
    game.last_saved = _safe_float(getattr(game, "last_saved", time.time()), time.time(), 0.0)
    game.log = [str(item) for item in (game.log if isinstance(game.log, list) else [])][-100:]

    counties = game.counties if isinstance(game.counties, dict) else {}
    game.counties = {
        str(name): _normalise_county(name, county)
        for name, county in counties.items()
    }
    if not game.counties:
        game.counties = sim.new_game().counties
    if not any(county.owner == "player" for county in game.counties.values()):
        first = next(iter(game.counties.values()))
        first.owner = "player"
        game.log.append(f"Save recovery restored {first.name} to player control.")

    army = game.army if isinstance(game.army, dict) else {}
    game.army = {troop: _safe_int(army.get(troop), 0, 0) for troop in sim.TROOPS}

    opponents = game.opponents if isinstance(game.opponents, dict) else {}
    defaults = sim.new_game().opponents
    fixed_opponents = {}
    for name, default in defaults.items():
        noble = opponents.get(name, default)
        if not isinstance(noble, sim.Noble):
            values = asdict(default)
            if isinstance(noble, dict):
                values.update({key: value for key, value in noble.items() if key in values})
            noble = sim.Noble(**values)
        noble.counties = sum(1 for county in game.counties.values() if county.owner == name)
        noble.strength = _safe_int(noble.strength, default.strength, 0)
        noble.relation = _safe_int(noble.relation, 0, -100, 100)
        noble.alive = noble.counties > 0
        fixed_opponents[name] = noble
    game.opponents = fixed_opponents
    return game


def load_game():
    game = normalise_game(_original_load_game())
    # Commit the post-offline timestamp immediately so a crash cannot replay turns.
    try:
        save_game(game)
    except OSError:
        game.log.append("Warning: the save file could not be updated.")
    return game


def save_game(game):
    """Write saves atomically so interruption cannot leave truncated JSON."""
    game = normalise_game(game)
    game.last_saved = time.time()
    data = asdict(game)
    directory = os.path.dirname(os.path.abspath(sim.SAVE_FILE))
    os.makedirs(directory, exist_ok=True)
    handle = None
    temp_name = None
    try:
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
        )
        temp_name = handle.name
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temp_name, sim.SAVE_FILE)
    finally:
        if handle is not None:
            handle.close()
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)


def recruit(game, troop, amount):
    amount = _safe_int(amount, 0)
    if troop not in sim.TROOPS:
        return "Unknown troop type."
    if amount <= 0:
        return "Recruitment amount must be greater than zero."
    return _original_recruit(game, troop, amount)


def attack(game, target_name):
    wanted = str(target_name).strip().casefold()
    match = next((name for name in game.counties if name.casefold() == wanted), None)
    if match is None:
        return "Unknown county. Use the exact county name shown in status."
    result = _original_attack(game, match)
    normalise_game(game)
    return result


def resolve_rivals(game):
    # Re-evaluate available player counties for each noble so one county cannot
    # be selected again after being captured earlier in the same season.
    for noble in game.opponents.values():
        if not noble.alive:
            continue
        player_counties = [county for county in game.counties.values() if county.owner == "player"]
        if not player_counties:
            break
        temperament_bonus = {
            "aggressive": 30,
            "scheming": 15,
            "honourable": 10,
            "cautious": 0,
        }.get(noble.temperament, 0)
        noble.strength += sim.random.randint(8, 24) + temperament_bonus // 5
        if sim.random.random() >= 0.10 + temperament_bonus / 300:
            continue
        target = sim.random.choice(player_counties)
        defence = sim.troop_strength(target.garrison) + (200 if target.castle != "none" else 0)
        attack_power = int(noble.strength * sim.random.uniform(0.7, 1.2))
        if attack_power > defence and len(player_counties) > 1:
            target.owner = noble.name
            noble.counties += 1
            game.log.append(f"{noble.name} captured {target.name}.")
        else:
            noble.strength = max(20, noble.strength - defence // 4)
            game.log.append(f"{target.name} resisted an attack by {noble.name}.")


def apply():
    sim.load_game = load_game
    sim.save_game = save_game
    sim.recruit = recruit
    sim.attack = attack
    sim.resolve_rivals = resolve_rivals
