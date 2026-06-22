#!/usr/bin/env python3
"""Small dependency-free regression suite for the seasonal simulation."""

import json
import os
import tempfile

import manual_simulation as sim
import simulation_fixes

simulation_fixes.apply()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_new_game_turns():
    game = simulation_fixes.normalise_game(sim.new_game())
    for _ in range(8):
        sim.automated_management(game)
        sim.resolve_season(game, offline=True)
    check(game.year == 1270, "Eight seasons should advance two years")
    check(any(county.owner == "player" for county in game.counties.values()), "Player must retain a county")


def test_recruit_validation():
    game = simulation_fixes.normalise_game(sim.new_game())
    before = dict(game.army)
    message = sim.recruit(game, "peasants", -10)
    check("greater than zero" in message, "Negative recruitment must be rejected")
    check(game.army == before, "Rejected recruitment must not change the army")


def test_case_insensitive_attack():
    game = simulation_fixes.normalise_game(sim.new_game())
    game.army["knights"] = 1000
    result = sim.attack(game, "kent")
    check("Unknown county" not in result, "County lookup should ignore case")


def test_malformed_state_recovery():
    game = sim.new_game()
    game.counties["Bedfordshire"].fields = None
    game.counties["Bedfordshire"].stocks = {"iron": "bad"}
    game.counties["Bedfordshire"].garrison = None
    game.army = {"knights": "7"}
    fixed = simulation_fixes.normalise_game(game)
    county = fixed.counties["Bedfordshire"]
    check(sum(county.fields.values()) > 0, "Fields must recover to a valid allocation")
    check(county.stocks["iron"] == 0, "Invalid stock values must recover safely")
    check(fixed.army["knights"] == 7, "Numeric strings should migrate")


def test_atomic_save():
    game = simulation_fixes.normalise_game(sim.new_game())
    old_save = sim.SAVE_FILE
    with tempfile.TemporaryDirectory() as directory:
        sim.SAVE_FILE = os.path.join(directory, "save.json")
        sim.save_game(game)
        with open(sim.SAVE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        check(data["year"] == 1268, "Atomic save should produce valid JSON")
        check(not any(name.endswith(".tmp") for name in os.listdir(directory)), "Temporary files should be cleaned")
    sim.SAVE_FILE = old_save


def main():
    tests = [
        test_new_game_turns,
        test_recruit_validation,
        test_case_insensitive_attack,
        test_malformed_state_recovery,
        test_atomic_save,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"All {len(tests)} smoke tests passed.")


if __name__ == "__main__":
    main()
