#!/usr/bin/env python3
import curses
import lords_of_realm as game_module

DEFAULT_BUILDINGS = {
    "farm": {"level": 1, "max_level": 100},
    "mine": {"level": 0, "max_level": 100},
    "quarry": {"level": 0, "max_level": 100},
    "iron_works": {"level": 0, "max_level": 100},
    "barracks": {"level": 0, "max_level": 100},
    "castle": {"level": 0, "max_level": 100},
    "market": {"level": 0, "max_level": 100},
    "academy": {"level": 0, "max_level": 100},
}

_original_init = game_module.Game.__init__


def fixed_init(self):
    self.buildings = {key: value.copy() for key, value in DEFAULT_BUILDINGS.items()}
    _original_init(self)

    for key, value in DEFAULT_BUILDINGS.items():
        self.buildings.setdefault(key, value.copy())

    if isinstance(self.provinces, list):
        provinces = {}
        for index, entry in enumerate(self.provinces, 1):
            if isinstance(entry, dict):
                data = entry.copy()
                name = str(data.pop("name", "Province " + str(index)))
            else:
                name = str(entry) if entry else "Province " + str(index)
                data = {}
            data.setdefault("owner", "player")
            data.setdefault("fortification", 1)
            provinces[name] = data
        self.provinces = provinces

    if not isinstance(self.provinces, dict) or not self.provinces:
        self.provinces = {"Home Province": {"owner": "player", "fortification": 1}}


game_module.Game.__init__ = fixed_init

if __name__ == "__main__":
    print("Starting Lords of the Realm II - Text Edition...")
    print("Loading game and neural network...")
    curses.wrapper(game_module.main)
