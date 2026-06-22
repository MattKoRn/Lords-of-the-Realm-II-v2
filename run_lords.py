#!/usr/bin/env python3
"""Safe launcher for Lords of the Realm II - Text Edition.

This repairs the missing Game.buildings initialization before starting the
existing curses interface. Existing save data is preserved.
"""

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


def _fixed_game_init(self):
    # load_game() references self.buildings, so it must exist first.
    self.buildings = {
        name: values.copy() for name, values in DEFAULT_BUILDINGS.items()
    }
    _original_init(self)

    # Keep loaded progress and add defaults for any missing building types.
    for name, values in DEFAULT_BUILDINGS.items():
        self.buildings.setdefault(name, values.copy())


game_module.Game.__init__ = _fixed_game_init


if __name__ == "__main__":
    print("Starting Lords of the Realm II - Text Edition...")
    print("Loading game and neural network...")
    curses.wrapper(game_module.main)
