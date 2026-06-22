#!/usr/bin/env python3
"""Legacy launcher retained for compatibility.

The original curses/idle implementation has been replaced by the seasonal
manual simulation. Running this old filename now starts the supported game.
"""

import manual_simulation
import simulation_fixes

simulation_fixes.apply()

if __name__ == "__main__":
    manual_simulation.main()
