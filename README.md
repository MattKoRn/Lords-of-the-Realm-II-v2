# Neural Lords of the Realm

A self-contained browser strategy prototype inspired by the seasonal realm-management loop of *Lords of the Realm II*.

## Play

Open `index.html` in a modern browser. No build step, server, dependencies, or network connection are required.

## Included systems

- Seasonal turns from Spring 1268 onward
- Population, happiness, food, cattle, grain, taxation, and treasury
- Labour allocation between farming and industry
- Wood, stone, iron, and weapon production
- Castle construction
- Army recruitment and county conquest
- Rival counties that develop and raid independently
- An online-learning neural council that recommends economic, industrial, or military policy
- A separate neural policy network used by rival lords

## Neural-network design

The game contains two small multilayer perceptrons implemented directly in JavaScript:

1. **Royal Neural Council** — reads eight normalized realm signals and predicts one of three policies: food/happiness, industrial growth, or military expansion. It trains after every season using the outcome of the player's decisions.
2. **Rival Lord Network** — reads each rival county's strategic condition and chooses growth, mobilisation, or aggression. It updates online as the campaign proceeds.

This is intentionally dependency-free and transparent: the network implementation, forward pass, and backpropagation are all visible in `index.html`.

## Original-game-inspired loop

The prototype focuses on the broad systems described in the original manual: county stewardship, labour allocation, agriculture, industry, happiness, taxation, castle building, armies, conquest, and seasonal resolution. It does not use original game assets or source code.
