# optima-labs

This project contains KAU Nonlinear Programming course labs and solutions. The commands below are applicable for
Unix-like OS, if you have Windows — use appropriate utilities to run [pdm](https://pdm.fming.dev/latest/).

## Setup

Clone this repository, then run pdm to download required packages:

```shell
pdm sync
```

## Usage

Run this module to compare ralgb5a and emshor algorithms:

```shell
pdm run python -m optimalabs.comparison
```

Execute this file to see the difference between various solvers of the enterprise resource planning problem:

```shell
pdm run python -m optimalabs.planning
```
