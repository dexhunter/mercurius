Yet Another Portfolio Management Toolbox.

IMPORTANT: **Live Trading at your own risk.**

## Installation

```
pip install mercurius
```

## Example Usage

```
# Uniform Constant Rebalanced Portfolio

from mercurius.strategy import ucrp

up = ucrp()
up.trade(input_data, tc=0.025)
result = up.finish(True, True)
print(result['portfolio'])
```

## Features

* Data Augmentation
    * Concat different assets ohlcv
    * Concat different assets ticker
* Backtest
* Trading algorithms [full list](docs/algorithms.md)


## Contributing

TODOs (Check [ISSUES](https://github.com/dexhunter/mercurius/issues))

---

Disclaimer

This project is based on OLPS<sup>1</sup> and PGPortfolio<sup>2</sup>. Many thanks to the authors.

1. https://github.com/olps/olps
2. https://github.com/zhengyaoJiang/PGPortfolio
