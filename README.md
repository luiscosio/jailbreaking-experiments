
# jailbreaking-experiments

This repository is a playground for jailbreaking experiments.

Intially, we are playing with the GCG (Greedy Coordinate Gradient) algorithm.
 
## Installation

```
git clone https://github.com/luiscosio/jailbreaking-experiments.git
cd nanoGCG
pip install -e .
```

## Overview

The GCG algorithm was introduced in [Universal and Transferrable Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043) [1] by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, Zico Kolter, and Matt Fredrikson.
This repository is based on the work of:

```
@misc{zou2023universal,
    title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
    author={Andy Zou and Zifan Wang and Nicholas Carlini and Milad Nasr and J. Zico Kolter and Matt Fredrikson},
    year={2023},
    eprint={2307.15043},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## Usage

See the `examples` folder.

## License

This repository is licensed under the MIT license.

## References and Citation

```
[1] https://arxiv.org/pdf/2307.15043
[2] https://blog.haizelabs.com/posts/acg
[3] https://arxiv.org/pdf/2402.12329
[4] https://confirmlabs.org/posts/TDC2023
[5] https://arxiv.org/pdf/1612.05628
```