# OT-Flow (Jax version)

This is my implementation of [OT-Flow](https://github.com/EmoryMLIP/OT-Flow) using the Jax-based [Equinox](https://docs.kidger.site/equinox/) and [Diffrax](https://docs.kidger.site/diffrax/) libraries. I've kept things mostly faithful to the original implementation, other than a few stylistic and package architecture changes. This is still a work in progress; questions, comments and PRs are welcome.

## References
1. Onken, D., Wu Fung, S., Li, X., & Ruthotto, L. (2021). <i>OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport</i>. Proceedings of the AAAI Conference on Artificial Intelligence, <b>35</b>(10), 9223-9232. https://doi.org/10.1609/aaai.v35i10.17113
2. Kidger, P. <i>On Neural Differential Equations</i>. Doctoral Dissertation. University of Oxford. 2021. URL: https://arxiv.org/abs/2202.02435. Accessed 07 March 2023.
3. Kidger, P., Garcia, C., <i>Equinox: neural networks in JAX via callable PyTrees and filtered transformations</i>. Differentiable Programming workshop at Neural Information Processing Systems, 2021. URL: https://arxiv.org/abs/2111.00254. Accessed 07 March 2023.