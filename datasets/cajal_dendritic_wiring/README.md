# Ramón y Cajal Minimum-Wiring-Length & Conduction-Delay Dendritic Trees Dataset (`datasets/cajal_dendritic_wiring/`)

Reference dataset and empirical Ramón y Cajal Purkinje neuron dendritic arbor specimen for **MorphoEval Slot 058 (`caleo-cajal-dendritic-wiring`)**, implemented in `2007-caleo-cajal-dendritic-wiring.py` and formally verified in `morphoeval/lean/loops/Loop_058_cajal_dendritic_wiring.lean`.

## Primary Literature & DOIs

1. **Cuntz, H., Forstner, F., Borst, A., & Häusser, M. (2010).** One Rule to Grow Them All: A General Theory of Neuronal Dendritic Trees. *PLoS Computational Biology*, 6(8), e1000877. DOI: [`10.1371/journal.pcbi.1000877`](https://doi.org/10.1371/journal.pcbi.1000877)
2. **Cuntz, H., Borst, A., & Segev, I. (2007).** Optimization principles of dendritic structure. *Proceedings of the National Academy of Sciences*, 104(33), 13319–13324. DOI: [`10.1073/pnas.0704731104`](https://doi.org/10.1073/pnas.0704731104)
3. **Cuntz, H., Forstner, F., Haag, J., & Borst, A. (2008).** A Quantitative Description of Sensory Neuron Organization in *Drosophila*: From Individual Dendritic Trees to the Population. *PLoS Computational Biology*, 4(12), e1000251. DOI: [`10.1371/journal.pcbi.1000251`](https://doi.org/10.1371/journal.pcbi.1000251)
4. **Rall, W. (1959).** Branching dendritic trees and motoneuron membrane resistivity. *Experimental Neurology*, 1(5), 491–527. DOI: [`10.1016/0014-4886(59)90046-9`](https://doi.org/10.1016/0014-4886(59)90046-9)
5. **Wen, Q., & Chklovskii, D. B. (2008).** A Cost-Benefit Analysis of Neuronal Morphology. *Journal of Neurophysiology*, 99(5), 2320–2328. DOI: [`10.1152/jn.00280.2007`](https://doi.org/10.1152/jn.00280.2007)
6. **Ramón y Cajal, S. (1899).** *Textura del Sistema Nervioso del Hombre y de los Vertebrados*. Madrid: Nicolás Moya. DOI: [`10.5962/bhl.title.48637`](https://doi.org/10.5962/bhl.title.48637)

## Files & Integrity

- `scout_manifest.json`: Literature and Wikimedia Commons scout dossier for Slot 058.
- `cajal_dendritic_wiring_reference.json`: Reference parameters and quantitative benchmarks for Cuntz et al. (2010) greedy MST balancing factor $bf \in [0, 1]$ Pareto trade-off between total wiring length $W$ and total soma-to-synapse path length $P$, 2D ($W \propto N^{1/2} A^{1/2}$) and 3D ($W \propto N^{2/3} V^{1/3}$) scaling laws, Rall's $3/2$ diameter power law ($d_{\text{parent}}^{3/2} = \sum_k d_{\text{daughter},k}^{3/2}$), and empirical specimen morphometrics.
- `paper_citations.bib`: BibTeX citations with verified DOIs.
- `cajal_purkinje_dendritic_arbor_specimen.jpg`: Santiago Ramón y Cajal's (1899) Golgi-stained drawing of a human cerebellar Purkinje cell (`File:Cajal - a purkinje neuron from the human cerebellum.jpg`, Wikimedia Commons, Public Domain, 720×1024 px, SHA-256: `6255a396655b56c44c7be2d07b5b2d4fc3da21a25542a37`).
