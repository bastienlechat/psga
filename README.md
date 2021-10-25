# Polysomnography analysis (psga)

This package is a collection of tools used at the Adelaide 
Institute for Sleep Health for the analysis of polysomnography files. 
Specifically, this packages can perform the following analyses:
- Spectral power analysis of EEG (sometimes refered as quantitative EEG)
- K-complexes detection (TBA)
- Analysis of pulse wave amplitude (and pulse arrival time) signals (TBA)
- R-peaks detection and heart rate variability analyses (TBA)
- Breaths detection and ventilation summary metrics (TBA)

---

## Getting started

PSGA is not yet available on pip/conda. Our tools were developed in Python 3.7 
and require the following dependencies: 
[MNE](https://mne.tools/stable/index.html), [pandas](https://pandas.pydata.org/)
and [scikit-learn](https://scikit-learn.org/stable/).

## Citation

If you find this code useful, please consider citing the following publication:

Lechat, B., Hansen, K. L., Melaku, Y. A., Vakulin, A., Micic, G., 
Adams, R. J., . . . Zajamsek, B. (2021). A Novel EEG Derived Measure of 
Disrupted Delta Wave Activity during Sleep Predicts All-Cause Mortality Risk. 
Ann Am Thorac Soc, (in press). doi:10.1513/AnnalsATS.202103-315OC


## Acknowledgments
Several functions were adapted or inspired from 
[MNE-features](https://mne.tools/mne-features/index.html) and 
[YASA](https://raphaelvallat.com/yasa/build/html/index.html). Credit should 
be given to functions/modules adapted from these packages. If credit is 
missing, please let us know.


