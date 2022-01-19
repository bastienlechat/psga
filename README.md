# Polysomnography analysis (psga)

This package is a collection of tools used at the Adelaide 
Institute for Sleep Health for the analysis of polysomnography files.

**WARNING**: PSGA is still a work in progress, use at your own risk. Consider using [YASA](https://raphaelvallat.com/yasa/build/html/index.html) for a more stable alternative.

---

## Getting started

PSGA is not yet available on pip/conda. Our tools were developed in Python 3.7 
and require the following dependencies: 
[MNE](https://mne.tools/stable/index.html), [pandas](https://pandas.pydata.org/)
and [scikit-learn](https://scikit-learn.org/stable/).

## User-interface (TBA)

PSGA comes with a user-friendly dashboard to quickly visualise your data. 
The dashboard additionaly requires 
[Pyside2](https://wiki.qt.io/Qt_for_Python) and 
[plotly](https://plotly.com/). The dashboard is not designed for manual 
scoring of events. If you are after an UI to score sleep, consider using 
[visbrain-sleep](http://visbrain.org/sleep.html).


## Citation

If you find this code useful, please consider citing the following publication:

Lechat, B., Hansen, K. L., Melaku, Y. A., Vakulin, A., Micic, G., 
Adams, R. J., . . . Zajamsek, B. (2021). A Novel EEG Derived Measure of 
Disrupted Delta Wave Activity during Sleep Predicts All-Cause Mortality Risk. 
Ann Am Thorac Soc, (in press). 
[doi:10.1513/AnnalsATS.202103-315OC](https://doi.org/10.1513/AnnalsATS.202103-315OC)


## Acknowledgments
Several functions were adapted or inspired from 
[MNE-features](https://mne.tools/mne-features/index.html) and 
[YASA](https://raphaelvallat.com/yasa/build/html/index.html). Credit should 
be given to functions/modules adapted from these packages. If credit is 
missing, please let us know.


