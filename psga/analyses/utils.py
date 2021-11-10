from .hrv import HRV
#from .flow import BB

def check_is_fitted(estimator, attributes, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.
    This utility is meant to be used internally by estimators themselves,
    typically in their own predict / transform methods.
    Parameters
    ----------
    estimator : estimator instance
        estimator instance for which the check is performed.
    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["raw", "hypnogram", ...], "coef_"``
    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    None
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """

    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])

    if not attrs:
        raise ValueError(msg % {'name': type(estimator).__name__})

def get_rpeaks(raw, hypno, ECG_chan, plot=False):
    hrv = HRV()
    hrv.fit(raw, hypno, ECG_chan=ECG_chan)
    scoring, _ = hrv.score(plot=plot)
    return scoring

#def get_breath_by_breath(raw, hypno, nasal_pressure):
#    bbyb = BB()
#    bbyb.fit(raw, hypno, nasal_pressure=nasal_pressure, flow_chan=None)
#    scoring, _ = bbyb.score()

#    return scoring[nasal_pressure]
