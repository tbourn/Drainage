# Drainage
- Extracts data (time, drainage rate, relative humidity) from xlsx file and performs curve fitting on them for a given function .
- Returns the best fit values for parameters of the given model.
- Returns root mean squared error (used as non Linear regression metric).
- Returns negative gradient values, their total sum and their minimum value.
- Returns the given model, its first derivative and saves their plot as png.

Subfolder **Data** includes data in xlsx format about the product.
- **pls_bag.py**

## Prerequisites
- [Python 3.6.3 or later version](https://www.python.org/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [SymPy](http://www.sympy.org/en/index.html)
- [xlrd](https://pypi.python.org/pypi/xlrd)
