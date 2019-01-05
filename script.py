import pandas as pd

data = pd.read_csv('test.csv', error_bad_lines=False, sep=';')
print(data)