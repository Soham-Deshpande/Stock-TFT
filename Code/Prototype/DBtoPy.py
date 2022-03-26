#----------------------------------------------------#
#
#   File       : Dbtopy.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: Extract data from the database
#
#
#
# ----------------------------------------------------#

import psycopg
import numpy as numpy
import pandas as pd

con2 =  psycopg.connect(" user=postgres")
print(con2)

cur = con2.cursor()
query = "SELECT * FROM ftse"

cur.execute(query)
data = list(cur.fetchall())

#sudo -u postgres psql

dataset = pd.DataFrame(data)

print(dataset)
print(dataset.head())



