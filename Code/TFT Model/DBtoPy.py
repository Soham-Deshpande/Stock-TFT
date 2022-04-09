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



def DBExtraction():
    """
    Database Extraction

    Simple query to access the stock data
    Function not used as CSV is more efficient

    Design:




                  Table "public.ftse"
     Column |     Type     | Collation | Nullable | Default
    --------+--------------+-----------+----------+---------
    date   | date         |           | not null |
    open   | numeric(7,2) |           | not null |
    high   | numeric(7,2) |           | not null |
    low    | numeric(7,2) |           | not null |
    close  | numeric(7,2) |           | not null |

    Indexes:
        "ftse_pkey" PRIMARY KEY, btree (date)

    """
    con2 =  psycopg.connect(" user=postgres")
    print(con2)

    cur = con2.cursor()
    query = "SELECT * FROM ftse"

    cur.execute(query)
    data = list(cur.fetchall())


    dataset = pd.DataFrame(data)

    print(dataset)


DBExtraction()
#sudo -u postgres psql


