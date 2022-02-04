import psycopg

#con = psycopg.connect(host="localhost",database='mydb',user='postgres',password='')
#print(con)
con2 =  psycopg.connect(" user=postgres")
print(con2)

cur = con2.cursor()
query = "SELECT * FROM ftse"

cur.execute(query)
data = list(cur.fetchall())
print(data)
