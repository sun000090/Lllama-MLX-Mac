import subprocess

subprocess.run('pip install -q mysql-connector-python',shell=True)

import mysql.connector

# Connect to server
cnx = mysql.connector.connect(
    host="sql5.freesqldatabase.com",
    port=3306,
    user="sql5765837",
    password="sFh3wmIDPI")

# Get a cursor
cur = cnx.cursor()

# Execute a query and fetch table
cur.execute("SELECT * from sql5765837.user_credentials")
data = cur.fetchall()