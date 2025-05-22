import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('licensePlatesDatabase.db')
cursor = conn.cursor()

# Query to retrieve all rows from the LicensePlates table
cursor.execute('SELECT * FROM LicensePlates')

# Fetch all results
rows = cursor.fetchall()

# Print all rows
for row in rows:
    print(row)

# Close the connection
conn.close()
