from sqlitecloud.client import SqliteCloudClient, SqliteCloudAccount
#import streamlit as st

account = SqliteCloudAccount("sqlitecloud://user:pass@host.com:port/dbname?timeout=10&key2=value2&key3=value3")
client = SqliteCloudClient(cloud_account=account)
conn = client.open_connection()
result = client.exec_statement(
    "SELECT * FROM table_name WHERE id = ?",
    [1],
    conn=conn
)
for row in result:
    print(row)