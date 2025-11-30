import psycopg2

conn = psycopg2.connect('postgresql://neondb_owner:npg_EA4q8IszvkYm@ep-red-tree-ags6al72-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')
cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())
conn.close()

# 'postgresql://neondb_owner:npg_EA4q8IszvkYm@ep-red-tree-ags6al72-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'