import pyarrow.parquet as pq

root_folder = 'data'
table = pq.read_table(f'{root_folder}/web/refinedweb0.parquet')
df = table.to_pandas()
json_data = df.to_json(orient='records', lines=True)
with open(f'{root_folder}/web/refinedweb0.json', 'w') as f:
    f.write(json_data)