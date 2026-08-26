import json

path = r'c:\Users\medog\OneDrive\Desktop\Projects\symba\SYMBA_Burgers_Lie_Symmetry_Enhanced_v4.ipynb'
nb = json.load(open(path, encoding='utf-8'))
out = open(r'c:\Users\medog\OneDrive\Desktop\Projects\symba\_nb_dump.txt', 'w', encoding='utf-8')
for i, c in enumerate(nb['cells']):
    out.write(f"=== CELL {i} type={c['cell_type']} id={c.get('id')} ===\n")
    out.write(''.join(c['source']))
    out.write('\n\n')
out.close()
print('done', len(nb['cells']))
