import os


entry = dict()

with open('dump') as f:
    s = f.readlines()
    for i in s:
        entry[i] = entry[i] + 1 if i in entry else 1
        
        
for i in entry:
    print(f"{i.strip()}: {entry[i]}")
        
        