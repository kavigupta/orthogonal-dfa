import pickle, sys
d=pickle.load(open(sys.argv[1],"rb"))
fam=[tuple(map(int,s)) for s in d["dt"].base_family]
with open(sys.argv[2],"w") as f:
    for t in sorted(set(fam)):
        f.write(",".join(map(str,t))+"\n")
print(sys.argv[2], "unique", len(set(fam)), "total", len(fam))
