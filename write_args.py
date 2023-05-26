# for grid search
qs = [0.05, 0.1, 0.3, 0.5, 1, 2, 5]
taus = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 12]
cond = 0.2
ca = 0.1
with open("args.txt", "w") as f:
    for q in qs:
        for tau in taus:
            f.write(f"{q} {tau} {cond} {ca}\n")


# for figures 4 and 5 (?)
conds = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 1.4, 1.5, 3]
cas = [0, 0.1]
q = 0.3
tau = 0.1
with open("args.txt", "a") as f:
    for c in conds:
        for ca in cas:
            f.write(f"{q} {tau} {c} {ca}\n")
    f.write(f"{q} {tau} {0} {0}\n")   # Additional parameter set for absence of ASICs
