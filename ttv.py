import subprocess
import sys
import random
import numpy as np

# sed_command = ['sed', '/^>/d', sys.argv[1]]
# # output = subprocess.run(sed_command,shell=True, capture_output=True, text=True)
# # print(output.stdout)
# output = subprocess.check_output(sed_command, universal_newlines=True)
output = open(f'PF00018uniprot_nope10_short_seqs_20.0filter', 'r')
seqs = []
for line in output:
    seqs.append(line)
random.shuffle(seqs)
seqs = np.asarray(seqs)
train, test, val = np.split(seqs, [int(len(seqs)*0.7), int(len(seqs)*0.9)]) #split into 70%, 20%, 10%
print(f"Train: {len(train)}")
print(f"Test: {len(test)}")
print(f"Val: {len(val)}")
print(f"Seqs: {len(seqs)} | {len(train)+len(test)+len(val)} = Sum")
with open("train_PF00018uniprot_filtered0.2","w") as fp:
    for seq in train:
        fp.write(str(seq))
with open("val_PF00018uniprot_filtered0.2","w") as fp:
    for seq in val:
        fp.write(str(seq))
with open("test_PF00018uniprot_filtered0.2","w") as fp:
    for seq in test:
        fp.write(str(seq))