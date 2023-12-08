import numpy as np
from Bio import AlignIO
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Replace `filename` with the name of your .fasta MSA file
msaType = "PF00018"

# Read the MSA file using AlignIO
fig, ax = plt.subplots()

if msaType == "PF00018":
    alignment = AlignIO.read("PF00018_interpro_uniprot.fa", 'fasta')
    mask_name = "./masks/PF00520_full_mask.npy"
    hist_name = "./hists/PF00520_full_AlignOut.png"
    out_name = "./PF00520_full_nope10_short.fa"
    fig.set_figheight(10)
    fig.set_figwidth(200)
elif msaType == "fullSeq":
    alignment = AlignIO.read("./InterproFull.fa", 'fasta')
    mask_name = "./masks/maskInterproFull.npy"
    hist_name = "./hists/aaHistInterproFull.png"
    out_name = "./InterproFull_shortMSA_nope10.fa"
    fig.set_figheight(7)
    fig.set_figwidth(50)
elif msaType == "uniprot":
    alignment = AlignIO.read("./PF00520_interpro_uniprot_noX_InsGap.fa", 'fasta')
    mask_name = "./masks/PF00520_uniprot_mask.npy"
    hist_name = "./hists/PF00520_uniprot_AlignOut.png"
    out_name = "./PF00520_uniprot_nope10_short.fa"
    fig.set_figheight(10)
    fig.set_figwidth(200)
elif msaType == "PF00069":
    alignment = AlignIO.read("./PF00069_interpro_uniprot.fa", 'fasta')
    mask_name = "./masks/PF00069_uniprot_mask.npy"
    hist_name = "./hists/PF00069_uniprot__AlignOut.png"
    out_name = "./PF00069_uniprot_nope10_short.fa"
    fig.set_figheight(10)
    fig.set_figwidth(200)
elif msaType == "PF00018":
    alignment = AlignIO.read("./PF00018_interpro_uniprot.fa", 'fasta')
    mask_name = "./masks/PF00018_uniprot_mask.npy"
    hist_name = "./hists/PF00018_uniprot__AlignOut.png"
    out_name = "./PF00018_uniprot_nope10_short.fa"
    fig.set_figheight(10)
    fig.set_figwidth(200)    

# Convert the MSA to a numpy ndarray
msa = np.array([list(rec) for rec in alignment])

# Get the number of columns in the MSA
num_columns = msa.shape[1]
num_rows = msa.shape[0]

freqs = list(map(lambda i: np.unique(msa[:, i], return_counts=True), range(num_columns)))
# print(freqs)
mask = []
dels=[]
freqDicts = []    
# keeps = 0
for i, freq in enumerate(freqs):
    freqDict = dict(zip(freq[0],freq[1]))
    freqDicts.append(freqDict)
    try:
        gap_percent = (freqDict['-']/num_rows)*100
    except KeyError:
        gap_percent = 0

    if gap_percent >= 20:
       mask.append("_")
       dels.append(i)
    else:
        mask.append(1)
        # keeps+=1
    print(f"Column {i}: Gap Percent: {gap_percent}%")

mask = np.asarray(mask)
msa = np.delete(msa,dels,axis=1)
print(mask)
np.save(mask_name, mask)

symbols = ['G','A','V','P','M','L','I','F','W','C','S','T','Y','H','R','K','Q','N','D','E','-']
a_color = { 
    "A" : "#7600A8",
    "C" : "#382903",
    "D" : "#D24D57",
    "E" : "#FF0000",
    "F" : "#806C00",
    "G" : "#58007E",
    "H" : "#3455DB",
    "I" : "#9D8319",
    "K" : "#00008B",
    "L" : "#AA8F00",
    "M" : "#B8860B",
    "N" : "#D35400",
    "P" : "#FF00FF",
    "Q" : "#D46A43",
    "R" : "#0000E0",
    "S" : "#1BA39C",
    "T" : "#00AA00",
    "V" : "#BF6EE0",
    "W" : "#5A440D",
    "Y" : "#005500",
    "-" : "lightgrey"
}
plotDict = dict((k, []) for k in symbols)

for d in freqDicts:
    for s in symbols:
        try:
            plotDict[s].append((d[s]/num_rows)*100)
        except KeyError:
             plotDict[s].append(0)

xLabels = range(1,num_columns+1)

bot = np.zeros(num_columns)
for s in symbols:
    ax.bar(xLabels, plotDict[s], width=0.75, bottom = bot, color=a_color[s])
    bot += np.array(plotDict[s])

plt.legend(symbols)
plt.savefig(hist_name)

count = 0
with open(out_name, "w") as fp:
    for i, record in enumerate(alignment):
        out_str=""
        for char in msa[i]:
            out_str += char
        if "----------" in out_str or "X" in out_str or "Z" in out_str or "B" in out_str:
            count+=1
            continue
        fp.write(">"+record.id+"\n")
        fp.write(out_str)
        fp.write('\n')
        
print(count)
fp.close()
