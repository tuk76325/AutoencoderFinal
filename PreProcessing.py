import re
import sys, os
import string
import random
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randint
from Bio import AlignIO
from mi3gpu.mi3gpu.utils import seqload

'''HELP'''
from mi3gpu.mi3gpu.utils import seqtools
''''''

# -- CHANGE NAMES BELOW --
DATASET_NAME = "PF00018"
FILE_PATH = "/home/tuk76325/work/PythonProjects/MinaAutoEncoder/"
# -- -- -- -- -- -- -- -- 

filter_val = 0.2 #standard = 0.2

# -- -- -- -- -- funcs -- -- -- -- -- 
# STEP 1
# This function converts an interpro database file into a FASTA file

def makeFasta(FILEPATH, DATASET):
    #file names
    input_file = FILEPATH + DATASET + ".alignment.uniprot"
    output_file = FILEPATH + DATASET + "_interpro_uniprot.fa"

    table = str.maketrans('', '', string.ascii_lowercase)
    fp = open(input_file,"r")
    with open(output_file,"w") as out:
        for line in fp:
            if line.startswith(("#","\n","//")):
                continue
            else:
                line = line.strip().split()
                print(">" + line[0]+ '\n')
                out.write(">"+ line[0]+'\n')
                out.write(line[1].upper().replace(".","-")+'\n')
                print(line[1].upper().replace(".","-")+'\n')

    fp.close()
    return("FASTA file written to:", output_file)


# STEP 2
# This function condenses sequences from a FASTA file and outputs a new FASTA file

def munge(FILEPATH, DATASET):
    # Read the MSA file using AlignIO
    fig, ax = plt.subplots()
    alignment = AlignIO.read(FILEPATH + DATASET + "_interpro_uniprot.fa", 'fasta')
    mask_name = FILEPATH + "masks/" + DATASET + "_uniprot_mask.npy"
    hist_name = FILEPATH + "hists/" + DATASET + "_uniprot_AlignOut.png"
    out_name = FILEPATH + DATASET + "_uniprot_nope10_short.fa"
    fig.set_figheight(10)
    fig.set_figwidth(200)
    
    # Convert the MSA to a numpy ndarray
    msa = np.array([list(rec) for rec in alignment])

    # Get the number of columns in the MSA
    num_columns = msa.shape[1]
    num_rows = msa.shape[0]

    freqs = list(map(lambda i: np.unique(msa[:, i], return_counts=True), range(num_columns)))
    mask = []
    dels=[]
    freqDicts = []    
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
        print(f"Column {i}: Gap Percent: {gap_percent}%")

    mask = np.asarray(mask)
    msa = np.delete(msa,dels,axis=1)
    print(mask)
    open(mask_name, 'w')
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
        
    print("Sequences removed:", count)
    print("Munge complete. Saved to:", out_name)
    fp.close()


# STEP 3
# This function excludes sequences that are too similar

def filterMSA(FILEPATH, DATASET, deg_similarity):
    # remove FASTA names --> want sequences only
    in_name = FILEPATH + DATASET + "_uniprot_nope10_short.fa"
    out_name = FILEPATH + DATASET + "_uniprot_nope10_short_seqs.fa"
    output_file = FILEPATH + DATASET + "uniprot_nope10_short_seqs_" + str(deg_similarity * 100) + "filter"
    
    out = open(out_name, 'w')
    pattern = r'^>.*\n'
    for line in open(in_name, 'r'):
        line = re.sub(pattern, '', line, flags=re.MULTILINE)
        out.write(line)
    
    s = seqload.loadSeqs(FILEPATH + DATASET + "_uniprot_nope10_short_seqs.fa")[0] #file name
    print(s)
    cutoff = 1-deg_similarity #standard = 0.2
    L = s.shape[1]
    print(L)
    inds = []
    out_seq = []
    while s.shape[0] != 0:
        ind = randint(s.shape[0])
        out_seq.append(s[ind].copy()) # no ref to s
        s = s[np.sum(s == s[ind,:], axis=1)/float(L) < cutoff,:]
        # print(s.shape)

    #with os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as fp:
    with open(output_file, 'wb') as fp:
        seqload.writeSeqs(fp, np.array(out_seq))

    print("Filtering complete. saved to: ", output_file)


# STEP 4
# This script divides sequences into 3 groups: training data, testing data, and validation data

def ttv(FILEPATH, DATASET):
    #initialize output file names
    output_file_train = "train_" + DATASET + "uniprot_filtered0.2"
    output_file_val = "val_" + DATASET + "uniprot_filtered0.2"
    output_file_test = "test_" + DATASET + "PF00018uniprot_filtered0.2"
    
    #change split ratios 
    '''
    test
    train 
    val 
    '''
    ###
    # remove FASTA names --> want sequences only
    in_name = FILEPATH + DATASET + "_uniprot_nope10_short.fa"
    out_name = FILEPATH + DATASET + "_uniprot_nope10_short_seqs.fa"
    out = open(out_name, 'w')
    pattern = r'^>.*\n'
    for line in open(in_name, 'r'):
        line = re.sub(pattern, '', line, flags=re.MULTILINE)
        out.write(line)
    ###

    seqs = out.split("\n")[:-1]
    random.shuffle(seqs)
    seqs = np.asarray(seqs)
    train, test, val = np.split(seqs, [int(len(seqs)*0.7), int(len(seqs)*0.9)])
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")
    print(f"Val: {len(val)}")
    print(f"Seqs: {len(seqs)} | {len(train)+len(test)+len(val)} = Sum")
    with open(output_file_train,"w") as fp:
        for seq in train:
            fp.write(seq+'\n')
    with open(output_file_val,"w") as fp:
        for seq in val:
            fp.write(seq+'\n')
    with open(output_file_test,"w") as fp:
        for seq in test:
            fp.write(seq+'\n')
    return("STeP 4 COmPletE")

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# STEP 1
step1 = input("Convert to FASTA? (y/n): ")
if step1 == 'y':
    makeFasta(FILE_PATH, DATASET_NAME)

# STEP 2
step2 = input("Munge? (y/n): ")
if step2 == 'y':
    munge(FILE_PATH, DATASET_NAME)

# STEP 3
step3 = input("filter MSA? (y/n): ")
if step3 == 'y':
    filterMSA(FILE_PATH, DATASET_NAME, filter_val)

# STEP 4
#ttv(DATASET_NAME)