from pathlib import Path

import pandas as pd
import numpy as np

from Preprocessing.gmt import gmt_to_table

# Define global vars
PPI_FILEPATH='data/BIOGRID-ALL-4.4.203.tab3.txt'
GENE_SET_FILEPATH='data/h.all.v7.4.entrez.gmt'

PPI_FILEPATH_PROCESSED = 'data/human_ppi.csv'

LABELLED_EDGE_LIST_CLASSIFICATION = 'data/node_classification_edge_list.csv'
LABELLED_VERTEX_LIST_CLASSIFICATION = 'data/node_classification_vertex_list.csv'

LABELLED_EDGE_LIST_LINK_PREDICTION = 'data/link_prediction_edge_list.csv'
LABELLED_VERTEX_LIST_LINK_PREDICTION = 'data/link_prediction_vertex_list.csv'

CHUNKSIZE = 1000
ORGANISM_ID = 9606 # Human
ORGANISM_NAME = 'Homo sapiens'

# Configure Pandas
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


# ++++++++++++++ GENE LABELS ++++++++++++++

# Make a table of all genes which have labelled processes
table = gmt_to_table(GENE_SET_FILEPATH, out='dict')

df = pd.DataFrame(data=table)

col = 'HALLMARK'
print('\nClass label summary: ')

# Count unique class labels
print(f"> {len(df[col].unique())} unique class labels")

col = 'IDENTIFIER'

# Check for null values
if df[col].notnull().all():
    print(f"> No null values are present in col ['{col}']")
else:
    print(f"> Null values are present in col ['{col}']")

# Check if values are unique
if df[col].count() == len(df[col].unique()):
    print(f"> Col ['{col}'] values are unique - multi-class classification")
else:
    print(f"> Col ['{col}'] values are not unique - multi-label classification")

    # Count occurences of repeated values
    duplicated_counts = df['IDENTIFIER'].value_counts()
    duplicate_distribution = duplicated_counts.value_counts()
    sum_genes = 0
    for index, value in duplicate_distribution.items():
        sum_genes += value
        print(f"\t> {value} genes with {index} class labels")
    print(f"\n\t>> {sum_genes} unique genes")


# ++++++++++++++ PPI DATA ++++++++++++++

df1 = pd.read_table(PPI_FILEPATH, nrows=100)

# Display col names
print('\n', 'PPI Data Columns')
for col in df1.columns:
    print(f"\t > {col}")

# Check values for columns
columns =   ['Entrez Gene Interactor A',
            'Entrez Gene Interactor B',
            'Systematic Name Interactor A',
            'Systematic Name Interactor B',
            'Official Symbol Interactor A',
            'Official Symbol Interactor B',
            'Organism ID Interactor A',
            'Organism ID Interactor B',
            'Organism Name Interactor A',
            'Organism Name Interactor B',
            'Tags']

print('\n', 'Sample data from columns of interest')
for col in columns:
    # Sample entry
    print('\n', df1[col].sample())  

# Read in human data
human_ppi_data = Path(PPI_FILEPATH_PROCESSED)

if not human_ppi_data.exists():

    #   Inititliase an empty table based of column names in file
    df1 = pd.read_table(PPI_FILEPATH, nrows=0, 
                usecols=['Entrez Gene Interactor A',
                        'Entrez Gene Interactor B',
                        'Organism ID Interactor A',
                        'Organism ID Interactor B',
                        'Organism Name Interactor A',
                        'Organism Name Interactor B'
                        ])

    print(df1)

    #   Read in the data in chunks and appened to df1 if organism 
    #   is human
    with pd.read_table(PPI_FILEPATH, chunksize=CHUNKSIZE) as data_in:
        for chunk in data_in:
            # print(chunk.shape)
            human_data = chunk.loc[ (chunk['Organism ID Interactor A'] == ORGANISM_ID) & 
                                    (chunk['Organism ID Interactor B'] == ORGANISM_ID)]
            human_interactors = human_data[['Entrez Gene Interactor A',
                                            'Entrez Gene Interactor B',
                                            'Organism ID Interactor A',
                                            'Organism ID Interactor B',
                                            'Organism Name Interactor A',
                                            'Organism Name Interactor B'
                                            ]]
            frames = [df1, human_interactors]
            df1 = pd.concat(frames, ignore_index=True)
            print(df1.shape)

    df1.to_csv(PPI_FILEPATH_PROCESSED, index=False)

else:

    df1 = pd.read_csv(PPI_FILEPATH_PROCESSED)
    print(f"\n df1 Loaded. Shape: {df1.shape}")

# Validate df1
assert not df1['Entrez Gene Interactor A'].isna().any()
assert not df1['Entrez Gene Interactor B'].isna().any()
assert ((len(df1['Organism Name Interactor A'].unique()) == 1) and
        (df1['Organism Name Interactor A'].unique()[0] == ORGANISM_NAME))
assert ((len(df1['Organism Name Interactor B'].unique()) == 1) and
        (df1['Organism Name Interactor B'].unique()[0] == ORGANISM_NAME))  


# ++++++++++++++ CREATE LABELLED EDGE LIST ++++++++++++++
'''
LABELLED_EDGE_LIST_CLASSIFICATION
LABELLED_VERTEX_LIST_CLASSIFICATION
'''

node_classification_edge_list = Path(LABELLED_EDGE_LIST_CLASSIFICATION)



if not node_classification_edge_list.exists():

    labelled_genes = df['IDENTIFIER'].unique()

    edges_with_labels = df1.loc[
                            df1['Entrez Gene Interactor A'].apply(lambda x: x in labelled_genes) &
                            df1['Entrez Gene Interactor B'].apply(lambda x: x in labelled_genes)
                        ]

    edges_with_labels[[ 'Entrez Gene Interactor A', 
                        'Entrez Gene Interactor B']].to_csv(
                            LABELLED_EDGE_LIST_CLASSIFICATION, index=False)

else: 

    edges_with_labels = pd.read_csv(LABELLED_EDGE_LIST_CLASSIFICATION)    

print(edges_with_labels.head(10))
print(edges_with_labels.shape)


# ++++++++++++++ CREATE LABELLED VERTEX LIST ++++++++++++++

node_classification_vertex_list = Path(LABELLED_VERTEX_LIST_CLASSIFICATION)

if not node_classification_vertex_list.exists():
    
    unique_labels = df['HALLMARK'].unique().tolist()
    unique_genes = df.groupby(by='IDENTIFIER')
    columns = ['GENE']
    columns.extend(unique_labels)

    vertices_with_labels = pd.DataFrame(columns=columns)

    idx = 0

    for k, v in unique_genes.groups.items():

        labels = df.loc[v]['HALLMARK']
        vertices_with_labels.loc[idx, 'GENE'] = k

        for label in labels:
            vertices_with_labels.loc[idx, label] = 1

        vertices_with_labels.loc[idx].fillna(0, inplace=True)   

        idx += 1 

    vertices_with_labels.to_csv(LABELLED_VERTEX_LIST_CLASSIFICATION, index=False)

else: 
    
    vertices_with_labels = pd.read_csv(LABELLED_VERTEX_LIST_CLASSIFICATION)


# print(vertices_with_labels.head(10))
print(vertices_with_labels.shape)
print(len(df['IDENTIFIER'].unique()))


