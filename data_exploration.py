import pandas as pd

from Preprocessing.gmt import gmt_to_table

# Define global vars
PPI_FILEPATH='data/BIOGRID-ALL-4.4.203.tab3.txt'
GENE_SET_FILEPATH='data/h.all.v7.4.entrez.gmt'

# Configure Pandas
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

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






# print(df[col].count())
print(len(df[col].unique()))
# print(df[col].count() - len(df[col].unique()))

# print(df[col].duplicated(keep=False))
# print(df.loc[df[col].duplicated(keep=False)]['IDENTIFIER'].value_counts().value_counts())

# # print(df.groupby(['IDENTIFIER']))

# print()
# print(len(df['HALLMARK'].unique()))



# df = pd.read_table(PPI_FILEPATH, nrows=100)

# print(df.head)
# print('\n')
# print (df.columns)
# print('\n')
# print(df[['Systematic Name Interactor A','Organism Name Interactor A']])

# Load in the data in chunks
#   for each chunk 
#       save the row if both Organism Name Interactor A and B are homo sapien
# 

