[SMOTE](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node6.html), or Synthetic Minority Oversampling TEchnique, is an algorithm for synthesizing "reasonable" data from a base dataset. This tool is a modification of the basic SMOTE algorithm, intended to work with non-numeric data.

For a brief explanation of how this works, please see [this blog post](https://www.pilosa.com/blog/smote/).

## Usage

This is a prototype, and not as user-friendly as it might be. To use with another CSV file, modify the `template` data set:

- `fin` is the input filename.
- `fout` is the output filename.
- `fin_enum` is a secondary input filename, required for working with non-numeric data. It is a csv file with the same structure and size as the primary file, but with categorical values replaced with integers. This can be automated, but defining the mapping manually allows for imparting structure to the enumerated values, which may be helpful in classification tasks.
- `feature_nametypes` defines how data is synthesized for each field. Currently supports three types: float, int, and enum. 
- `feature_enums` is a mapping from categorical values to enumerated integers. Not currently used.
- `target_field` is the dependent variable, the class label. Each synthetic data point is generated from original data within a single class. This is currently a required parameter, but the algorithm can easily be modified to work without it.

Ensure the dependencies listed in requirements.txt are installed. Then run `python smote.py`.


Generating the enumerated CSV file can be done with a script like the following:

```bash
orig="bank-additional-full.csv"
enum="bank-additional-full-enum.csv"
cp $orig $enum

gsed -i 's/divorced/-2/g' $enum
```

Note that replacing a short string like `s/no/0` may have unexpected effects.
