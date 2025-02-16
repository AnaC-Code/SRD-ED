# SRD-ED
# Synthetic Relational Datasets using Euclidean Distance

## Overview
This project generates synthetic data for relational datasets using the concept of extended tables and Euclidean distance to connect related records. The synthetic data generation process is powered by the `synthpop` library, which creates synthetic data for individual extended tables.

## Features
- Utilizes **extended tables** to represent individual tables.
- Uses **synthpop** to generate synthetic data for each extended table independently.
- Applies **Euclidean distance** to establish relationships between records.
- Provides example implementations for the following datasets:
  - `DCG_v1`
  - `CORA_v1`
  - `Biodegradability_v1`
  - `imdb_MovieLens_v1`

## Dataset Structure
Each dataset folder contains the following subdirectories:
- `data/` - Contains the original dataset.
- `metadata/` - Stores metadata information for the dataset.
- `relationship/` - Defines the relationships between different tables.
- `synthetic_data/` - Contains the generated synthetic data.

The information about `data`, `metadata`, and `relationship` is sourced from datasets obtained via the **SDV library**.

## Requirements  

The project contains two files: `Synthethic_data.py` and `Compare_data.py`.  

- `Synthethic_data.py` generates synthetic data for a selected dataset using the `synthpop` library, which runs properly in Python 3.8. It also requires the dependencies listed in the `requirements.txt` file.  
- `Compare_data.py` evaluates the synthetic data using metrics from the `sdv` library, which requires Python 3.12.  

Since each script requires a different Python environment, ensure that you have two separate virtual environments before running the scripts:  
- One environment for Python 3.8 (for `Synthethic_data.py`).  
- Another environment for Python 3.12 (for `Compare_data.py`).  

### `Synthethic_data.py`  
- Requires Python 3.8  
- Dependencies are specified in `requirements.txt`  

#### Installation  
```bash
pip install -r requirements.txt
```

#### Running the Code  
Execute the script to generate synthetic data:  
```bash
python Synthethic_data.py
```

---

### `Compare_data.py`  
- Requires Python 3.12  
- Only requires the `sdv` library  

#### Installation  
```bash
pip install sdv
```

#### Running the Code  
Execute the script to evaluate synthetic data:  
```bash
python Compare_data.py
```
## Limitations

The project currently supports numerical and categorical columns. The primary and foreign keys should be single columns. It does not handle null values, so it would be helpful to replace them, using a random number for numerical columns or a new category for categorical columns.

## Datasets

There are four datasets available from the SDV library, with a brief explanation of each provided below.

### `DCG_v1`
The `DCG_v1` dataset consists of multiple tables linked through relational mappings. The relationships among these tables are established using Euclidean distance for connecting related records. The relationship details are stored in the `relationship/` folder for reference.

#### Relationship Diagram
```mermaid
graph TD;
    Sentences[Sentences Table] -->|id_sentence| Terms[Terms Table]
```

### `CORA_v1`
The `CORA_v1` dataset consists of multiple tables connected by relational links representing paper citations and content relationships.

#### Relationship Diagram
```mermaid
graph TD;
    Paper[Paper Table] -->|cited_paper_id| Cites[Cites Table]
    Paper[Paper Table] -->|citing paper id| Cites[Content Table]
    Paper[Paper Table] -->|paper id| Content[Content Table]
```

### `Biodegradability_v1`
The `Biodegradability_v1` dataset consists of multiple tables representing molecules, atoms, bonds, and groups.

#### Relationship Diagram
```mermaid
graph TD;
    Molecule[Molecule Table] -->|molecule_id| Atom[Atom Table]
    Atom[Atom Table] -->|atom_id| Bond[Bond Table]
    Atom[Atom Table] -->|atom_id2| Bond[Bond Table]
    Atom[Atom Table] -->|atom_id| Gmember[Gmember Table]
    Group[Group Table] -->|group_id| Gmember[Gmember Table]
```

### `imdb_MovieLens_v1`
The `imdb_MovieLens_v1` dataset consists of multiple tables connecting movies, actors, directors, users, and their interactions.

#### Relationship Diagram
![imdb_MovieLens_v1](images/Movie.png)



