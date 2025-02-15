# SRD-ED
# Synthetic Data Generation for Relational Datasets using Euclidean Distance

## Overview
This project generates synthetic data for relational datasets using the concept of extended tables and Euclidean distance to connect related records. The synthetic data generation process is powered by the `synthpop` library, which creates synthetic data for individual extended tables.

## Features
- Utilizes **extended tables** to represent individual tables.
- Applies **Euclidean distance** to establish relationships between records.
- Uses **synthpop** to generate synthetic data for each extended table independently.
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
- **Python 3.8**
- Dependencies specified in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Running the Code
Execute the main script to generate synthetic data:
```bash
python Synthethic_data.py
```

## Relationships in `DCG_v1`
The `DCG_v1` dataset consists of multiple tables linked through relational mappings. The relationships among these tables are established using Euclidean distance for connecting related records. The relationship details are stored in the `relationship/` folder for reference.

### Relationship Diagram
```mermaid
graph TD;
    Sentences[Sentences Table] -->|id_sentence| Terms[Terms Table]
```

## Relationships in `CORA_v1`
The `CORA_v1` dataset consists of multiple tables connected by relational links representing paper citations and content relationships.

### Relationship Diagram
```mermaid
graph TD;
    Paper[Paper Table] -->|paper_id| Cites[Cites Table]
    Paper[Paper Table] -->|paper_id| Content[Content Table]
```

## Relationships in `Biodegradability_v1`
The `Biodegradability_v1` dataset consists of multiple tables representing molecules, atoms, bonds, and groups.

### Relationship Diagram
```mermaid
graph TD;
    Molecule[Molecule Table] -->|molecule_id| Atom[Atom Table]
    Atom[Atom Table] -->|atom_id| Bond[Bond Table]
    Atom[Atom Table] -->|atom_id| Gmember[Gmember Table]
    Gmember[Gmember Table] -->|gmember_id| Group[Group Table]
```

## Relationships in `imdb_MovieLens_v1`
The `imdb_MovieLens_v1` dataset consists of multiple tables connecting movies, actors, directors, users, and their interactions.

### Relationship Diagram
```mermaid
graph TD;
    Actors[Actors Table] -->|actor_id| Movies2Actors[Movies2Actors Table]
    Movies[Movies Table] -->|movie_id| Movies2Actors[Movies2Actors Table]
    Movies[Movies Table] -->|movie_id| Movies2Directors[Movies2Directors Table]
    Directors[Directors Table] -->|director_id| Movies2Directors[Movies2Directors Table]
    Movies[Movies Table] -->|movie_id| U2Base[U2Base Table]
    Users[Users Table] -->|user_id| U2Base[U2Base Table]
```



