import pandas as pd
from pathlib import Path
import json
from synthpop import Synthpop
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class MultiTable:
    def __init__(self, dataset_name: str):
        """
        Initialize the MultiTable object.

        :param dataset_name: The name of the dataset.
        """
        self.dataset_name = dataset_name
        self.metadata = self.set_metadata()
        self.tables = self.set_tables()
        self.extended_tables = {}
        self.extended_metadata = {}

    def set_tables(self) -> None:
        """
        Initialize variables by reading CSV and JSON files into memory. 
        The function populates self.tables and self.metadata.
        """
        # Paths to data, metadata, and relationship files
        location = f"datasets/{self.dataset_name}"
        path_data_tables = Path(f"{location}/data").glob("**/*.csv")

        tables = {}

        # Read all CSV files into tables
        for path in path_data_tables:
            table_name = path.stem  # 'path.stem' gives the filename without extension
            tables[table_name] = pd.read_csv(path)
        
        return tables

    def set_metadata(self) -> None:
        """
        Initialize variables by reading CSV and JSON files into memory. 
        The function populates self.tables and self.metadata.
        """
        # Paths to data, metadata, and relationship files
        location = f"datasets/{self.dataset_name}"
        path_metadata_tables = Path(f"{location}/metadata").glob("**/*.json")
        path_relationship_tables = Path(f"{location}/relationship/relationships.json")

        metadata = {"tables": {}}

        # Read all JSON metadata files into metadata["tables"]
        for path in path_metadata_tables:
            table_name = path.stem
            with open(path, "r", encoding="utf-8") as f:
                metadata["tables"][table_name] = json.load(f)

        # Read relationship JSON file if it exists
        if path_relationship_tables.exists():
            with open(path_relationship_tables, "r", encoding="utf-8") as f:
                metadata["relationships"] = json.load(f)
        else:
            metadata["relationships"] = {}

        return metadata

    def get_key_relationship(self, id_rel):
        """
        The functions returns information about the relationship
        """
        for relationship in self.metadata["relationships"]:
            if relationship["id"] == id_rel:
                return relationship["child_foreign_key"], relationship["parent_primary_key"]
            
    def get_fk_relationship(self, id_rel):
        """
        The functions returns information about the relationship
        """
        for relationship in self.metadata["relationships"]:
            if relationship["id"] == id_rel:
                return relationship["child_foreign_key"]

    def get_tables_names_relationship(self, id_rel):
        """
        The functions returns information about the relationship
        """
        for relationship in self.metadata["relationships"]:
            if relationship["id"] == id_rel:
                return relationship["child_table_name"], relationship["parent_table_name"]

    def get_parents(self, table_name):
        """
        The functions returns the list of parents
        """
        parents = []
        for relationship in self.metadata["relationships"]:
            if relationship["child_table_name"] == table_name:
                parents.append((relationship["id"], relationship["parent_table_name"]))
        return parents

    def get_children(self, table_name):
        """
        The functions returns the list of children
        """
        children = []
        for relationship in self.metadata["relationships"]:
            if relationship["parent_table_name"] == table_name:
                children.append((relationship["id"],relationship["child_table_name"]))
        return children

    def get_numerical_int_columns(self, table_name):
        """
        The functions returns the list of integer columns
        """
        numerical_columns = []
        for column_name,column_value in self.metadata["tables"][table_name]["columns"].items():
            if column_value["sdtype"] == "numerical":
                if column_value["computer_representation"] == "Int64":
                    numerical_columns.append(column_name)
        return numerical_columns

    def get_numerical_float_columns(self, table_name):
        """
        The functions returns the list of float columns
        """
        numerical_columns = []
        for column_name,column_value in self.metadata["tables"][table_name]["columns"].items():
            if column_value["sdtype"] == "numerical":
                if column_value["computer_representation"] == "Float":
                    numerical_columns.append(column_name)
        return numerical_columns

    def get_numerical_columns(self, table_name):
        """
        The functions returns the list of numerical columns
        """
        numerical_columns = []
        for column_name,column_value in self.metadata["tables"][table_name]["columns"].items():
            if column_value["sdtype"] == "numerical":
                numerical_columns.append(column_name)
        return numerical_columns

    def get_categorical_columns(self, table_name):
        """
        The functions returns the list of categorical columns
        """
        categorical_columns = []
        for column_name,column_value in self.metadata["tables"][table_name]["columns"].items():
            if column_value["sdtype"] == "categorical":
                categorical_columns.append(column_name)
        return categorical_columns

    def get_c_n_columns(self, table_name):
        """
        The functions returns the list of numerical and categorical columns
        """
        c_n_columns = []
        for column_name,column_value in self.metadata["tables"][table_name]["columns"].items():
            if column_value["sdtype"] in {"categorical", "numerical"}:
                c_n_columns.append(column_name)
        return c_n_columns

    def extend_metadata_parent(self, id_relationship):
        """
        The function extends the metadata by adding information about 
        categorical and numerical column from parent table
        """
        table_name, name_parent = self.get_tables_names_relationship(id_relationship)

        # Obtain the list of numerical and categorical columns
        column_types = {
            "category": self.get_categorical_columns(name_parent),
            "int": self.get_numerical_int_columns(name_parent),
            "float": self.get_numerical_float_columns(name_parent),
        }

        # Iterate over column types and update metadata
        for dtype, columns in column_types.items():
            for column in columns:
                self.extended_metadata[table_name][f"{id_relationship}_{column}"] = dtype

    def extend_metadata_table(self, table_name):
        """
        The function extends the metadata by adding information about 
        categorical and numerical column from parent table
        """

        # Obtain the list of numerical and categorical columns
        column_types = {
            "category": self.get_categorical_columns(table_name),
            "int": self.get_numerical_int_columns(table_name),
            "float": self.get_numerical_float_columns(table_name),
        }

        # Iterate over column types and update metadata
        for dtype, columns in column_types.items():
            for column in columns:
                self.extended_metadata[table_name][f"{column}"] = dtype

    def extend_parent(self, table_name, parent):
        """
        The function extends the data table adding information about 
        categorical and numerical column in the parent
        """
        id_relationship = parent[0]
        name_parent = parent[1]
        data_parent = self.tables[name_parent].copy()

        # Obtain the list of numerical and categorical columns
        list_c_n_columns = self.get_c_n_columns(name_parent)

        # We obtain the fk in the current table and the pk of the parent table
        fk, pk = self.get_key_relationship(id_relationship)

        # Filter the data parent table
        data_parent = data_parent[list_c_n_columns + [pk]]

        # Rename the columns to avoid duplicity
        rename_mapping = {
            col: f'{id_relationship}_{col}' for col in list_c_n_columns if col in data_parent.columns
        }
        data_parent.rename(columns=rename_mapping, inplace=True)

        # Merge
        self.extended_tables[table_name] = pd.merge(
            self.extended_tables[table_name],
            data_parent,
            how="left",
            left_on=fk,
            right_on=pk
        )

        # We drop the keys
        self.extended_tables[table_name].drop(columns=[fk, pk], inplace=True, errors="ignore")

        # Extend metadata
        self.extend_metadata_parent(id_relationship)

    def extent_parents(self, table_name):
        """
        The function extends the data table adding information about the parents
        """
        list_parents = self.get_parents(table_name)

        for parent in list_parents:
            # add information for each parent
            self.extend_parent(table_name, parent)

    def extent_child(self, table_name, child):
        """
        The functions extend the data table adding information about the children
        """
        id_relationship = child[0]
        child_table_name = child[1]

        #Storage the table
        data_children = self.tables[child_table_name].copy()
        fk,pk = self.get_key_relationship(id_relationship)

        #Count of references
        counts = data_children[fk].value_counts()

        #Add the referencial columns to the table
        self.extended_tables[table_name][f"{child[0]}_count"] = self.extended_tables[table_name][pk].map(counts).fillna(0).astype(int)

        #Add the referencial columns to the table
        self.extended_metadata[table_name][f"{child[0]}_count"] = "int"

    def extent_children(self, table_name):
        """
        The functions extend the data table adding information about the children
        """
        children_tables = self.get_children(table_name)

        for child  in children_tables:
            self.extent_child(table_name, child)

        pk = self.metadata["tables"][table_name]["primary_key"]
        self.extended_tables[table_name].drop(columns=[pk], inplace=True, errors="ignore")

    def extend_data_table(self, table_name):
        """
        The functions extend the table
        """
        # Add the categorical and numerical columns of the parents table
        self.extent_parents(table_name)

        # Add the number of references of each records in the child table
        self.extent_children(table_name)

    def extend_tables(self):
        """
        The functions extend the table
        """
        self.extended_tables = self.tables.copy()
        for table_name in self.tables:
            self.extended_metadata[table_name] = {}
            self.extend_metadata_table(table_name)
            self.extend_data_table(table_name)
            self.adjust_extended_table(table_name)

    def adjust_extended_table(self, table_name) -> pd.DataFrame:
        """
        Adjusts the data types of a DataFrame according to a given dictionary.
        """
        for col, dtype in self.extended_metadata[table_name].items():
            # Ensure the column exists in the DataFrame before converting
            if col in self.extended_tables[table_name].columns:
                if dtype == "int":
                    # Convert column to numeric and then to int
                    self.extended_tables[table_name][col] = pd.to_numeric(self.extended_tables[table_name][col], errors='coerce').astype(int)
                elif dtype == "float":
                    # Convert column to numeric and then to float
                    self.extended_tables[table_name][col] = pd.to_numeric(self.extended_tables[table_name][col], errors='coerce').astype(float)
                elif dtype == "category":
                    # Convert column to a categorical type
                    self.extended_tables[table_name][col] = self.extended_tables[table_name][col].astype("category")
                else:
                    # If the dtype is not recognized, display a warning
                    print(f"Warning: Unsupported dtype '{dtype}' for column '{col}'. Skipping.")


class Synthethizer:
    """
    Class responsible for generating synthetic data from a dataset.
    It includes methods to initialize metadata, synthesize tables, and
    connect tables based on relationships (foreign keys and primary keys).
    """

    def __init__(self, dataset: str):
        """
        Initialize the Synthethizer object with a dataset.

        :param dataset: MultiTable object
        """
        self.dataset = dataset
        self.synthethic_tables = {}
        self.synthethic_extended_tables = {}

    def synthtetize(self, table_name):
        """
        Generates synthetic data for a single table using Synthpop.

        :param table_name: Name of the table to synthesize.
        :return: A pandas DataFrame containing synthetic data for the table.
        """
        # Get extended version of the table
        extended_table = self.dataset.extended_tables[table_name]

        # Retrieve data type info from metadata
        extended_metadata = self.dataset.extended_metadata[table_name]

        # Fit the Synthpop model
        synthetizer = Synthpop()
        synthetizer.fit(extended_table, extended_metadata)

        # Generate synthetic data (sample size is 1/10 of the original records)
        sample_size = self.dataset.tables[table_name].shape[0]//10

        return synthetizer.generate(sample_size)

    def synthtetize_extended_table(self, table_name):
        """
        Generates synthetic data for a single table using Synthpop.

        :param table_name: Name of the table to synthesize.
        :return: A pandas DataFrame containing synthetic data for the table.
        """
        # Retrieve primary key name from metadata
        pk = self.dataset.metadata["tables"][table_name]["primary_key"]

        # Synthesize the table
        synthethic_table = self.synthtetize(table_name)

        # Assign a new primary key for the synthetic table
        synthethic_table[pk] = range(0, len(synthethic_table))

        # Store the extended synthetic table
        self.synthethic_extended_tables[table_name] = synthethic_table

    def synthtetize_extended_tables(self):
        """
        Generates synthetic data for all tables in the dataset.
        """
        # Iterate over all tables
        for table_name in self.dataset.tables.keys():

            # Synthethize each table independently
            self.synthtetize_extended_table(table_name)

    def get_euclidian_distance(self, child, parent):
        """
        Computes the Euclidean distance between two vectors (numpy arrays).
        """
        return np.linalg.norm(child - parent)

    def convert_categorical_columns(self, table_name, name_parent, id_relationship):
        """
        This function conver the categorical columns into a numerical one hot coding representation
        now we can realize calculation with the columns 
        """
        # Retrieve just the categorical columns
        list_cc = self.dataset.get_categorical_columns(name_parent)

        # One-hot encode each categorical column for comparison
        for cc in list_cc:
            # Encoder to transform categorical columns
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

            # Fit the encoder on the original master DataFrame for consistent categories
            encoder.fit(self.dataset.tables[name_parent][[cc]])

            # Transform parent data
            data_parent_encoded = encoder.transform(self.synthethic_tables[name_parent][[cc]])

            # Transform child data
            data_child_encoded = encoder.transform(self.synthethic_tables[table_name][[f'{id_relationship}_{cc}']])

            # Extract the feature names
            feature_names = encoder.categories_[0]

            # Convert the encoded arrays to DataFrames
            data_parent_encoded = pd.DataFrame(data_parent_encoded, columns=feature_names)
            data_child_encoded = pd.DataFrame(data_child_encoded, columns=feature_names)

            # Concatenate encoded columns
            self.synthethic_tables[name_parent] = pd.concat([self.synthethic_tables[name_parent], data_parent_encoded], axis=1)
            self.synthethic_tables[table_name] = pd.concat([self.synthethic_tables[table_name], data_child_encoded], axis=1)

            # Drop the original categorical columns
            self.synthethic_tables[name_parent].drop(columns=[cc], inplace=True)
            self.synthethic_tables[table_name].drop(columns=[f'{id_relationship}_{cc}'], inplace=True)


    def convert_numerical_columns(self, table_name, name_parent, id_relationship):
        """
        This function normalized the numerical columns
        the max and min value are obtained from the real column
        """
        list_nc = self.dataset.get_numerical_columns(name_parent)

        # Iterate over each numerica column
        for nc in list_nc:
            min_value = self.dataset.tables[name_parent][nc].min()
            max_value = self.dataset.tables[name_parent][nc].max()

            # Create normalized columns (range 0 to 1)
            self.synthethic_tables[name_parent][f'Normalized_{nc}'] = (self.synthethic_tables[name_parent][nc] - min_value) / (max_value - min_value)
            self.synthethic_tables[table_name][f'Normalized_{nc}'] = (
                self.synthethic_tables[table_name][f'{id_relationship}_{nc}'] - min_value
            ) / (max_value - min_value)

            # Drop original numeric columns (we now have normalized versions)
            self.synthethic_tables[name_parent].drop(columns=[nc], inplace=True)
            self.synthethic_tables[table_name].drop(columns=[f'{id_relationship}_{nc}'], inplace=True)

    def get_parent_columns(self, name_parent, id_relationship):
        """
        Obtain the categorical and numerical columns that belong to the parent table,
        we also add the column that contain the qapacity
        this is important because the qapacity limits the number of references from the parent records
        """
        # Obtain the list of columns (categorical + numerical) from parent
        parent_list_cc_nc = self.dataset.get_c_n_columns(name_parent)

        # Obtain the qapacity constrain column
        parent_list_cc_nc.append(f"{id_relationship}_count")

        return parent_list_cc_nc

    def get_child_columns(self, name_parent, id_relationship):
        """
        Obtain the columns that belong to the child table, to avoid redundance of columns
        the id_relationship was added when we extended the table
        in an ideal world where different tables do not use the same columns name we wouldnt use
        the id_relationship prefix
        """
        # Obtain the list of columns (categorical + numerical) from parent
        list_cc_nc = self.dataset.get_c_n_columns(name_parent)

        # Change the list by adding the prefix
        child_list_cc_nc = [f'{id_relationship}_{column}' for column in list_cc_nc]

        return child_list_cc_nc

    def filter_relevant_columns(self, table_name, name_parent, id_relationship):
        """
        This funciton filter the relevant columns, we only use the columns that belong to the parent
        """
        # Obtain the categorical and numerical columns from parent table
        # it also include the qapacity column
        parent_list_cc_nc = self.get_parent_columns(name_parent, id_relationship)

        # Obtain the categorical and numerical columns from parent table
        # it includes the id_relationship prefix
        child_list_cc_nc = self.get_child_columns(name_parent, id_relationship)

        # Filter the parent table
        self.synthethic_tables[name_parent] = self.synthethic_extended_tables[name_parent][parent_list_cc_nc].copy()

        # Filter the child table
        self.synthethic_tables[table_name] = self.synthethic_extended_tables[table_name][child_list_cc_nc].copy()

    def convert_columns(self, table_name, name_parent, id_relationship):
        """
        This function converts categorical and numerical columns into a 
        numerical representation, allowing us to perform calculations
        """
        # First we filter the columns that we care about
        # We only used the columns that belong to the parent table
        self.filter_relevant_columns(table_name, name_parent, id_relationship)
        
        # Convert the categorical columns
        self.convert_categorical_columns(table_name, name_parent, id_relationship)
        
        # Convert the numerical columns
        self.convert_numerical_columns(table_name, name_parent, id_relationship)

    def connect_related_tables(self, table_name, name_parent, id_relationship):
        """
        Connect related records by calculating the distance between them
        """
        # Obtain Parent Capacity
        qapacity = self.synthethic_tables[name_parent][f'{id_relationship}_count'].astype(int).tolist()

        # Remove the capacity columns from the parent data 
        self.synthethic_tables[name_parent].drop(columns=[f'{id_relationship}_count'], inplace=True)

        # Convert both parent and child DataFrame to numpy arrays for distance calculations
        parent = self.synthethic_tables[name_parent].to_numpy()
        child = self.synthethic_tables[table_name].to_numpy()

        # Get foreign key (fk)
        fk = self.dataset.get_fk_relationship(id_relationship)

        # Initialize foreign key with empty values
        self.synthethic_tables[table_name][fk] = np.nan

        # Match each child row to the closest parent row (with capacity > 0)
        for i in range(child.shape[0]):
            distance_closest_parent = float('inf')
            index_closest_parent = -1
            # Compare with each parent
            for j in range(parent.shape[0]):
                current_distance = self.get_euclidian_distance(child[i], parent[j])
                if current_distance <= distance_closest_parent and qapacity[j] > 0:
                    distance_closest_parent = current_distance
                    index_closest_parent = j
            # Assign the foreign key to the closest parent and reduce capacity
            if index_closest_parent > -1:
                self.synthethic_tables[table_name].loc[i, fk] = index_closest_parent
                qapacity[index_closest_parent] -= 1

    def get_connected_synthehtic_table(self, table_name):
        """
        Connects a synthetic child table with its parent tables by:
        :param table_name: Name of the child table to connect with its parents.
        :return: A DataFrame of the child table with the appropriate foreign keys set.
        """

        # Copy the synthetic child table to manipulate
        data_table = self.synthethic_extended_tables[table_name].copy()

        # Get list of parents tables
        parent_tables = self.dataset.get_parents(table_name)

        # Iterate over all parents
        for parent in parent_tables:
            
            # Obtain Parent relationship info
            id_relationship = parent[0]
            name_parent = parent[1]

            # Convert the categorial and numerical columns
            # Categorical columns gets converted to one hot coding representation
            # Numerical columns gets normalized
            self.convert_columns(table_name, name_parent, id_relationship)

            # When all columns are transformed into numerical values
            # now it is possible to calculate euclidian distance
            self.connect_related_tables(table_name, name_parent, id_relationship)

            # Get foreign key (fk)
            fk = self.dataset.get_fk_relationship(id_relationship)

            # Copy the computed foreign key values back into the main table
            data_table[fk] = self.synthethic_tables[table_name][fk]

        # Remove columns that start with digits (these were intermediate columns for comparisons)
        data_table = data_table[[col for col in data_table.columns if not col[0].isdigit()]]
        return data_table

    def connect(self):
        """
        We iterate over each table and connect it to its respective parent tables
        The connections is done by assigning the corresponding foreing key value
        """
        for table_name in self.dataset.tables.keys():

            # Get syntehthic data for each table
            # This contains the proper foreing key values
            self.synthethic_tables[table_name] = self.get_connected_synthehtic_table(table_name)

            # Save the table in a csv file
            self.synthethic_tables[table_name].to_csv(
                            f'datasets/{self.dataset.dataset_name}/synthethic_data/{table_name}.csv',
                            index=False
                        )

    def fit(self):
        """
        We generate synthethic data for each extended table
        then we connect each extended table to all its parentt tables
        """
        # Generate syntehthic data for each extended tables
        self.synthtetize_extended_tables()

        # Connect the related tables
        self.connect()

if __name__ == "__main__":
    # We can select the dataframe that we prefer
    dataset = MultiTable(dataset_name="Biodegradability_v1")

    # We extend the tables
    dataset.extend_tables()

    # We generate and storage synthethic data
    synthethizer = Synthethizer(dataset=dataset)
    synthethizer.fit()
