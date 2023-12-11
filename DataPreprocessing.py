from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
 # Assemble the features into a vector
from pyspark.ml.feature import VectorAssembler


def preprocess_and_save_data(input_path, output_path):
    try:
        # Create a Spark session
        spark = SparkSession.builder.appName("DatasetPreprocessing").getOrCreate()
        #spark.conf.set("spark.hadoop.fs.s3a.access.key", "Ommited for private reasons)
        #spark.conf.set("spark.hadoop.fs.s3a.secret.key", "Ommited for private reasons")

        # Load the dataset from S3
        df = spark.read.csv(input_path, header=True, inferSchema=True)

        # Columns to drop
        columns_to_drop = ['discharge_disposition_id','admission_source_id', 'number_inpatient','encounter_id','number_emergency',
                           'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
        df = df.drop(*columns_to_drop)


        # Define the value you want to replace
        value_to_replace = '?'
        # Loop through all columns in the DataFrame
        for column in df.columns:
        # Replace the specified value with None
            df = df.withColumn(column, when(df[column] == value_to_replace, None).otherwise(df[column]))


        #Filling missing values in the 'race' column with the most frequent value
        most_frequent_race = df.groupby("race").count().orderBy(col("count").desc()).select("race").first()[0]
        df = df.withColumn("race", F.when(col("race").isNull(), most_frequent_race).otherwise(col("race")))

        # Dropping rows with any remaining missing values
        df = df.dropna()

    
        #diag_1, diag_2, and diag_3 columns to preprocess
        columns_to_preprocess = ['diag_1', 'diag_2', 'diag_3']

        # Step 1: Replace non-numeric values with NaN
        for col_name in columns_to_preprocess:
            non_numeric_condition = ~(col(col_name).cast("double").isNotNull())
            df = df.withColumn(col_name, F.when(non_numeric_condition, None).otherwise(col(col_name)))

        # Step 2: Convert to Numeric
        for col_name in columns_to_preprocess:
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

        # Drop rows with NaN (adjust as needed)
        df = df.dropna(subset=columns_to_preprocess)

        categorical_columns = [ 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
                        'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
                        'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                        'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
                        'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                        'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']

        # Use StringIndexer to convert categorical values to numerical indices
        indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") for col in categorical_columns]

        # Use OneHotEncoder to one-hot encode the categorical indices
        encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") for col in categorical_columns]

        # Map the target variable 'readmitted' to numerical values
        label_indexer = StringIndexer(inputCol="readmitted", outputCol="label")

        # Define the feature columns
        feature_columns = [col+"_encoded" for col in categorical_columns]

       
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # Define the stages of the pipeline
        stages = indexers + encoders + [label_indexer, assembler]

        # Create a pipeline
        pipeline = Pipeline(stages=stages)

        # Fit the pipeline to transform the data
        transformed_data = pipeline.fit(df).transform(df)



        # Save the preprocessed dataset back to S3
        transformed_data.write.format("csv").mode("overwrite").option("header", "true").save(output_path)

        # Stop the Spark session
        spark.stop()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Define input and output paths
    input_path = "s3://diabete/diabetic_data.csv"
    output_path = "s3://diabete/output/"

    # Call the function
    preprocess_and_save_data(input_path, output_path)

