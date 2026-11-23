 # EXPLORATORY DATA ANALYSIS

print("\n========== DESCRIPTIVE STATISTICS ==========")

# Numerical variables
numerical_data = df.select_dtypes(
    include=np.number
)

print("\nNumerical Summary:")
print(numerical_data.describe().T)

numerical_data.describe().T.to_csv(
    OUTPUT_DIR / "numerical_summary.csv"
)


# Categorical variables
categorical_data = df.select_dtypes(
    include=["object", "string", "category"]
)

print("\nCategorical Summary:")
print(categorical_data.describe().T)

categorical_data.describe().T.to_csv(
    OUTPUT_DIR / "categorical_summary.csv"
)



#  HISTOGRAMS

histogram_columns = [
    "Age",
    "BMI",
    "Tumor_Size_cm",
    "Survival_Months"
]

for column in histogram_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_histogram.png",
        dpi=300
    )

    plt.close()

print("Histograms saved successfully!")


# CATEGORICAL ANALYSIS
categorical_columns = [
    "Gender",
    "Smoking_Status",
    "Cancer_Type",
    "Cancer_Stage",
    "Survived"
]

for column in categorical_columns:

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index
    )

    plt.title(f"Count of Patients by {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Records")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{column}_countplot.png",
        dpi=300
    )

    plt.close()

print("Categorical charts saved successfully!")


 #CORRELATION ANALYSIS

correlation_columns = [
    "Age",
    "Cigarettes_Per_Day",
    "Years_Smoking",
    "BMI",
    "Tumor_Size_cm",
    "Survival_Months"
]

correlation_matrix = df[
    correlation_columns
].corr(method="pearson")

print("\nCorrelation Matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()

