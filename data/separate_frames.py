#sørg for at field_samples_with_targets.csv er i data mappen (den kan hentes fra drive)
import pandas as pd
df = pd.read_csv("data/field_samples_with_targets.csv")
index_to_fieldsample_barcode = df[["fieldsample_barcode"]].to_csv("data/index_to_fieldsample_barcode.csv")
target_df = df[[col for col in df if "TARGET" in col]].to_csv("data/targets.csv")
feature_df = df[[col for col in df if "TARGET" not in col and "passed_filter_reads" not in col and "fieldsample_barcode" not in col]].to_csv("data/features.csv")
passed_filter_reads_df = df[[col for col in df if "passed_filter_reads" in col]].to_csv("data/passed_filter_reads.csv")