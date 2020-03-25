# https://bioconductor.org/packages/release/bioc/vignettes/SC3/inst/doc/SC3.html

library(SC3)
library(SingleCellExperiment)
library(scater)

head(ann)

yan[1:3, 1:3]

# create a single cell experiment object
sce <- SingleCellExperiment(
  assays = list(
    counts = as.matrix(yan), 
    logcounts = log2(as.matrix(yan) + 1)
  ),
  colData = ann
)

# define feature names in the feature_symbol column
rowData(sce)$feature_symbol <- rownames(sce)
# remove duplicate features
sce <- sce[!duplicated(rowData(sce)$feature_symbol), ]

# define spike in
isSpike(sce, "ERCC") <- grepl("ERCC", rowData(sce)$feature_symbol)

# use scater to visualise the single cell object
plotPCA(sce, colour_by = "cell_type1")

# run clustering in the range of 2 to 4 clusters
# use n_cores to manually set the number of cores
sce <- sc3(sce, ks = 2:4, biology = TRUE)

# visualise in a Shiny app
sc3_interactive(sce)

# export results
sc3_export_results_xls(sce)

# identify sc3 results from the colData dataframe
col_data <- colData(sce)
head(col_data[ , grep("sc3_", colnames(col_data))])

# it is simple to use these results in plots
plotPCA(
  sce,
  colour_by = "sc3_3_clusters",
  size_by = "sc3_3_log2_outlier_score"
)

# results for features
row_data <- rowData(sce)
head(row_data[ , grep("sc3_", colnames(row_data))])

##########

# CONVENIENCE FUNCTIONS

# consensus matrix
sc3_plot_consensus(sce, k = 3)

sc3_plot_consensus(
  sce, k = 3,
  show_pdata = c(
    "cell_type1",
    "sc3_3_clusters",
    "sc3_3_log2_outlier_score"
  )
)

# silhouette index
sc3_plot_silhouette(sce, k = 3)

# expression matrix after cell and gene filters
sc3_plot_expression(
  sce, k = 3,
  show_pdata = c(
    "cell_type1",
    "sc3_3_clusters",
    "sc3_3_log2_outlier_score"
  )
)

# cluster stability: how often does it appear in every solution for different k
sc3_plot_cluster_stability(sce, k = 3)

# de genes
sc3_plot_de_genes(sce, k = 3)

# marker genes
sc3_plot_markers(sce, k = 3)
