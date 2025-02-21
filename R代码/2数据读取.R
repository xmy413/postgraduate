
if(!require(multtest))install.packages("multtest") #多重假设检验，提供了多种方法来控制多重比较的错误率
if(!require(Seurat))install.packages("Seurat") #单细胞数据分析工具包，提供了从原始数据读取、质量控制、数据标准化、特征选择、降维（如 PCA、t-SNE、UMAP 等）、聚类、细胞类型鉴定、差异表达分析、基因集富集分析等一系列完整的单细胞数据分析流程和功能。
if(!require(dplyr))install.packages("dplyr") #专注于数据操作和处理，提供了一组简洁高效的数据处理函数
if(!require(mindr))install.packages("mindr") #构建、可视化和操作思维导图结构
if(!require(mindr))install.packages("tidyverse") #提供了一套方便的数据处理和可视化的工具和方法

#初始化，清理工作空间，节省内存
rm(list = ls()) #利用ls()函数返回当前环境中的所有变量名称给list，最后使用rm()函数从内存中删除

#读取数据
##使用seurat包中Read10X函数:专门用于读取 10X Genomics 输出数据
##pbmc.data:常为稀疏矩阵的数据结构，行为基因、列代表细胞条形码（每个单独细胞）、矩阵中的元素值为相应基因在对应细胞中的表达量
##matrix.mtx:存储基因表达矩阵信息
##genes.tsv:存储基因相关信息，比如基因名称、基因 ID 等
##barcodes.tsv:存储细胞条形码信息，用于标识不同的细胞
pbmc.data = Read10X(data.dir = "D:/xmy/Rproject/单细胞数据基础分析学习手册/数据读取/filtered_gene_bc_matrices/hg19") 
#pbmc.data包含了2700个细胞中32738个基因的表达情况
#其每个元素对应了特定基因在特定细胞中的表现情况，较低的值表示该基因在这个细胞中表达较低，可能是该基因在这个细胞的生理过程中不太活跃；较高的值则表示基因在这个细胞中有较高的表达，可能在这个细胞的功能执行、信号传导等过程中发挥重要作用。

pbmc = CreateSeuratObject(counts = pbmc.data, project = "pbmc3k") #筛选条件：,min.cells = 3(一个基因至少在3个细胞中表达),min.features = 200(一个细胞至少有200个feature才能保留)


#### 仅有一个稀疏矩阵时的读取方法 ####
matrix_data <- read.table("D:/xmy/Rproject/单细胞数据基础分析学习手册/数据读取/single_cell_datamatrix.txt",sep="\t",header=T,row.names=1)

dim(matrix_data) #包含13714个基因和2700个细胞
seurat_obj <- CreateSeuratObject(counts = matrix_data)


#### 读取RDS文件 ####
rm(list = ls())
pbmc <- readRDS("D:/xmy/Rproject/单细胞数据基础分析学习手册/数据读取/panc8.rds")

saveRDS(pbmc,"pbmc.rds")


##了解seurat结构的对象
library(tidyverse)
str(pbmc)
library(mindr)
(out <- capture.output(str(pbmc)))
out2 <- paste(out,collapse = "\n")
mm(gsub("\\.\\.@","# ",gsub("\\.\\. ","#",out2)),type="text",root="Seurat")



