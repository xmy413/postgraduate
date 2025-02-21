if(!require(multtest))install.packages("multtest") #多重假设检验，提供了多种方法来控制多重比较的错误率
if(!require(Seurat))install.packages("Seurat") #单细胞数据分析工具包，提供了从原始数据读取、质量控制、数据标准化、特征选择、降维（如 PCA、t-SNE、UMAP 等）、聚类、细胞类型鉴定、差异表达分析、基因集富集分析等一系列完整的单细胞数据分析流程和功能。
if(!require(dplyr))install.packages("dplyr") #专注于数据操作和处理，提供了一组简洁高效的数据处理函数
if(!require(patchwork))install.packages("patchwork") 
if(!require(R.utils))install.packages("R.utils")

rm(list = ls())
#download.file('http://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz','my.pbmc.gz')

#加载数据
pbmc.data <- Read10X(data.dir = "D:/xmy/Rproject/单细胞数据基础分析学习手册/3单样本分析/filtered_gene_bc_matrices/hg19/")
#创建Seurat对象
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbm3k", min.cells = 3, min.features = 200) #包含初步过滤：①基因在至少3个细胞中表示才能保留下来；②每个细胞至少表达200个基因才能保留下来

pbmc
##数据观察
nrow(pbmc.data)  #初始有32738个基因
nrow(pbmc)  #初筛过后有13714个基因
ncol(pbmc.data) #初始有32738个细胞
ncol(pbmc) #初始有32738个细胞
test_value1 <- pbmc[1999, 413]
test_value1

#"桥"思想另存数据
counts_data <- GetAssayData(pbmc, assay = "RNA", slot = "counts")
lalala <- as.data.frame(counts_data)
#将上述数据保存
write.table(lalala,'mycount.txt',sep = '\t') #证明lalala为稀疏矩阵，行为基因名，列为细胞名

pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc,pattern = "^MT-")
#调取线粒体的比例
#if percent过高（线粒体&RNA含量过高），可能存在：
#1.细胞核不活跃，活性较差
#2.本身细胞含量就高，根据研究的组织器官而定

#鼠源的需要换为mt(小写)

head(pbmc@meta.data,5)

VlnPlot(pbmc,features = c("nFeature_RNA","nCount_RNA","percent.mt"), ncol = 3)

plot1 <- FeatureScatter(pbmc,feature1 = "nCount_RNA",feature2 = "percent.mt")
plot2 <- FeatureScatter(pbmc,feature1 = "nCount_RNA",feature2 = "nFeature_RNA")
if(!require(patchwork))install.packages("patchwork") 
CombinePlots(plots = list(plot1,plot2)) #CombinePlots()来源于patchwork包


pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5) #筛选数据（质控）
#nFeature_RNA 过少表示低质量细胞；过多表示多细胞的混合体，在建库的时候没有形成单细胞悬液

pbmc
ncol(as.data.frame(pbmc[["RNA"]]@layers$counts))
#counts_data1 <- GetAssayData(pbmc, assay = "RNA", slot = "counts")
#ncol(counts_data1)
#52行的平意替换（53-54）

pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000) #存储表达量值
#每一个细胞的标准化，对数据进行归一化处理，即计算出单个基因在整个细胞基因表达中的所占的比例
#TPM(Transcripts Per Million):百万转录本数，一种用于评估和标准化基因表达水平的度量单位
#数据存放在pbmc-assays-RNA-layers-data中

#寻找高变基因
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
#取出高变基因
top10 <- head(VariableFeatures(pbmc), 10)
top10

#高变基因可视化
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2 #plot variable features with and without labels 
#将两个图画在一起，高变基因表达&正常基因表达&前十个高变基因表达

#每一个基因在所有细胞中的标准化
pbmc <- ScaleData(pbmc, features = rownames(pbmc)) #缩放所有基因
#在ScaleData中默认是将高变基因进行
#pbmc <- ScaleData(pbmc) ##仅缩放高变基因


pbmc <- RunPCA(pbmc, features = rownames(pbmc))
###########问题：PCA(主成分分析)是什么？？###############
print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca")
DimHeatmap(pbmc,dims = 1, cells = 500, balanced = TRUE)
DimHeatmap(pbmc,dims = 1:15, cells = 500, balanced = TRUE)

#高阶PCA分析
pbmc <- JackStraw(pbmc, num.replicate = 100)
pbmc <- ScoreJackStraw(pbmc, dims = 1:20)
JackStrawPlot(pbmc, dims = 1:20)
ElbowPlot(pbmc)

#根据变化趋势选取前10个PCA
#正式开始对细胞进行分群
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5) #resolution 正常范围在0.4~1.2之间，根据后续的时间进行不断调整（类似与超参数）
#resolution越大，获取的细胞类群就越多
pbmc <- RunUMAP(pbmc, dims = 1:10)
DimPlot(pbmc, reduction = "umap")

pbmc <- RunTSNE(pbmc, dims = 1:10)
DimPlot(pbmc, reduction = "tsne")

#寻找细胞的特异性Marker
cluster5.markers <- FindMarkers(pbmc, ident.1 = 5, ident.2 = c(0,3), min.pct = 0.25)
#一类类群为5号细胞群，二类类群为0~3号类群
pbmc.markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
#一类类群为0~9(各是各的)，二类类群为除当前类群外的剩余类群
#cluster0test.markers <- FindMarkers(pbmc, ident.1 = 0, ident.2 = c(1:9), min.pct = 0.25)

if(!require(dplyr))install.packages("dplyr") #管道符的来源
pbmc.markers %>% group_by(cluster) %>% top_n(n=2, wt = avg_log2FC)
#top2 <- pbmc.makers %>% group_by(cluster) %>% top_n(n=2, wt = avg_log2FC) #每个cluster中排名前2的基因表达
VlnPlot(pbmc, features = c("NKG7", "PF4"), slot = "counts", log = TRUE)
FeaturePlot(pbmc, features = c("MS4A1","GNLY","CD3E","CD14","FCER1A","FCGR3A","LYZ"))

Gentop10 <- pbmc.markers %>% group_by(cluster) %>% top_n(n=10, wt = avg_log2FC)
DoHeatmap(pbmc, features = Gentop10$gene) + NoLegend()
#找到细胞类型的Marker基因
#可以对细胞类型进行注释（不推荐使用自动注释SingleR），推荐使用文献检索寻找到你所做的组织/器官类似的文章中的Marker基因的参考
#http：//biocc.hrbmu.edu.cn/cellMarker/
############ 问题：没听懂 ###################

new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T", "FCGR3A+ Mono", "NK", "DC", "Platelet", "test")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, new.cluster.ids)
DimPlot(pbmc, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()

#查看各类版本
sessionInfo()

#保存Seurat对象
saveRDS(pbmc, 'pbmc.rds')

#再次读取
pbmc <- readRDS('D:/xmy/Rproject/单细胞数据基础分析学习手册/3单样本分析/pbmc.rds')
DimPlot(pbmc, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()









