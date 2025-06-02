# setwd('D:/Project/ProteomeGC_ZheYi/reports/peps/Vis/')
#BiocManager::install('maftools')
library(maftools)
library(dplyr)

fake.cn <- read.table('./fake_cn.txt',header = 1)
#fake.cn %>% head()

laml <- read.maf(
  './Somatic_mutation_fixed.maf',
#  clinicalData = './ZJU_type.txt',
)

# mafr <- annovarToMaf(
#   annovar = './Somatic_snp_indel_merged_fixed.txt',
#   MAFobj = F,
# )

mafr <- read.maf('./Syn13.maf')

mmaf <- merge_mafs(
  mafs=c(laml,mafr),
  cnTable = './fake_cn.txt'
)

all_mut_type <- (mmaf@variant.classification.summary %>% names)[2:8]
#vc_cols = RColorBrewer::brewer.pal(n = length(all_mut_type), name = 'YlOrRd')
#names(vc_cols) <- all_mut_type
vc_cols <- list(
  Frame_Shift_Del='#98c6ff',
  Frame_Shift_Ins='#5c8ad6',
  In_Frame_Del='#3c6e3b',
  In_Frame_Ins='#87a171',
  Missense_Mutation='#ffcb93',
  neoantigen='#cc7669',
  Nonstop_Mutation='#677a8f'
)
vc_cols <- as.character(vc_cols)
names(vc_cols) <- all_mut_type

gene.rank <- read.csv('./gene_rank.csv')

oncoplot(
  maf = mmaf,
  genes = gene.rank[gene.rank$Gene >=3,]$X,
  fontSize = 0.5,
  legendFontSize = 0.8,
  # top = 50,
  colors = vc_cols,removeNonMutated = F,
  showTumorSampleBarcodes = T,
  sampleOrder=droplevels(mmaf@variants.per.sample$Tumor_Sample_Barcode %>% sort(),c('Syn_01M','Syn_05M')),
  )
