library(ggplot2)
afile = read.csv(commandArgs(trailingOnly = TRUE)[1], header = F)
bfile = read.csv(commandArgs(trailingOnly = TRUE)[2], header = F)
a = afile[ afile$V1 != -1, ]
b = bfile[ bfile$V1 != -1, ]
df <- data.frame(x = c(a, b), g = c(rep(1, length(a)), rep(2, length(b))))
ggplot(df, aes(x = x, colour = as.factor(g))) + stat_ecdf()
ggsave('out.pdf')
