library(ggplot2)
library(scales)

args <- commandArgs(trailingOnly = TRUE)

ifname = args[1]
ifname.exact = args[2]
ofname = args[3]
ofname_histogram = args[4]

OBSERVED_ERROR_Y_LABEL = "Observed percent error"
PERCENT_ERROR_X_LABEL = "Guess number"
COLNAMES_EXACT = c("pwd", "guess.number")
COLNAMES_BROWSER = c("pwd", "guess.number")

browser <- read.delim(ifname, sep = "\t", quote = "")
actual <- read.delim(ifname.exact, sep = "\t", quote = "")
colnames(actual) <- COLNAMES_EXACT
colnames(browser) <- COLNAMES_BROWSER
both.values <- merge(browser, actual, by = "pwd")
both.values$percent.error <- ((
    both.values$guess.number.y - both.values$guess.number.x)
    / both.values$guess.number.y)
p <- ggplot(both.values, aes(x = guess.number.y, y = percent.error))
p <- p + geom_point()
p <- p + scale_x_log10(breaks = 10^seq(1,25, 2),
                       limits = c(1, 10^19),
                       labels = trans_format("log10", math_format(10^.x)))
p <- p + xlab(PERCENT_ERROR_X_LABEL)
p <- p + theme_bw()
p <- p + scale_y_continuous(labels = percent)
p <- p + ylab(OBSERVED_ERROR_Y_LABEL)
ggsave(filename = ofname, plot = p)

if (!is.na(ofname_histogram)) {
    p <- ggplot(both.values, aes(x = percent.error))
    p <- p + geom_histogram()
    p <- p + theme_bw()
    p <- p + scale_y_continuous()
    p <- p + scale_x_continuous(labels = percent)
    p <- p + ylab("Number of passwords")
    p <- p + xlab("Percent error")
    ggsave(filename = ofname_histogram, plot = p)
}
