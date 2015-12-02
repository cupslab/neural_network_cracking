library(ggplot2)
library(scales)

IFNAME = 'monte_carlo.tsv'
IFNAME_EXACT = 'exact_numbers.tsv'

PERCENT_ERROR_Y_LABEL = "Estimated percent error (95% confidence interval)"
OBSERVED_ERROR_Y_LABEL = "Observed percent error"
PERCENT_ERROR_X_LABEL = "Guess number"
LEGENED_LABEL = "Outside Confidence Interval"
OFNAME_ESTIMATE = 'monte_carlo_error_estimate.pdf'
OFNAME_OBSERVED = 'monte_carlo_observed_error.pdf'
COLNAMES = c("pwd", "prob", "guess.number", "Steve", "sample.size", "std.error",
             "percent.std.error")
COLNAMES_EXACT = c("pwd", "prob", "guess.number")

add_scales <- function (p) {
    p <- p + scale_x_log10(breaks = 10^seq(0,26, 3),
                           labels = trans_format("log10", math_format(10^.x)))
    p <- p + xlab(PERCENT_ERROR_X_LABEL)
    p <- p + theme_bw()
    p
}

estimates <- read.table(IFNAME, sep = "\t", quote = "\"")
estimates[7] <- (estimates[6] / estimates[3])
colnames(estimates) <- COLNAMES
p <- ggplot(estimates, aes(guess.number, percent.std.error))
p <- p + geom_point()
p <- add_scales(p)
p <- p + scale_y_continuous(labels = percent, limits = c(0, .5))
p <- p + ylab(PERCENT_ERROR_Y_LABEL)
ggsave(filename = OFNAME_ESTIMATE, plot = p)

actual <- read.table(IFNAME_EXACT, sep = "\t", quote = "\"")
colnames(actual) <- COLNAMES_EXACT
total <- merge(estimates, actual, by = "pwd")

bothvalues <- total[total$guess.number.y > 0, ]
print("Buggy passwords")
print(bothvalues[ bothvalues$prob.x != bothvalues$prob.y, ]$pwd)
bothvalues <- bothvalues[ bothvalues$prob.x == bothvalues$prob.y, ]
bothvalues$actual.percent.error <- (abs(
    bothvalues$guess.number.y - bothvalues$guess.number.x) /
    bothvalues$guess.number.y)
bothvalues$outside.interval <- ifelse(abs(
    bothvalues$guess.number.y - bothvalues$guess.number.x) <
    bothvalues$std.error, 0, 1)
p <- ggplot(bothvalues,
            aes(x = guess.number.x,
                y = actual.percent.error,
                colour = factor(outside.interval)))
p <- p + geom_point()
p <- add_scales(p)
p <- p + scale_color_discrete(LEGENED_LABEL)
p <- p + scale_y_continuous(labels = percent, limits = c(0, .5))
p <- p + ylab(OBSERVED_ERROR_Y_LABEL)
ggsave(filename = OFNAME_OBSERVED, plot = p)

print(sum(bothvalues$outside.interval) / nrow(bothvalues))
print(sum(bothvalues$outside.interval))
print(bothvalues[ bothvalues$actual.percent.error > .10, ]$pwd)
print(nrow(bothvalues[ bothvalues$prob.x != bothvalues$prob.y, ]))
