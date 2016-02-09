#################################################################################
# Process lookup results and plot
# Version: 1.6b
#
# Author: Saranga Komanduri (sarangak@cmu.edu)
# Modifications by Rich
#  Line lengths are now taken from the totalcounts file
#  (There must be a total count for each condition)
#  Labels are larger (16 point)
#  Lines are thicker
#
#
# Execution: This file can be used either as a library meant to be sourced from
#   other R scripts, or run from the command line in the following manner:
#   Rscript PlotResults.R makeplot <filename>
#       Write pdf guessing plot to given file
#
# Input: Lookup results files (lookupresults and totalcounts, where matching
#   files have the same extension)
#   Results are always presented separated by values of the "condition" column
#   in the lookup results files.  It is assumed that data points with the same
#   condition name were produced by the same experiment.
#
# Output: Various graphs of cracking results, statistical analysis results,
#   and other results files relevant to cracking-results analysis
#

#####
##### Load packages
#####
for (pkg in c("plyr", "stringr",  "reshape2", "ggplot2", "scales", "gridExtra",
              "survival", "plotrix", "xtable", "hash", "grid")) {
  if ((pkg %in% installed.packages()[,1])) {
    library(pkg, character.only = T)
  } else {
    # Try installing package and stop if that fails (install.packages does not return an error level)
    install.packages(pkg,
                     repos = "http://lib.stat.cmu.edu/R/CRAN/",
                     dependencies = T)
    if ((pkg %in% installed.packages()[,1])) {
      library(pkg, character.only = T)
    } else {
      stop(paste("Package not found:", pkg))
    }
  }
}

#####
##### Initialize environment
#####
# Options
theme_set(theme_bw())  # Set the default theme for plotting to have a white background and black foreground
options(stringsAsFactors = F,  # If I need factors I explicitly define them
        width = 1000,  # When output is written to a file, it can be unwieldy to have lines wrapped at 80
        warn = 1)


# Constants - these are plot configuration variables
if (!exists("kExtraFonts")) {
  kExtraFonts = FALSE
}
if (kExtraFonts) {
  library(extrafont)
  if (!exists("kExtraFontsLoaded")) {  # don't do this twice in the same session
    loadfonts()  # Load fonts from extrafont database
    kExtraFontsLoaded <- 1
  }
}

if(!exists("defaultybreaks")){
    defaultybreaks <- c()
}
if(!exists("defaultylimits")){
    defaultylimits <- c()
}

if (!exists("kPalette")) {
  # Start with Dark2 from Brewer palette and add color-blind palette so there are more entries -- note that names are not used in the code
  kPalette <- c(mediumseagreen = "#1B9E77", darkorange3 = "#D95F02", lightslategray = "#7570B3", violetred2 = "#E7298A", olivedrab = "#66A61E", darkgoldenrod2 = "#E6AB02", goldenrod4 = "#A6761D", gray40 = "#666666",
               steelblue2 = "#56B4E9", khaki2 = "#F0E442", deepskyblue4 = "#0072B2", pink3 = "#CC79A7", black = "#000000", orange2 = "#E69F00", cyan4 = "#009E73")
}

if (!exists("kLabelSmidgeFactor")) {
  # Start with Dark2 from Brewer palette and add color-blind palette so there are more entries -- note that names are not used in the code
  kLabelSmidgeFactor <- 1.2
}

if (!exists("kLabelVerticalDistance")) {
  # Start with Dark2 from Brewer palette and add color-blind palette so there are more entries -- note that names are not used in the code
  kLabelVerticalDistance <- 0.03
}

if (!exists("kPlotMarginRight")) {
  # Start with Dark2 from Brewer palette and add color-blind palette so there are more entries -- note that names are not used in the code
  kPlotMarginRight <- 0.2
}


#####
##### Auxilliary functions
#####
caterr <- function (...) {
  # Trivial function for writing output to stderr
  cat(..., file = stderr())
}

cerr <- function (...) {
  # Trivial function for redirecting output from a print function to stderr
  capture.output(..., file = stderr())
}

FormatPropTable <- function(intable, dec = 2) {
  # Print a prop.table with given number of decimal places and percent sign
  y <- apply(round(intable * 100, dec), c(1,2), paste, "%", sep = "")
  return(y)
}

FormatPropTable2 <- function(intable, dec = 2) {
  # Print a prop.table with given number of decimal places and percent sign using sprintf so it has zeroes after the decimal
  y <- apply(intable, c(1,2), sprintf, fmt = paste("%0.", dec, "f%%", sep = ""))  # %% means add a literal percent
  return(y)
}

RecodeVector <- function(vector, oldvalues, newvalues) {
  # Function for recoding values of a vector based on a vector of matching oldvalues and newvalues
  # Ex. oldvalues = c("Male", "Female") and newvalues = c("M", "F") will change all instances of "Male" and "Female" in the given vector with "M" and "F", leaving the other values alone
  if (length(oldvalues) != length(newvalues)) {
    stop("oldvalues and newvalues must be the same length!")
  }

  # Make copy of vector and replace values
  vec2 <- vector
  for (i in seq_along(oldvalues)) {
    vec2[which(vec2 %in% oldvalues[i])] <- newvalues[i]
  }

  return(vec2)
}

trim <- function (x) gsub("^\\s+|\\s+$", "", x)  # Function for trimming whitespace from a string

safe.ifelse <- function(cond, yes, no) {
  # The built-in ifelse has a habit of mangling the class of input
  return(structure(ifelse(cond, yes, no), class = class(no)))
}

#####
##### Library functions
#####
ValidLookupResults <- function(lookup.results) {
  # Validate that the lookup.results data frame has condition and guess.number
  #   columns, and that guess.numbers are valid
  # Output: warnings and return value of FALSE if not valid, else TRUE
  #
  if (class(lookup.results) != "data.frame") {
    warning("lookup.results is not of class \"data.frame\"!")
  } else if (!("condition" %in% colnames(lookup.results))) {
    warning("\"condition\" column not found in lookup.results!")
  } else if (!("guess.number" %in% colnames(lookup.results))) {
    warning("\"guess.number\" column not found in lookup.results!")
  } else if (any(is.na(lookup.results$guess.number))) {
    warning("Missing / NA guess number values found!")
  } else {
    # All tests passed!
    return(TRUE)
  }
  return(FALSE)
}


ValidGuessCutoffs <- function(lookup.results, guesscutoffs) {
  # Validate guesscutoffs - every unique condition in lookup.results
  #   should have a corresponding value in guesscutoffs
  #
  # Output: warnings and return value of FALSE if not valid, else TRUE
  #
  if (!ValidLookupResults(lookup.results)) {
    warning("lookup.results not valid in ValidGuessCutoffs!")
  } else if (class(guesscutoffs) != "hash") {
    warning("guesscutoffs is not a hash!")
  } else {
    # Compare conditions in lookup.results and guesscutoffs, using sort to ignore order
    lookup.results.conditions <- sort(unique(lookup.results$condition))
    guesscutoffs.conditions <- sort(keys(guesscutoffs))
    # Use all.equal to compare conditions -- this requires the isTRUE function
    #   for comparison, see help("all.equal") for more information.
    if (!isTRUE(all.equal(lookup.results.conditions,
                          guesscutoffs.conditions))) {
      warning("Conditions in guesscutoffs do not match conditions in lookup.results!",
              " Difference found: ",
              all.equal(lookup.results.conditions,
                        guesscutoffs.conditions))
      # If this error occurs, it will be difficult to debug, so output more
      #   debugging info than usual.
      caterr("Conditions in lookup.results:",
             lookup.results.conditions, "\n",
             "Conditions in guesscutoffs:",
             guesscutoffs.conditions, "\n\n")
    } else {
      return(TRUE)
    }
  }
  return(FALSE)
}


ComputeStatisticsForGuessed <-
  function(lookup.results,
           guesscutoff = max(lookup.results$guess.number)) {
  # This function computes chi-square statistics using percent guessed per
  #   condition, at various logarithmic points: 1e3, 1e6, ..., guesscutoff
  #
  # Inputs:
  # lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # guesscutoff
  #     maximum guess number evaluated for this experiment
  #
  # Output: tables and statistical test results to stdout
  #
  if (!ValidLookupResults(lookup.results)) {
    stop("data.frame given to ComputeChiSquares function is not in valid format! Check warning messages for more information.\n")
  }

  cat("Examining guesspoints across conditions:\n")
  guesspoint <- 1
  while (guesspoint < guesscutoff) {
    # Iterate over logarithmic points: 1e3, 1e6, ..., guesscutoff
    guesspoint <- guesspoint * 1e3
    if (guesspoint > guesscutoff) {
      guesspoint <- guesscutoff
    }

    # Determine how many passwords were guessed up to this point
    guesses <- with(lookup.results, guess.number > 0 & guess.number <= guesspoint)
    if (sum(guesses) == 0) {
      cat("No passwords guessed for either condition at",
          signif(guesspoint, 3),    # Use signif so the exact guess number is not output (only the 3 most significant digits)
          "guesses.\nNo statistics computed.\n\n")
    } else {
      # Output tables of frequencies, percentages, and chi-square test results
      cat("Table for conditions at guess number:", signif(guesspoint, 3), ":\n")
      guesstable <- table(lookup.results$condition, guesses)
      # Make usable table axis titles
      ytitle <- ifelse(guesspoint == guesscutoff,
                       "under.guess.cutoff",
                       paste("under10^",
                             round(log10(guesspoint), 2),
                             "guesses", sep = ""))
      names(dimnames(guesstable)) <- c("Condition", ytitle)
      print.table(guesstable)
      cat("\n  with percentages:\n")
      print.table(FormatPropTable(prop.table(guesstable, 1)))
      cat("\nChi-square test for above table:")
      print(chisq.test(guesstable))
      cat("\n\n")
    }
  }
  cat("\n\n")
}


MakeLatexTableOfGuessing <-
  function(lookup.results,
           guesscutoff = max(lookup.results$guess.number)) {
  # This function uses the xtable package to produce a Latex table
  #   (more specifically, rows of a tabular environment)
  #   for percent guessed per condition, at various logarithmic points:
  #   1e3, 1e6, ..., guesscutoff
  # The Latex table assumes use of the booktabs package.
  # If you want the conditions displayed in a specific order, make the condition
  #   column of lookup.results a factor and set the factor levels to the
  #   order that you want.  By default, conditions are sorted alphabetically.
  #
  # Inputs:
  # lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # guesscutoff
  #     maximum guess number evaluated for this experiment
  #
  # Output: Latex table to stdout
  #
  if (!ValidLookupResults(lookup.results)) {
    stop("lookup.results data frame given to MakeLatexTable function is not in valid format! Check warning messages for more information.\n")
  }

  if (!is.factor(lookup.results$condition)) {
    lookup.results$condition <- factor(lookup.results$condition,
                                       levels = sort(unique(lookup.results$condition)))
  }
  # Create a data.frame to store percentages - only has condition column initially
  guessing.data <- data.frame(Condition = levels(lookup.results$condition))
  guesspoint <- 1
  while (guesspoint < guesscutoff) {
    # Iterate over logarithmic points: 1e3, 1e6, ..., guesscutoff
    guesspoint <- guesspoint * 1e3
    if (guesspoint > guesscutoff) {
      guesspoint <- guesscutoff
    }

    # Get raw percentages up to this point
    guesses <- with(lookup.results, guess.number > 0 & guess.number <= guesspoint)
    guesstable <- prop.table(table(lookup.results$condition, guesses), 1)

    # Format percentages and add to table
    columnheading <- ifelse(guesspoint == guesscutoff,
                            "Cutoff",
                            paste("$10^{", round(log10(guesspoint)), "}$", sep = ""))
    guessing.data[[columnheading]] <- FormatPropTable2(guesstable * 100, dec = 1)[,2]
  }

  # Reformat table to xtable and output
  print(xtable(guessing.data),
        hline.after = NULL,
        floating = F,
        sanitize.colnames.function = function(x) {x},
        include.rownames = F,
        add.to.row=list(pos=list(-1,0, nrow(xtable(guessing.data))),
                        command=c("\\toprule\n",
                                  "\\midrule\n",
                                  "\\bottomrule\n")))
}


ComputeCumulativePercentages <- function(lookup.results) {
  # For each data point in the data set, use the ecdf built-in function to
  #   compute cumulative cracking proportions.
  #
  # Output: lookup.results with an additional proportion column
  #
  if (!ValidLookupResults(lookup.results)) {
    stop("data.frame given to ComputeCumulativePercentages function is not in valid format! Check warning messages for more information.\n")
  }

  # Before applying ecdf, it is necessary to set the guess number of uncracked
  #   passwords to Inf so they are not counted as cracked.
  original.results <- lookup.results
  lookup.results$guess.number.adj <-
    ifelse(lookup.results$guess.number < 0,
           Inf,
           lookup.results$guess.number)

  # Use ddply to split the data by levels of condition and compute cumulative
  #   percentages individually
  gndata <- ddply(lookup.results,
                  "condition",
                  mutate,
                  proportion = ecdf(guess.number.adj)(guess.number.adj))

  # Augment the original results with the new data and return
  return(join(original.results, gndata))
}


PlotGuessingCurves <- function(lookup.results,
                               guesscutoff = max(lookup.results$guess.number),
							   guesscutoffs = hash(),
                               graph.equalizecutoff = T,
                               xlimits = NA, ylimits = NA,
                               ybreaks = NA,
                               xlog = T, ylog = F,
                               plottitle = "Guessing curves by condition",
                               xtitle = "Guesses",
                               ytitle = "Percent guessed",
                               labelsoncurves = T,
                               labelsalign = T,
                               continuecurvestocutoff = F,
                               logxbreaks = 10^(0:15)[rep(c(F,T))],
                               pointatorigin = T,
                               config = data.frame()) {
  # This function generates guessing curves to show how the percent guessed
  #   varies over levels of the condition column.
  #
  # Inputs:
  # lookup.results
  #     data.frame where each row corresponds to an independent data point
  #     the "condition" column must exist and contain condition labels
  #     the "guess.number" column must exist and contain guess numbers
  #
  # guesscutoff
  #     maximum guess number evaluated for this experiment
  #     used to set the right-limit of the x-axis
  #
  # graph.equalizecutoff
  #     if TRUE, curves are truncated on the right to the guesscutoff
  #
  # xlimits / ylimits
  #     x-axis and y-axis limits, respectively, each specified as a 2-element
  #     vector, e.g. c(1,1e6)
  #
  # xlog / ylog
  #     if TRUE, the x-axis (or y-axis) is plotted in log-scale
  #
  # plottitle / xtitle / ytitle
  #     graph titles
  #
  # labelsoncurves
  #     if TRUE, text labels will be placed alongside the right end of each curve.
  #     if FALSE, a standard legend will be used (default if ylog = T).
  #
  # labelsalign
  #     this option has no effect if labelsoncurves is FALSE
  #     if TRUE, text labels will be aligned to the guesscutoff, otherwise
  #     labels will be placed near the maximum guess number for each condition
  #
  # continuecurvestocutoff
  #     for safety, this option has no effect unless graph.equalizecutoff
  #     is TRUE (otherwise this might create fake data points that are inaccurate)
  #     if TRUE, an additional point will be added to each curve at the guess
  #     cutoff position so the curve continues to the cutoff
  #     this implicitly sets labelsalign to true also
  #
  # logxbreaks
  #     x-axis major tick mark locations, when xlog is TRUE
  #
  # Output: a plot of guessing curves to the current output device
  #
  if (length(guesscutoff) != 1 || !is.numeric(guesscutoff)) {
    stop("guesscutoff parameter is not in the correct format! ",
         "If you have a hash of guesscutoffs, try passing max(values(guesscutoffs)) ",
         "as the guesscutoff value (or min depending on what you want).")
  }

  if (graph.equalizecutoff) {
    # Truncate the data set of observations above the cutoff
    lookup.results <- subset(lookup.results,
                             guess.number <= guesscutoff)
  }
  guessing.data <- ComputeCumulativePercentages(lookup.results)
  # Throw away the uncracked passwords, we don't need them now
  guessing.data <- subset(guessing.data,
                          guess.number > 0)
  # Determine max cracked per condition and max guess number.
  # This is used for sorting conditions and might be used for the
  #   y-coordinates and x-coordinates of the text labels.
  maxproportion.data <- ddply(guessing.data,
                              "condition",
                              summarize,
                              MaxGN = max(guess.number),
                              MaxProp = max(proportion))

  if (graph.equalizecutoff) {
    graph.x.limits <- c(1, guesscutoff * kLabelSmidgeFactor)
  } else {
    graph.x.limits <- c(1, max(maxproportion.data$MaxGN) * kLabelSmidgeFactor)
  }
  if(pointatorigin){
      for(cond in unique(guessing.data$condition)){
          templaterow <- guessing.data[1, ]
          templaterow$guess.number <- 0
          templaterow$proportion <- 0
          templaterow$condition <- cond
          guessing.data[nrow(guessing.data)+1,] <- templaterow
      }
  }
  if (continuecurvestocutoff & graph.equalizecutoff) {
    # Add a point to each curve
    for (cond in unique(guessing.data$condition)) {
      # Replicate the row with maximum proportion
      maxtofind <- maxproportion.data[maxproportion.data$condition == cond,
                                      "MaxProp"]
      rowtofind <- which(guessing.data$condition == cond &
                         guessing.data$proportion == maxtofind)
      templaterow <- guessing.data[rowtofind,]
      # Modify the guess.number of this row to be the guess cutoff so that
      #   an additional data point is created
	  if (is.empty(guesscutoffs)) {
		  templaterow$guess.number <- guesscutoff
	  } else {
		  templaterow$guess.number <- guesscutoffs[[as.character(cond)]]
		  maxproportion.data <- ddply(guessing.data,
		                              "condition",
		                              summarize,
		                              MaxGN = max(guess.number),
		                              MaxProp = max(proportion))
	  }

      # Add this row to the data frame
      guessing.data[nrow(guessing.data)+1,] <- templaterow
    }
    # Force labelsalign, otherwise the graph won't look right
    #labelsalign = F
  }

  if (!is.na(xlimits)) {
    guesscutoff <- xlimits[2]
    graph.x.limits <- c(xlimits[1], guesscutoff * kLabelSmidgeFactor)
    # Truncate the data set of observations above the top xlimit
    guessing.data <- subset(guessing.data,
                            guess.number <= xlimits[2])
  }
  if (labelsoncurves) {
    # Set x-coordinate
    if (labelsalign) {
		# GoHere to adjust based on location
      maxproportion.data$LabelX = rep(guesscutoff * kLabelSmidgeFactor, nrow(maxproportion.data))
    } else {
      maxproportion.data$LabelX = maxproportion.data$MaxGN * kLabelSmidgeFactor
    }
    # For y-coordinate, use the plotrix::spreadout to force all labels to have
    #   a minimum distance from each other
    maxproportion.data$LabelY = spreadout(maxproportion.data$MaxProp,
                                          mindist = kLabelVerticalDistance)
  }
  if (is.na(ylimits)) {
    if (labelsoncurves) {
      # The label-spreading operation can make the top label go higher than
      #   the highest data point
      graph.y.limits <- c(0, max(maxproportion.data$LabelY))
    } else {
      graph.y.limits <- c(0, max(maxproportion.data$MaxProp))
    }
  } else {
    graph.y.limits <- ylimits
  }
  if(length(defaultylimits) > 0){
      graph.y.limits <- defaultylimits
  }

  # Sort condition names in descending order of proportion guessed
  cond.order <- arrange(maxproportion.data,
                        desc(MaxProp)
                        )$condition
  guessing.data$condition <- factor(guessing.data$condition,
                                    levels = unique(as.character(cond.order)))

  # Make the graph!
  print(head(guessing.data))
  baseplot <- ggplot(guessing.data,
                     aes_string(x = "guess.number",
                                y = "proportion",
                                color = "condition",
                                linetype = "condition"))

  ## look for colors in the config file
  colors <- unname(kPalette)
  for(i in 1:length(cond.order)){
    configed <- config[ config$name == cond.order[i], ]
    if(length(rownames(configed)) > 0){
      colors[i] <- configed$color[1]
    }
  }

  ## look for line types in config files
  data.linetypes <- rep("solid", length(cond.order))
  for(i in 1:length(cond.order)){
    configed <- config[ config$name == cond.order[i], ]
    if(length(rownames(configed)) > 0){
      data.linetypes[i] <- configed$linetype[1]
    }
  }

  plot.plus.curves <- baseplot +
    geom_step(## size = 1.3
              ) + #geom_point(shape = 1) +  # Use thin step line overlaid with hollow circle for data point
    # Custom theme for plot - these are settings I think look good
    theme(axis.line = element_line(colour = "grey50"),
          legend.title = element_blank(),
          legend.key = element_blank(),
          legend.position = ifelse(labelsoncurves, "none", "right"),
          text = safe.ifelse(kExtraFonts,
                             element_text(family = "Helvetica Neue"),
                             element_text()),
          plot.margin =
            unit(c(0,
                   # Add margin to right side of plot for labels if labels on curves
                   ifelse(labelsoncurves, kPlotMarginRight, 0),
                   0,0), units = "npc"),
          panel.border = element_blank()) +
    scale_linetype_manual(values = data.linetypes) +
    scale_colour_manual(values = colors)
                                        #+
    #labs(title = plottitle)

  # With legend on the right, and cutoff equalized, we want the graph to cut off
  #   exactly on the right side.  Otherwise it looks like there is no data to
  #   the right of the cutoff (as opposed to the graph itself being cut off).
  #   This requires the "expand" argument to the ggplot scale function, but the
  #   scale function is different for each axis type.  This leads to the messy
  #   code below.
  if (xlog) {
    if (labelsoncurves | !graph.equalizecutoff) {
      plot.plus.x.axis <- plot.plus.curves +
        scale_x_log10(name = xtitle,
                      limits = graph.x.limits,
                      labels = trans_format('log10',math_format(10^.x)),
                      breaks = logxbreaks)
    } else {
      plot.plus.x.axis <- plot.plus.curves +
        scale_x_log10(name = xtitle,
                      limits = graph.x.limits,
                      labels = trans_format('log10',math_format(10^.x)),
                      # Use a tiny (0.01) expansion used to cover the circle around each data point
                      expand = c(0.01,0),
                      breaks = logxbreaks)
    }
  } else {
    if (labelsoncurves | !graph.equalizecutoff) {
      plot.plus.x.axis <- plot.plus.curves +
        scale_x_continuous(name = xtitle,
                           limits = graph.x.limits)
    } else {
      plot.plus.x.axis <- plot.plus.curves +
        scale_x_continuous(name = xtitle,
                           limits = graph.x.limits,
                           expand = c(0.01,0))
    }
  }

  # Add y-axis
  if (ylog) {
    plot.plus.axes <- plot.plus.x.axis +
      scale_y_log10(name = ytitle,
                    labels = percent,
                    limits = graph.y.limits)
  } else {
      if(length(defaultybreaks) > 0){
          plot.plus.axes <- plot.plus.x.axis +
              scale_y_continuous(name = ytitle,
                                 labels = percent,
                                 limits = graph.y.limits,
                                 breaks=defaultybreaks)
      } else {
          plot.plus.axes <- plot.plus.x.axis + scale_y_continuous(name = ytitle,
                         labels = percent,
                         limits = graph.y.limits)
      }

  }

  # Increase the size of the axis labels
  plot.plus.axes <- plot.plus.axes + theme(axis.title.x = element_text(size=16))
  plot.plus.axes <- plot.plus.axes + theme(axis.title.y = element_text(size=16))
  plot.plus.axes <- plot.plus.axes + theme(axis.text.x = element_text(size=16))
  plot.plus.axes <- plot.plus.axes + theme(axis.text.y = element_text(size=16))

  # The minor gridlines don't align to log-scale, so turn them off
  if (xlog || ylog) {
    plot.plus.axes <- plot.plus.axes + theme(panel.grid.minor = element_blank())
  }
  # Add log-ticks to axes in log-scale, but not if labelsoncurves is set
  #   Printing labels on curves requires turning off clipping, which makes
  #   the log ticks extremely ugly
  if (!labelsoncurves) {
    if (xlog) {
      if (ylog) {
        plot.plus.axes <- plot.plus.axes + annotation_logticks(sides = "lb")
      } else {
        plot.plus.axes <- plot.plus.axes + annotation_logticks(sides = "b")
      }
    }
  }

  # Finally, add labels to curves if needed, using code from:
  #   http://learnr.wordpress.com/2009/04/29/ggplot2-labelling-data-series-and-adding-a-data-table/
  if (labelsoncurves) {
    plot.plus.axes <- plot.plus.axes +
      geom_text(data = maxproportion.data,
                aes_string(x = "LabelX", y = "LabelY", label = "condition"),
                hjust = 0, vjust = 0.5,
                family = safe.ifelse(kExtraFonts, "Helvetica Neue", "Helvetica"))
  }
  plot.plus.axes <- plot.plus.axes + theme(panel.grid.major = element_line(color="grey"))

  # Render plot, again using code from http://learnr.wordpress.com/2009/04/29/ggplot2-labelling-data-series-and-adding-a-data-table/
  gtable.object <- ggplot_gtable(ggplot_build(plot.plus.axes))
  # Turn off clipping if labels on curves
  gtable.object$layout$clip[gtable.object$layout$name == "panel"] <-
    ifelse(labelsoncurves, "off", "on")
  print(grid.draw(gtable.object))
}


GetStatusFromGuessNumbers <- function(lookup.results) {
  # This function categorizes the guess numbers in lookup.results based on
  #   their numerical code.
  #
  # Inputs:
  # lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # Output: lookup.results data frame with a password.status character column
  #
  if (!ValidLookupResults(lookup.results)) {
    stop("data.frame given to GetStatusFromGuessNumbers function is not in valid format! Check warning messages for more information.\n")
  }

  # Use a combination of ifelse and RecodeVector to categorize guess numbers
  guess.number.adj <-
    as.character(ifelse(lookup.results$guess.number > 0,
                        1,
                        lookup.results$guess.number))
  lookup.results$password.status <-
    RecodeVector(guess.number.adj,
                 c("-3",            "-2",              "-1",                  "1"),
                 c("Beyond cutoff", "Chunk not found", "Structure not found", "Guessed"))
  return(lookup.results)
}


ComputeStatisticsForUnguessed <- function(lookup.results) {
  # This function computes tables on the proportion of unguessed passwords
  #   for each condition.
  #
  # Inputs:
  # lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # Output: tables to stdout
  #
  lookup.results <- GetStatusFromGuessNumbers(lookup.results)

  cat("Original percentages:\n")
  print.table(FormatPropTable(
    prop.table(table(lookup.results$condition,
                     lookup.results$password.status),
               1)))
  cat("\n\n")

  cat("Out of unguessed:\n")
  unguessed.only <- subset(lookup.results, password.status != "Guessed")
  print.table(FormatPropTable(
    prop.table(table(unguessed.only$condition,
                     unguessed.only$password.status),
               1)))
  cat("\n\n")
}


PlotUnguessedProportions <- function(lookup.results) {
  # This function provides similar data to ComputeStatisticsForUnguessed, but
  #   in plot form instead of text tables.
  #
  # Inputs:
  # lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # Output: tables to stdout
  #
  lookup.results <- GetStatusFromGuessNumbers(lookup.results)

  # Make data frame for guessed percentages
  # The inner ddply tabulates the number of passwords in each status category
  #   and the outer ddply converts the frequencies into proportions
  status.proportions <-
    ddply(
      ddply(lookup.results,
            c("condition", "password.status"),
            summarize,
            Frequency = length(password.status),
            .drop = F),
    "condition",
    transform,
    Proportion = Frequency / sum(Frequency))

  # Make a similar data frame but ignore all guessed passwords in the
  #   percentages.  This is to allow comparisons regardless of how
  #   many passwords were cracked in each condition.
  unguessed.only <- subset(lookup.results, password.status != "Guessed")
  unguessed.proportions <-
    ddply(
      ddply(unguessed.only,
            c("condition", "password.status"),
            summarize,
            Frequency = length(password.status),
            .drop = F),
      "condition",
      transform,
      Proportion = Frequency / sum(Frequency))

  # Make two plots and then align then vertically on the same graph
  graph.1 <-
    ggplot(status.proportions,
           aes_string(x = "password.status",
                      y = "Proportion",
                      group = "condition",
                      fill = "condition")) +
    geom_bar(stat = "identity", position = "dodge") +
    theme(text = safe.ifelse(kExtraFonts, element_text(family = "Helvetica Neue"), element_text())) +
    scale_y_continuous(name = "% of condition", labels = percent_format()) +
    scale_x_discrete("Password status", drop = F) +
    scale_fill_manual(values = unname(kPalette), drop = F) +
    labs(title = "Comparison of Proportions (all guess numbers)")
  graph.2 <-
    ggplot(unguessed.proportions,
           aes_string(x = "password.status",
                      y = "Proportion",
                      group = "condition",
                      fill = "condition")) +
    geom_bar(stat = "identity", position = "dodge") +
    theme(text = safe.ifelse(kExtraFonts, element_text(family = "Helvetica Neue"), element_text())) +
    scale_y_continuous(name = "% out of unguessed per condition", labels = percent_format()) +
    scale_x_discrete("Password status", drop = F) +
    scale_fill_manual(values = unname(kPalette), drop = F) +
    labs(title = "Comparison of Proportions (unguessed only)")
  print(grid.arrange(graph.1, graph.2, nrow = 2))
}

LogRanks <- function(lookup.results, guesscutoffs) {
  # This function takes a data frame and named guesscutoffs vector and computes
  #   the log-rank statistic (with rho = 1) between each pair of guessing
  #   curves, and provides Holm-corrected p-values.  The statistic is computed
  #   using the survdiff function from the survival package.
  # The guesscutoff hash must have a mapping for each condition in
  #   lookup.results, otherwise we can't perform a survival analysis!
  #
  # Inputs:
  # lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # guesscutoffs
  #     hash where keys are conditions and values are guess cutoffs
  #
  # Output: tables to stdout
  #
  if (!ValidLookupResults(lookup.results)) {
    stop("lookup.results data frame given to LogRanks function is not in valid format! Check warning messages for more information.\n")
  }
  if (!ValidGuessCutoffs(lookup.results, guesscutoffs)) {
    stop("guesscutoffs hash given to LogRanks function is not in valid format! Check warning messages for more information.\n")
  }

  # Set up data frame for survival analysis
  survival.data <- lookup.results
  # We need to get the guess cutoff for each data point in lookup.results,
  #   based on its condition (guesscutoffs[[]] lookup is not a vectorized
  #   function).
  guesscutoff.column <- sapply(lookup.results$condition,
                               function(element) {
                                 return(guesscutoffs[[element]])
                               },
                               USE.NAMES = F)
  # If a password was cracked, then at time = guess_number the password is dead (1).
  # If not cracked, then at time = GuessCutoff, the password was still alive (0).
  survival.data$surv.time   <- unlist(ifelse(survival.data$guess.number > 0,
                                             survival.data$guess.number,
                                             guesscutoff.column))
  survival.data$surv.status <- ifelse(survival.data$guess.number > 0,
                                      1,
                                      0)


  # To compute pairwise p-values using the pairwise.table built-in,
  #   we need a comparison function that indexes levels by integer
  conditionnames <- keys(guesscutoffs)
  survcmp <- function(i, j) {
    # Extract subset of survival.data that only includes the two specified conditions, as indexed by integer
    subset.df <- subset(survival.data,
                        condition == conditionnames[i] | condition == conditionnames[j])
    result <- survdiff(Surv(surv.time, surv.status) ~ condition,
                       data = subset.df,
                       rho = 1)
    cat("Cracked ")
    cat(conditionnames[i])
    cat("\n")
    print(summary(factor(subset.df[subset.df$condition == conditionnames[i], ]$surv.status)))
    cat("\n")

    cat("Cracked ")
    cat(conditionnames[j])
    cat("\n")
    print(summary(factor(subset.df[subset.df$condition == conditionnames[j], ]$surv.status)))
    cat("\n")

    cat("Chi-squared for ")
    cat(conditionnames[i])
    cat(" vs ")
    cat(conditionnames[j])
    cat(": ")
    cat(result$chisq)
    cat("\n")
    # Extract chi-square statistics from survdiff result and compute p-value
    return(1 - pchisq(result$chisq, 1))
  }
  cat("Computing log-rank for conditions:", conditionnames, "\n")
  print(pairwise.table(survcmp, conditionnames, p.adjust.method = "holm"))
}


OutputGuessedAndUnguessed <- function(lookup.results) {
  # This function outputs guessed and unguessed passwords for qualitative analysis
  # Only some of the columns are output.  User IDs are redacted.
  #
  # Input: lookup.results
  #     data frame where each row corresponds to an independent data point
  #
  # Output: tables to stdout
  #
  cat("Guessed passwords:\n")
  guessed.data <- subset(lookup.results, guess.number > 0)
  ordered.data <- arrange(guessed.data, condition, guess.number)
  ordered.data$probability <- sprintf("%e", ordered.data$probability)
  subset.data <- ordered.data[,c("condition","password","guess.number","probability","source.ids")]
  write.table(subset.data,
              file = stdout(),
              quote = F,
              sep = "\t",
              row.names = F,
              col.names = T,
              fileEncoding = "utf-8")

  cat("\n\nUnguessed passwords:\n")
  unguessed.data <- subset(lookup.results, guess.number < 0)
  ordered.data <- arrange(unguessed.data, condition, desc(guess.number), desc(probability))
  ordered.data$probability <- sprintf("%e", ordered.data$probability)
  subset.data <- ordered.data[,c("condition","password","guess.number","probability","pattern","source.ids")]
  write.table(subset.data,
              file = stdout(),
              quote = F,
              sep = "\t",
              row.names = F,
              col.names = T,
              fileEncoding = "utf-8")
}


ReadSingleResultsPair <- function(filenamestub) {
  # This function takes a "filenamestub" and reads in:
  #     lookupresults.filenamestub
  #     totalcounts.filenamestub
  # The guess calculator framework scripts are assumed to return files of this
  #   type, where the pair of files describes that lookup results and guess
  #   cutoff for a particular experiment.
  #
  # Stop on any error.
  #
  # Returns a list containing two items:
  # lookup.results
  #     a data frame with the data from lookupresults.filenamestub
  #
  # guesscutoff
  #     a hash containing data from totalcounts.filenamestub
  #     it will have one entry for each unique condition in lookup.results
  #     mapped to the value from the totalcounts file
  #
  lookup.name <- paste("lookupresults.", filenamestub, sep = "")
  guesscutoff.name <- paste("totalcounts.", filenamestub, sep = "")
  for (name in c(lookup.name, guesscutoff.name)) {
    if (!file.exists(name)) {
      stop("Could not find file ", name, "!")
    }
  }

  # Read lookupresults.filenamestub
  lookup.results <- read.table(file = lookup.name,
                               header = F,
                               sep = "\t",
                               quote = "",
                               comment.char = "")
  # lookupresults files don't have a header row, so set column names here
  colnames(lookup.results)[1:7] <- c("workerid","condition","brokenpassword",
                                     "probability","pattern","guess.number","source.ids")
  # Make sure probability column is numeric
  lookup.results$probability <- as.numeric(lookup.results$probability)
  # Create a password column for future use
  lookup.results$password <- gsub("\001", "",
                                  lookup.results$brokenpassword,
                                  fixed = T)

  # Read totalcounts.filenamestub and map the count to all conditions
  guesscutoff.data <- read.table(file = guesscutoff.name,
                                 header = F,
                                 sep = "\t",
                                 quote = "",
                                 comment.char = "")
  # Total count is the second field on the first line
  guesscutoffs <- hash()
  for (condition in unique(lookup.results$condition)) {
	  index = which(gsub(":Total count", "", guesscutoff.data[,1], fixed=T)==condition)
	  if (length(index) != 1) {
		  stop(paste("Missing from totalcounts:", condition))
	  }
      guesscutoffs[[condition]] <- guesscutoff.data[index,2]
  }

  if (!ValidGuessCutoffs(lookup.results, guesscutoffs)) {
    stop("Error reading results from files in ReadSingleResultsPair! See warnings for more information.")
  }
  return(list(lookup.results = lookup.results,
              guesscutoffs = guesscutoffs))
}


CombineResults <- function(list.of.results.1, list.of.results.2) {
  # This function takes in two lists of the form returned by
  #   ReadSingleResultsPair, and combines them into a single list.
  # The lookup.results data frames from the lists are simply rbinded.  The
  #   guesscutoffs are combined such that if two conditions have the same name,
  #   the minimum of the two cutoff values is retained.
  #
  # Returns a list containing two items, in the same format as
  #   ReadSingleResultsPair.
  #
  lookup.results <- rbind(list.of.results.1$lookup.results,
                          list.of.results.2$lookup.results)
  # Make a deep copy of the first hash
  guesscutoffs <- copy(list.of.results.1$guesscutoffs)
  for (key in keys(list.of.results.2$guesscutoffs)) {
    val1 <- guesscutoffs[[key]]
    val2 <- list.of.results.2$guesscutoffs[[key]]
    if (has.key(key, guesscutoffs)) {
      guesscutoffs[[key]] <- min(val1, val2)
    } else {
      guesscutoffs[[key]] <- val2
    }
  }
  if (!ValidGuessCutoffs(lookup.results, guesscutoffs)) {
    stop("Error combining results in CombineResults! See warnings for more information.")
  }
  return(list(lookup.results = lookup.results,
              guesscutoffs = guesscutoffs))
}


ReadMultipleResultsPairs <- function(filenamestubs) {
  # This function takes in a vector of filenamestubs, reads in files using
  #   ReadSingleResultsPair, and combines them using CombineResults.
  #
  # Input:
  # filenamestubs
  #     a character vector of filename stubs
  #     files of the form lookupresults.filenamestub and totalcounts.filenamestub
  #     should exist in the current directory (this will be checked)
  #
  # Returns a list containing two items, in the same format as
  #   ReadSingleResultsPair.
  #
  curlist <- ReadSingleResultsPair(filenamestubs[1])
  if (length(filenamestubs) > 1) {
    for (i in 2:length(filenamestubs)) {
      curlist <- CombineResults(curlist,
                                ReadSingleResultsPair(filenamestubs[i]))
    }
  }
  return(curlist)
}


ReadFilesFromDirectoryPath <- function(dir.path) {
  # This simple function will search for and read all results files found in a
  #   directory path.
  files <- list.files(dir.path)
  lookup.results.files <- grep("^lookupresults\\.", files, perl = T, value = T)
  stubs <- sub("lookupresults.", "", lookup.results.files, fixed = T)

  # If the same basename is used with multiple cutoffs, take the one with the
  #   smallest cutoff.  This requires string manipulating the stubs to extract
  #   basename and cutoff, sorting by basename then cutoff, and removing
  #   duplicates.
  basenames <- sub("\\d+(\\.\\d+)?(e-\\d+)?$", "", stubs, perl = T)
  cutoff.strings <- str_sub(stubs, start = (nchar(basenames) + 1))
  cutoffs <- as.numeric(cutoff.strings)
  stub.strings <- data.frame(basenames, cutoffs, stubs)

  # Sort by base name then cutoff
  sorted.stubs <- arrange(stub.strings, basenames, cutoffs)

  # Filter out duplicates
  filtered.stubs <- subset(sorted.stubs, !duplicated(basenames))

  return(ReadMultipleResultsPairs(filtered.stubs$stubs))
}



MakePlotFromCurrentDirectory <-
  function(truncate.to.min.cutoff = F,
           pdf.filename = NA, ...) {
  # This is a simple wrapper function used to create a single guessing curves
  #   plot from all of the lookupresults files in the current directory.
  #
  # Inputs:
  # truncate.to.min.cutoff
  #     if TRUE, chop the graph at the minimum guesscutoff found across all
  #     results
  #
  # pdf.filename
  #     if specified, will save the plot to a pdf in the current directory
  #
  # Outputs:
  #     plot to current device or pdf
  #
  results.list <- ReadFilesFromDirectoryPath(".")
  lookup.results <- results.list$lookup.results
  # Make pdf current device if pdf.filename is specified
  if (!is.na(pdf.filename)) {
    pdf(pdf.filename, width = 8, height = 5)
  }
  if (truncate.to.min.cutoff) {
    guesscutoff <- min(unlist(values(results.list$guesscutoffs)))
    PlotGuessingCurves(lookup.results = lookup.results,
                       guesscutoff = guesscutoff,
                       graph.equalizecutoff = T,
                       ...)
  } else {
    guesscutoff <- max(unlist(values(results.list$guesscutoffs)))
    PlotGuessingCurves(lookup.results = lookup.results,
                       guesscutoff = guesscutoff,
					   guesscutoffs = results.list$guesscutoffs,
                       ...)
  }
  if (!is.na(pdf.filename)) {
    # Close pdf
    dev.off()
  }
}


StatsFromCurrentDirectory <- function() {
    results.list <- ReadFilesFromDirectoryPath(".")
    lookup.results <- results.list$lookup.results
    guesscutoffs <- results.list$guesscutoffs
    LogRanks(lookup.results, guesscutoffs)
}


################################################################################
# Main block

# If this script is run explicitly from the command-line with Rscript,
#   accept certain commands.
cargs <- commandArgs(trailingOnly = T)
if (length(cargs) > 1) {
  if (cargs[1] == "makeplot") {
      config <- data.frame()
      if(length(cargs) >= 3){
        configfile <- file(cargs[3])
        config <- read.csv(configfile, header=T)
      }

      if(length(cargs) >= 4){
        source(cargs[4])
      }
      MakePlotFromCurrentDirectory(pdf.filename = cargs[2],
                                   ## graph.equalizecutoff = T,
                                   continuecurvestocutoff = T,
                                   truncate.to.min.cutoff = T,
                                   config=config,
                                   ## Billy: added for paper
                                   logxbreaks = 10^seq(1, 25, 2),
                                   xlimits = c(1, 10^25)
                                   )

  }
}
if (length(cargs) >= 1) {
  if (cargs[1] == "makestat") {
      cat("Making stats...\n")
      StatsFromCurrentDirectory()
  }
}
