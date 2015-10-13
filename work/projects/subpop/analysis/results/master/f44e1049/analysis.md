
```r
library(ggplot2)
```

```
## Warning: package 'ggplot2' was built under R version 2.15.1
```

```r
library(plyr)
library(caret)
```

```
## Warning: package 'caret' was built under R version 2.15.1
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 2.15.1
```

```
## Loading required package: reshape
```

```
## Attaching package: 'reshape'
```

```
## The following object(s) are masked from 'package:plyr':
## 
## rename, round_any
```

```
## Loading required package: cluster
```

```
## Warning: package 'cluster' was built under R version 2.15.1
```

```
## Loading required package: foreach
```

```
## Warning: package 'foreach' was built under R version 2.15.1
```

```r
library(doMC)
```

```
## Loading required package: iterators
```

```
## Loading required package: multicore
```

```
## Attaching package: 'multicore'
```

```
## The following object(s) are masked from 'package:lattice':
## 
## parallel
```

```r
library(xtable)
library(reshape2)
```

```
## Attaching package: 'reshape2'
```

```
## The following object(s) are masked from 'package:reshape':
## 
## colsplit, melt, recast
```

```r
library(MASS)
```

```
## Warning: package 'MASS' was built under R version 2.15.1
```

```r
library(grid)
library(gridExtra)
```

```
## Warning: package 'gridExtra' was built under R version 2.15.1
```

```r
library(hash)
```

```
## Warning: package 'hash' was built under R version 2.15.1
```

```
## hash-2.2.4 provided by Decision Patterns
```

```r
library(devtools)
```

```
## Warning: package 'devtools' was built under R version 2.15.1
```

```r
library(knitr)
library(glmnet)
```

```
## Warning: package 'glmnet' was built under R version 2.15.1
```

```
## Loading required package: Matrix
```

```
## Warning: package 'Matrix' was built under R version 2.15.1
```

```
## Attaching package: 'Matrix'
```

```
## The following object(s) are masked from 'package:reshape':
## 
## expand
```

```
## The following object(s) are masked from 'package:stats':
## 
## toeplitz
```

```
## Loaded glmnet 1.8-2
```

```r
load_all("~/work/code/Rpackages/cytominr")
```

```
## Loading cytominr
```

```
## Loading required namespace: testthat
```

```
## Loading required namespace: yaml
```

```
## Warning: package 'testthat' was built under R version 2.15.1
```

```
## Loading required package: lpSolve
```

```
## Warning: package 'stringr' was built under R version 2.15.1
```

```r
rm(list = ls())
D <- read.table("../input/HANDTRAINED-B1/make-input-to-random-forest-training.txt")
X <- D[, 2:NCOL(D)]
y <- D[, 1]
```




```r
tbl <- table(y)
cnt <- data.frame(Class = names(tbl), Count = as.vector(tbl))
cnt <- cnt[with(cnt, order(Count)), ]
cnt$Class <- factor(cnt$Class, levels = cnt$Class)
p <- qplot(Class, Count, data = cnt) + coord_flip()
ggsave("counts.pdf", p, width = 6, height = 6)
```



```r
Xtr <- X
ytr <- y

# pvec <- y != 'interphase' Xtr <- X[pvec,] ytr <- droplevels(y[pvec])
```



```r
# classifier_type = 'knn' classifier_type = 'lda' classifier_type = 'rf'
# classifier_type = 'nb'

registerDoMC()

# registerDoSEQ() fit_ <- train(matrix(rnorm(1000), nrow=100, ncol=10),
# rnorm(100), method=method)


nfolds <- 5
nrepeats <- 3
fitControl <- trainControl(method = "repeatedCV", number = nfolds, repeats = nrepeats, 
    returnResamp = "all", selectionFunction = "oneSE")

for (classifier_type in c("knn", "lda", "nb", "rf")) {
    fit <- train(Xtr, ytr, method = classifier_type, trControl = fitControl)
    
    cfmat <- melt(confusionMatrix(fit, norm = "none")$table)
    cfmat$Prediction <- factor(cfmat$Prediction, levels = levels(y))
    cfmat$Reference <- factor(cfmat$Reference, levels = levels(y))
    p <- plot.confmat(cfmat, rotate_labels = T)
    p <- p + labs(title = sprintf("Classifier = %s.", classifier_type))
    p <- p + theme(axis.text.y = element_text(hjust = 0)) + theme(axis.text.x = element_text(hjust = 0))
    fname <- sprintf("confmat_%s.pdf", classifier_type)
    ggsave(fname, p, height = 8, width = 8)
    
    print(xtable(fit$results), file = sprintf("results_%s.tex", classifier_type), 
        size = "tiny")
    
}
```

```
## Warning: package 'e1071' was built under R version 2.15.1
```

```
## Loading required package: class
```

```
## Warning: package 'class' was built under R version 2.15.1
```

```
## Attaching package: 'class'
```

```
## The following object(s) are masked from 'package:reshape':
## 
## condense
```

```
## Warning: variables are collinear
```

```
## Warning: package 'klaR' was built under R version 2.15.1
```

```
## Warning: package 'randomForest' was built under R version 2.15.1
```

```
## randomForest 4.6-7
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r

```

