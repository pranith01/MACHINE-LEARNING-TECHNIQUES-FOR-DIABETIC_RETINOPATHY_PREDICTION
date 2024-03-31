

library(foreign)

loadData = function(path){
  
  df = read.arff(path)
  colnames(df) <- c(
    "q",      #  0 The binary result of quality assessment. 0 = bad quality 1 = sufficient quality
    "ps",     #  1 The binary result of pre-screening, 1 indicates severe retinal abnormality and 0 its lack
    "nma.a",  #  2 Number of MAs found at the confidence levels alpha = 0.5
    "nma.b",  #  3 Number of MAs found at the confidence levels alpha = 0.6
    "nma.c",  #  4 Number of MAs found at the confidence levels alpha = 0.7
    "nma.d",  #  5 Number of MAs found at the confidence levels alpha = 0.8
    "nma.e",  #  6 Number of MAs found at the confidence levels alpha = 0.9
    "nma.f",  #  7 Number of MAs found at the confidence levels alpha = 1.0
    "nex.a",  #  8 Number of Exudates found at the confidence levels alpha = 0.5
    "nex.b",  #  9 Number of Exudates found at the confidence levels alpha = 0.6
    "nex.c",  # 10 Number of Exudates found at the confidence levels alpha = 0.7
    "nex.d",  # 11 Number of Exudates found at the confidence levels alpha = 0.8
    "nex.e",  # 12 Number of Exudates found at the confidence levels alpha = 0.9
    "nex.g",  # 13 Number of Exudates found at the confidence levels alpha = 1.0
    "nex.f",  # 14 Number of Exudates found at the confidence levels alpha = 1.0
    "nex.h",  # 15 Number of Exudates found at the confidence levels alpha = 1.0
    "dd",     # 16 The euclidean distance of the center of the macula and the center of the optic disc
    "dm",     # 17 The diameter of the optic disc
    "amfm",   # 18 The binary result of the AM/FM-based classification
    "class"   # 19 Class label. 1 = contains signs of DR, 0 = no signs of DR
  )
  odf = df
  
  colnames(df)
  numericFeats = c(3:16)
  eyeFeats = c(17,18)
  df[, c(numericFeats, eyeFeats)] = scale(df[, c(numericFeats, eyeFeats)])
  
  df$class = as.factor(df$class)
  return(df)
}

df = loadData("D:/Freelance/Ninza_Solutions/ML_R_3/P/messidor_features.arff")

long = melt(df[,c(1:ncol(df)-1)])

# Load the reshape2 package
library(reshape2)

# Melt the data frame
long <- melt(df[, c(1:ncol(df)-1)])

library(ggplot2)

ggplot(long) + 
  geom_boxplot(aes(variable, value)) + 
  coord_flip() +
  labs(title="Unimodal feature distribution", x='Feature', y='Scaled value')

library(ggcorrplot)

# Select only numeric columns from the dataframe
numeric_df <- df[sapply(df, is.numeric)]

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_df)

# Plot the correlation matrix
ggcorrplot(correlation_matrix, title = "Feature correlation matrix")

ggplot(df) + geom_point(aes(nma.a, nma.b, color=class)) + facet_wrap(~amfm)

ggplot(df) + geom_point(aes(nma.a, nma.f, color=class)) + facet_wrap(~amfm)


ggplot(df) + geom_point(aes(nex.a, nex.b, color=class)) + facet_wrap(~amfm)

ggplot(df) + geom_point(aes(nex.a, nex.h, color=class)) + facet_wrap(~amfm)

ggplot(df) + geom_point(aes(nma.a, nex.h, color=class)) + facet_wrap(~amfm)

install.packages("kohonen")
library(kohonen)

som_grid <- somgrid(xdim = 15, ydim = 15, topo = "hexagonal")

cols = colnames(df)[1:(ncol(df)-1)]
base = df[,cols]
som_grid = somgrid(xdim=15, ydim=15, topo="hexagonal")
som_model = som(as.matrix(base), grid=som_grid, rlen=200)

plot(som_model, type="property", property = df$nma.a)
title("SOM colored for nma.a feature")

plot(som_model, type="property", property = df$nex.a)
title("SOM colored for nex.a feature")

plot(som_model, type="property", property = df$nex.h)
title("SOM colored for nex.h feature")

helpers.result = function(pred, real){
  return(data.frame(pred=pred, real=real, ok=(pred==real)))
}

fitpredict.svm = function(formula, trainingSet, validationSet, kernel="polynomial", degree=2, coef0=1){
  model = svm(formula, trainingSet, kernel=kernel, degree=degree, coef0=coef0)
  pred = as.integer(predict(model, validationSet))
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

fitpredict.forest = function(formula, trainingSet, validationSet, ntree=100){
  model = randomForest(formula, trainingSet, ntree=ntree)
  pred = as.integer(predict(model, validationSet))
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

fitpredict.knn = function(formula, trainingSet, validationSet, k=5){
  cols = all.vars(formula)[2:length(all.vars(formula))]
  pred = as.integer(knn(trainingSet[,cols], validationSet[,cols], trainingSet$class, k=k))
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

fitpredict.adaboost = function(formula, trainingSet, validationSet, boos=TRUE, mfinal=10, coeflearn='Breiman'){
  model = boosting(formula, trainingSet, boos=boos, mfinal=mfinal, coeflearn=coeflearn)
  pred = as.integer(predict.boosting(model, validationSet)$class)+1
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

fitpredict.nnet = function(formula, trainingSet, validationSet, size=10){
  model = nnet(formula, trainingSet, size=size, trace=FALSE)
  pred = as.integer(round(predict(model, validationSet)))+1
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

fitpredict.naiveBayes = function(formula, trainingSet, validationSet){
  model = naiveBayes(formula, trainingSet)
  pred = as.integer(predict(model, validationSet))
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

fitpredict.lda = function(formula, trainingSet, validationSet){
  model = lda(formula, trainingSet)
  pred = as.integer(data.frame(predict(model, validationSet))$class)
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(pred, real))
}

kfoldValidate = function(formula, data, learner, performance, ...){
  
  k=10
  ks = c(1:k)
  folds = createFolds(data$class, k, list=FALSE)
  result = rep(0, k)
  
  for (i in ks){
    
    trainingKs = ks[ks!=i]
    validationKs = ks[ks==i]
    trainingSet = data[which(folds %in% trainingKs), ]
    validationSet = data[which(folds %in% validationKs), ]
    result[i] = performance(learner(formula, trainingSet, validationSet, ...))
  }
  
  return(list(mean=mean(result), sd=sd(result), results=c(result)))
}



library(caret)


cols = all.vars(f1)[2:length(all.vars(f1))]
fit = prcomp(df[,cols], center=T, scale=T)

df = cbind(df, fit$x)

ind <- sample(2, nrow(df), replace=TRUE, prob=c(0.5, 0.5))

trainingDf   = df[ind==1,]
validationDf = df[ind==2,]


selectModel = function(formula, data, learner, performance, hyperParameters){
  
  if(is.null(hyperParameters)){
    
    return(list(model=NULL, data=NULL))
  }
  
  results = cbind(hyperParameters, performance=rep(0, nrow(hyperParameters)))
  
  for (i in 1:nrow(results)){
    hyper = c(hyperParameters[i, ])
    arguments = c(list(formula=formula, data=data, learner=learner, performance=performance), hyper)
    results[i, 'performance'] = do.call(kfoldValidate, arguments)$mean
  }
  
  selectedModel = results[which.max(results$performance), colnames(hyperParameters)]
  names(selectedModel) = colnames(hyperParameters)
  
  return(list(model=as.list(selectedModel), data=results))
}

fitpredict.votingEnsemble = function(formula, trainingSet, validationSet, learners, params){
  
  ensembleResults = data.frame(majority=rep(NA, nrow(validationSet)))
  
  for (name in names(learners)){
    
    if(!is.null(params)){
      
      arguments = c(
        list(params[name][[1]]$formula, trainingSet=trainingSet, validationSet=validationSet),
        params[name][[1]]$params
      )
      
    } else {
      
      arguments = list(params[name][[1]]$formula, trainingSet=trainingSet, validationSet=validationSet)
    }
    
    modelResult = list(predicted=do.call(learners[name][[1]], arguments)$pred)
    names(modelResult) = c(name)
    ensembleResults = cbind(ensembleResults, modelResult)
  }
  
  ensembleResults$majority = apply(ensembleResults[, names(learners)], 1, FUN=function(x){ 
    as(names(which.max(table(x))), mode(x))  
  })
  
  real = as.integer(validationSet[,all.vars(formula)[1]])
  return(helpers.result(ensembleResults$majority, real))
}

selectVotingEnsemble = function(formula, data, performance, learners, learnersParams){
  
  learnerNames = names(learners)
  results = data.frame(left=c('full', learnerNames), result=rep(NA, length(learners)+1))
  
  bestRoundPerformance = kfoldValidate(formula, data, fitpredict.votingEnsemble, performance=perf.auc,
                                       learners=learners, 
                                       params=learnersParams
  )$mean    
  
  modelLeftOut = NULL
  
  results[results$left=='full', 'result'] = bestRoundPerformance
  
  for(i in 1:length(learnerNames)){
    
    try({
      
      roundPerformance = kfoldValidate(formula, data, fitpredict.votingEnsemble, perf.auc,
                                       learners=learners[-c(i)], 
                                       params=learnersParams[-c(i)]
      )$mean
      
      results[results$left==learnerNames[i], 'result'] = roundPerformance
      
      if(roundPerformance > bestRoundPerformance){
        
        bestRoundPerformance = roundPerformance
        modelLeftOut = i
      }
    })
  }
  
  if(!is.null(modelLeftOut) && length(learners) > 2){
    
    return(selectVotingEnsemble(formula, data, performance,
                                learners[-c(modelLeftOut)], 
                                learnersParams[-c(modelLeftOut)]
    ))
    
  } else {
    
    return(learners)
  }
}





