# install packages
install.packages("splitstackshape")

#load the library
library(splitstackshape)

#load the dataset into R 
df <- read.csv("WISDM_at_v2.0_raw.txt", header = FALSE)

# view the dataset
head(df,10)

#remove the char ';' from last column
df.Sep <- concat.split(data = df,split.col = "V6", sep = ";")

#drop column V6 from df.Sep
df <- subset(df.Sep, select = -c(V6))

rm(df.Sep)

#now rename the dataframe columns
colnames(df) <- c("user", "activity", "timestamp", "xaccel", "yaccel", "zaccel")

df$timestamp <- as.numeric(df$timestamp)

#df <- subset(df, select = -c(timestamp))
table(df$activity)

toBeRemoved<-which(df$activity=="")
df<-df[-toBeRemoved,]
df$activity <- factor(df$activity)


# now clean the data for autoencoder model
mydat <- fread('https://s3.amazonaws.com/actidataset/WISDM_at_v2.0_unlabeled_raw.txt')

detach("package:h2o", unload=TRUE)
mydat$V3 <- as.numeric(mydat$V3)

mydat <- subset(mydat, select = -c(V2))

library(splitstackshape)

mydat <- concat.split(data = mydat,split.col = "V6", sep = ";")
mydat <- subset(mydat, select = -c(V6))
colnames(mydat) <- c("user", "timestamp", "xaccel", "yaccel", "zaccel")

# convert the dataframe to h2oframe
h2o.init(nthreads=-1)
mydat.hex <- as.h2o(mydat)

AutEnc <- h2o.deeplearning(
    model_id="dl_model_autoEncoder", 
    training_frame=mydat.hex, 
    x=predictors,
    hidden=c(200,200,200,200,200),
    epochs=500000,
    activation = "Tanh",
    autoencoder = TRUE,
    loss = "Automatic",
    stopping_metric = "AUTO"
)

h2o.saveModel(AutEnc, path = wd)

# train deeplearning model

df.hex <- as.h2o(df)

splits <- h2o.splitFrame(df.hex, c(0.6,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%


response <- "activity"
predictors <- setdiff(names(df.hex), response)
predictors

# not good model
modelActivity <- h2o.deeplearning(
    model_id="dl_model_Activity", 
    training_frame=train, 
    validation_frame=valid,
    x=predictors,
    y=response,
    hidden=c(200,200,200,200,200),
    epochs=10000,
    stopping_metric="misclassification",
    loss = "CrossEntropy",
    activation = "Tanh",
)
summary(modelActivity1)

# not good model
modelActivity2 <- h2o.deeplearning(
    model_id="dl_model_Activity", 
    training_frame=train, 
    validation_frame=valid,
    x=predictors,
    y=response,
    hidden=c(100,100),
    epochs=10,
    stopping_metric = "misclassification",
    activation = "TanhWithDropout",
    loss = "CrossEntropy"
)
summary(modelActivity2)

# good one
deepModelActi<- h2o.deeplearning(
    model_id="dl_model", 
    training_frame=train, 
    validation_frame=valid,
    x=predictors,
    y=response,
    hidden=c(200,200,200,200,200),
    activation = "Tanh",
    stopping_metric = "misclassification",
    loss = "CrossEntropy",
    stopping_rounds = 10,
    epochs = 500000,
    score_interval = 14,
    pretrained_autoencoder ="dl_model_autoEncoder"
)

# good two
> deepModelActi1<- h2o.deeplearning(
          model_id="dl_model1", 
          training_frame=train, 
          validation_frame=valid,
          x=predictors,
          y=response,
          hidden=c(200,200,200,200,200),
          activation = "Tanh",
          adaptive_rate = FALSE,
          nesterov_accelerated_gradient = TRUE,
          rate = 0.005,
          stopping_metric = "misclassification",
          loss = "CrossEntropy",
          epochs = 500000,
          pretrained_autoencoder ="dl_model_autoEncoder"
      )
