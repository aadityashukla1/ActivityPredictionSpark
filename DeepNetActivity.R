# install packages
install.packages("splitstackshape")

#load the library
library(splitstackshape)

#load the dataset into R 
df <- read.csv("WISDM_at_v2.0_raw.txt", header = FALSE, stringsAsFactors = FALSE)

# view the dataset
head(df,10)

#remove the char ';' from last column
df.Sep <- concat.split(data = df,split.col = "V6", sep = ";")

#drop column V6 from df.Sep
df <- subset(df.Sep, select = -c(V6))

rm(df.Sep)

#now rename the dataframe columns
colnames(df) <- c("user", "activity", "timestamp", "xaccel", "yaccel", "zaccel")

df <- subset(df, select = -c(timestamp,user))
table(df$activity)

toBeRemoved<-which(df$activity=="")
df<-df[-toBeRemoved,]
df$activity <- factor(df$activity)

# train the model
#h2o.init()

#h2o.shutdown()
h2o.init(nthreads = -1)

df.hex <- as.h2o(df)

splits <- h2o.splitFrame(df.hex, c(0.6,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%


response <- "activity"
predictors <- setdiff(names(df.hex), response)
predictors


modelActivity1 <- h2o.deeplearning(
    model_id="dl_model_Activity", 
    training_frame=train, 
    validation_frame=valid,
    x=predictors,
    y=response,
    hidden=c(200,200,200),                  ## small network, runs faster
    epochs=100,                      ## hopefully converges earlier...
    stopping_metric="misclassification", ## could be "MSE","logloss","r2"
    stopping_tolerance=0.0000001
)
summary(modelActivity1)


modelActivity2 <- h2o.deeplearning(
    model_id="dl_model_Activity", 
    training_frame=train, 
    validation_frame=valid,
    x=predictors,
    y=response,
    hidden=c(100,100),
    epochs=1000000,
    stopping_metric = "misclassification",
    score_validation_samples=10000,
    stopping_rounds=20,
    stopping_tolerance=0.0000001
)
summary(modelActivity2)