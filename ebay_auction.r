library(ggplot2)
library(randomForest)
library(dplyr)
library(caret)
library(doSNOW)
library(e1071)

# loading raw data
auction<-read.csv("ebay_auction.csv")

# checking/fixing variable types
str(auction)
auction$auctionid<-as.character(auction$auctionid)

# distribution across auction items
table(auction$item)
# distribution across auction type
table(auction$auction_type)


# creating derived features: 
## winning bid, number of bids & average bid per auction
u<-unique(auction$auctionid)
winner<-rep(0,nrow(auction))

average<-rep(0,nrow(auction))
num.bids<-rep(0,nrow(auction))

for (i in u){
        indexes<-which(auction$auctionid == i)
        cur.average<-sum(auction$bid[indexes])/length(indexes)
        
        for (k in indexes){
                winner[k]<-ifelse(auction$bid[k]==auction$price[k], 1 , 0)
                num.bids[k]<-length(indexes)
                average[k]<-cur.average
        }
}
auction$winner<-winner
auction$ave.bid<-average
auction$num.bids<-num.bids

## length of auction (number of days)
auction$length<-rep(0,nrow(auction))
indexes3<-which(auction$auction_type == "3 day auction")
auction$length[indexes3]<-3
indexes5<-which(auction$auction_type == "5 day auction")
auction$length[indexes5]<-5
indexes7<-which(auction$auction_type == "7 day auction")
auction$length[indexes7]<-7

## normalized bidding time (when bid was placed over auction length)
auction$time.fraction<-auction$bidtime/auction$length

## bid distance from open bid
auction<-mutate(auction,dist.open=(bid-openbid)/bid)
## bid distance from average bid
auction<-mutate(auction,dist.ave=(bid-ave.bid)/ave.bid)

#####

auction$winner<-as.factor(auction$winner)
levels(auction$winner)<-c("other bids","winning bid")
levels(auction$item)<-c("Cartier watch","Palm Pilot","Xbox")

######## PLOTS
# distribution of bids over item and aution type
ggplot(auction,aes(x =item, fill= winner))+
        geom_bar()+
        facet_wrap(~auction_type)+
        xlab("Auction Item")+ ylab("Total Count")

# check if no. of bids has predictive power
# hypothesis - As number of bids per item increases chance of winning drops

ggplot(auction,aes(x = num.bids, fill= winner))+
        geom_histogram(binwidth = 20)+
        facet_wrap(~auction_type)+
        xlab("Number of Bids per Auction")+ ylab("Total Count")

# check if distance from open bid has predictive power
# hypothesis - Bids much higher than openning bid have higher chance of winning
ggplot(auction)+
        geom_point(aes(x = dist.open,y=price, color= winner))+
        facet_wrap(~auction_type)+
        xlab("Distance form Openning Bid")+ ylab("Final Sale Price")


# check if distance from average bid has predictive power
# hypothesis - Bids below the average bid have zero chance of winning

ggplot(auction)+
        geom_point(aes(x =dist.ave,y=price, color= winner))+
        facet_wrap(~auction_type)+
        xlab("Distance form Average Bid")+ ylab("Final Sale Price")


# check if time fraction has predictive power
# hypothesis - Bids placed toward the end of the auction have higher chance of winning

ggplot(auction)+
        geom_point(aes(x =time.fraction,y=price, color= winner))+
        facet_wrap(~auction_type)+
        xlab("Bidding Time")+ ylab("Final Sale Price")


######## Machine learning algortihm to predict winning bids

# Train a random forests with dist. from open bid,dist. form average bid and no. of bids
rf.train1<-auction[,c("dist.open","dist.ave","num.bids")]

set.seed(123)
rf1<-randomForest(x=rf.train1,y=auction$winner, importance = TRUE,ntree = 1000)
rf1
varImpPlot(rf1)

# Train a random forests with dist. from open bid, 
# dist. form average bid, bidding time, and no. of bids
rf.train2<-auction[,c("dist.open","dist.ave","num.bids","time.fraction")]

set.seed(123)
rf2<-randomForest(x=rf.train2,y=auction$winner, importance = TRUE,ntree = 1000)
rf2
varImpPlot(rf2)

# Train a random forests with dist. from open bid, 
# dist. form average bid, bidding time,auction type and no. of bids
rf.train3<-auction[,c("dist.open","dist.ave","num.bids","time.fraction","auction_type")]

set.seed(123)
rf3<-randomForest(x=rf.train3,y=auction$winner, importance = TRUE,ntree = 1000)
rf3
varImpPlot(rf3)

# Train a random forests with dist. from open bid, 
# dist. form average bid, bidding time, item, and no. of bids
rf.train4<-auction[,c("dist.open","dist.ave","num.bids","time.fraction","item")]

set.seed(123)
rf4<-randomForest(x=rf.train4,y=auction$winner, importance = TRUE,ntree = 1000)
rf4
varImpPlot(rf4)


# overfitting? optimistic error rate?
## is it too dependent on that data? likely to have a higher error rate on new unseen data?


#### Cross Validation

#to estimate how accurately the predictive model will perform in practice. 
rf.label<-as.factor(auction$winner)

# 10 fold cross validation repeated 10 times (100 distinct slices of data used over and over again to train)
set.seed(123)
cv.10.folds<-createMultiFolds(rf.label,k=10,times = 10)

# Check stratification
table(rf.label)
table(rf.label[cv.10.folds[[33]]])

# Set up caret's trainControl object per above.
ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv.10.folds)


# Set up doSNOW package for multi-core training. This is helpful as we're going
# to be training a lot of trees.
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)


set.seed(123)
rf2.cv1 <- train(x = rf.train2, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 500, trControl = ctrl.1)
stopCluster(cl)

#check out the results
rf2.cv1
