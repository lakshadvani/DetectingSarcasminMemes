setwd("/Users/lakshadvani/Desktop/ml_project")
library(BBmisc)

df <- read.csv("sampledata1.csv")

#remove indicators of obvious incorrect answers (does not account for potential misconceptions)
df = df[df$untouched=="false",]
df = df[df$incomplete=="false",]

#discretize difficulty measure
# diff <.3 -> 0
# .3 < diff < .7
# .7 < diff
df[df$diff<=.3,"diff"]=0
df[(df$diff<=.7&df$diff>.3),"diff"]=.5
df[df$diff>.7,"diff"]=1

list = c("correct","level_summary.mastery.mean" ,"level_summary.mastery.std_dev",'diff',"time_spent")                         
keep = c(list,"student.school_id",'level_summary.subject',"student.grade")
df<-df[ , (names(df) %in% keep)]

#get rid of incomplete data
df <- df[complete.cases(df),]

list
df = df[df$student.grade!="7,8",]
df = df[df$student.grade!="",]
df$student.grade = strtoi(df$student.grade)

for (element in list){
  data = df[,element]
  if(length(unique(data))==2){
    if ((sum(unique(data)==c("true","false"))==2)|(sum(unique(data)==c("false","true"))==2)){
      #index = data == "true"
      df[,element] <- as.integer(as.logical(data)) 
    }
  }else{
    df[,element]<- as.numeric(df[,element])
  }
}
df$time_spent = df$time_spent/max(df$time_spent)
#df$level_summary.t_elapsed = df$level_summary.t_elapsed/max(df$level_summary.t_elapsed)
head(df)

write.csv(df,"sampledata_preprocessed.csv")
##Aggregate data per school##

schools = unique(df$student.school_id)
grades = unique(df$student.grade)
subjects = unique(df$level_summary.subject)

colnames = c(as.vector(outer(subjects,list, paste, sep=".")),"grade_level",paste0(subjects,".num_students"))

school_df <- data.frame(matrix(data = NA, nrow = length(schools), ncol = length(colnames)),row.names=schools)
colnames(school_df)<- colnames

for(school in schools){
  subset = df[df$student.school_id==school,]
  
  ##School type: elementary (0), middle school (1), high school (2)
  grade_level = mean(subset$student.grade)
  if (grade_level<6){
    school_df[school,"grade_level"] = 0
  }else if(grade_level < 9){
    school_df[school,"grade_level"] = 1
  }else{
    school_df[school,"grade_level"] = 2
  }
  
  
  for(subject in subjects){
    further_subset = subset[subset$level_summary.subject==subject,]
    n = dim(further_subset)[1]
    school_df[school,paste(subject,"num_students",sep=".")] = n
    for(element in list){
      if(n>0){
        school_df[school,paste(subject,element,sep=".")] = sum(further_subset[,element])/n
      }else{
        school_df[school,paste(subject,element,sep=".")] = 0
      }
    }
  }
  
}

school_df$grade_level <- as.factor(school_df$grade_level)

library(clustMixType)
clusters = 5
model = kproto(school_df,k=clusters)
summary(model)
sprintf("prot schools", summary(model))

model$cluster
model$centers

par(mfrow=c(clusters,1))
index = model$cluster==1
temp = school_df[index,]
hist(temp$fractions.correct,ylim=c(0,4))

for(i in 1:clusters){
  index = model$cluster==i
  temp = school_df[index,]
  hist(temp$fractions.correct,add=T)
}

##WORK IN PROGRESS FROM HERE DOWN
set.seed(42)
p1 <- hist(rnorm(500,4))                     # centered at 4
p2 <- hist(rnorm(500,6))                     # centered at 6
plot( p1, col=rgb(0,0,1,1/4), xlim=c(0,10))  # first histogram108/180
plot( p2, col=rgb(1,0,0,1/4), xlim=c(0,10), add=T) 


#hist(sort(df$time_spent)[1:52000],main = "Time Spent",xlab="seconds",ylab="frequency",breaks=100)
#hist(log(df$time_spent),main = "Log Time Spent",xlab="seconds",ylab="frequency",breaks=50)
##Plot Difficulty Against Percentage Correct
ID = unique(df$qual_id)

difficulty = rep(0,length(ID))
accuracy = rep(0,length(ID))

i=1
for (element in ID){
  subset = df[df$qual_id == element,]
  accuracy[i] <- sum(subset$correct==1)/dim(subset)[1]
  difficulty[i] <- subset[1,"level_summary.mastery.mean"]
  i <- i+1
}
plot(accuracy~difficulty)  
  

