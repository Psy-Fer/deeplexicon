library(ROCit)

# Get sensitivity for given specificity
Sn4Sp <- function( Metrics , Specificity ) {
    for ( i in c(1:length(Metrics$SENS))) { 
        if (  Metrics$SPEC[i] < Specificity ) {
            print(paste( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 
            break    
}    }    }

# Get Youden index (J-statistic)
Jstat <- function( Metrics ){
    index <- 1 ; max <- -100 
    for ( i in c(1:length(Metrics$SENS))) {   
         tmp <- ( Metrics$SENS[i] + Metrics$SPEC[i] - 1 ) 
         if ( tmp > max ) { max <-  tmp; index=i }  
     } 
     print(paste0("J-stat: ", max))
     print(paste0("ACC: ", Metrics$ACC[index]))
     print(paste(index, Metrics$Cutoff[index], Metrics$SENS[index], Metrics$SPEC[index]))
}

##################################
#import testing data
t <- read.delim("test_predictions_filt.csv",header=F)
ta <- read.delim("test_actuals_filt.csv",header=F)

#calculate accuracy metrics
tm <- measureit(score = as.numeric(t$V2), class = as.numeric(ta$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))

for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( tm, s) }
#[1] "6764 0.99904484 0.168779219480487 0.999899997499938"
#[1] "20870 0.9714065 0.518737968449211 0.998999974999375"
#[1] "35156 0.68332464 0.84889622240556 0.98999974999375"

Jstat( tm )
#[1] "J-stat: 0.901222530563264"
#[1] "ACC: 0.954505112627816"
#[1] "0.3230364 0.942823570589265 0.958398959973999"

#PLOT ROC
t_rocit_emp <- rocit( score = as.numeric(t$V2), class = as.numeric(ta$V2) , method = "emp")
plot( t_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))


#Ploc accuracy and recovery = f(cutoff)
par(mar = c(5, 4, 4, 4) + 0.3)
plot( tm$Cutoff, tm$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
axis(2, las=1)
box()
par(new=TRUE)
lines( tm$Cutoff, tm$SENS, type="l", col=4) 
mtext("% Recovery", side=4, col=4, line=3)
axis(4, col=4, col.axis=4,las=1)
axis(1)

#Precision Recall
plot( tm$TP/(tm$TP+tm$FP), tm$TP/(tm$TP+tm$FN), type="l", ylim=c(0,1), xlim=c(0,1), col=3, ylab="Recall", xlab="Precision")



##################################
#import validation data

v <- read.delim("val_predictions_filt.csv",header=F)
va <- read.delim("val_actuals_filt.csv",header=F)

#calculate accuracy metrics
vm <- measureit(score = as.numeric(v$V2), class = as.numeric(va$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))

for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( vm, s ) }
#[1] "6242 0.9992685 0.155728893222331 0.999899997499938"
#[1] "20152 0.9754141 0.500787519687992 0.998999974999375"
#[1] "35301 0.67786294 0.852521313032826 0.98999974999375"
#[1] "44082 0.27509543 0.952048801220031 0.949998749968749"

Jstat(vm)
#[1] "J-stat: 0.903189246397827"
#[1] "ACC: 0.954692617315433"
#[1] "42881 0.31575218 0.945398634965874 0.957790611431952"

#PLOT ROC
v_rocit_emp <- rocit( score = as.numeric(v$V2), class = as.numeric(va$V2) , method = "emp")
plot( v_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))

#Ploc accuracy and recovery = f(cutoff)
par(mar = c(5, 4, 4, 4) + 0.3)
plot( vm$Cutoff, vm$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
axis(2, las=1)
box()
par(new=TRUE)
lines( vm$Cutoff, vm$SENS, type="l", col=4) 
mtext("% Recovery", side=4, col=4, line=3)
axis(4, col=4, col.axis=4,las=1)
axis(1)

#Precision Recall
plot( vm$TP/(vm$TP+vm$FP), vm$TP/(vm$TP+vm$FN), type="l", ylim=c(0,1), xlim=c(0,1), col=3, ylab="Recall", xlab="Precision")


##################################
#import independent replicate data

r <- read.delim("rep3_dmux_uniqMapped_trim_filt.tsv",header=F)
ra <- read.delim("rep3_guppy_mm2_MQ60_uniq_expected_filt_demuxed.tsv",header=F)

r2 <- read.delim("subsample_pr_m2.tsv", header=F)
ra2 <- read.delim("subsample_ex_m2.tsv",header=F)

#calculate accuracy metrics
rm <- measureit(score = as.numeric(r$V2), class = as.numeric(ra$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))
rm2 <- measureit(score = as.numeric(r2$V2), class = as.numeric(ra2$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))


for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( rm, s ) }
Jstat(rm)


for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( rm2, s ) }
# [1] "384 0.999844 0.00925 0.999891666666667 0.75223125"
# [1] "2331 0.997698 0.055225 0.998991666666667 0.76305"
# [1] "8736 0.96866 0.18835 0.989991666666667 0.78958125"
# [1] "31791 0.570983 0.494725 0.899991666666667 0.798675"

Jstat(rm2)
# [1] "J-stat: 0.441941666666667"
# [1] "ACC: 0.74498125"
# [1] "54640 0.21879 0.67295 0.768991666666667"

#PLOT ROC
r_rocit_emp <- rocit( score = as.numeric(r$V2), class = as.numeric(ra$V2) , method = "emp")
plot( r_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))

#Ploc accuracy and recovery = f(cutoff)
par(mar = c(5, 4, 4, 4) + 0.3)
plot( rm$Cutoff, rm$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
axis(2, las=1)
box()
par(new=TRUE)
lines( rm$Cutoff, rm$SENS, type="l", col=4) 
mtext("% Recovery", side=4, col=4, line=3)
axis(4, col=4, col.axis=4,las=1)
axis(1)



par(mar = c(5, 4, 4, 4) + 0.3)
plot( rm2$Cutoff, rm2$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
axis(2, las=1)
box()
par(new=TRUE)
lines( rm2$Cutoff, rm2$SENS, type="l", col=4) 
mtext("% Recovery", side=4, col=4, line=3)
axis(4, col=4, col.axis=4,las=1)
axis(1)


#Precision Recall
plot( rm$TP/(rm$TP+rm$FN), rm$TP/(rm$TP+rm$FP), type="l", ylim=c(0,1), xlim=c(0,1), col=3, xlab="Recall", ylab="Precision")


plot( rm2$TP/(rm2$TP+rm2$FN), rm2$TP/(rm2$TP+rm2$FP), type="l", ylim=c(0,1), xlim=c(0,1), col=3, xlab="Recall", ylab="Precision")

