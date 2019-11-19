library(ROCit)

# Get sensitivity for given specificity
Sn4Sp <- function( Metrics , Specificity ) {
	for ( i in c(1:length(Metrics$SENS))) { 
		if (  Metrics$SPEC[i] < Specificity ) {
			print(paste( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 
			break	
}	}	}

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

Maxcuracy <- function( Metrics ){
	index <- 1 ; max <- -100 
	for ( i in c(1:length(Metrics$ACC))) {   
 		tmp <- ( Metrics$ACC[i]) 
 		if ( tmp > max ) { max <-  tmp; index=i }  
 	} 
 	print(paste0("Macc Accuracy: ", max))
 	print(paste(index, Metrics$Cutoff[index], Metrics$SENS[index], Metrics$SPEC[index]))
}

CutoffStats <- function( Metrics, Thresh ){ 
	for ( i in c(1:length(Metrics$Cutoff))) {
		if ( Metrics$Cutoff[i] < Thresh ) {
	    	print(paste("Index","Cutoff","Sens","Spec","Acc"))
	    	print(paste(i-1, Metrics$Cutoff[i-1], Metrics$SENS[i-1], Metrics$SPEC[i-1], Metrics$ACC[i-1]))
	    	print(paste(i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i]))
			break
		}
	}
}

##################################
#import testing data
t <- read.delim("test_predictions_split.tsv",header=F)
ta <- read.delim("test_actuals_split.tsv",header=F)

#calculate accuracy metrics
tm <- measureit(score = as.numeric(t$V2), class = as.numeric(ta$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))

for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( tm, s) }
#	print(paste( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 
# [1] "10104 1 0.411184880452935 0.999891382564186 0.852714757036373"
# [1] "22428 0.9969343 0.910472078530406 0.998995288718722 0.976864486171643"
# [1] "25122 0.013922857 0.993197833082155 0.989993618725646 0.990794672314773"
# [1] "31829 0.00040843227 0.996374893079712 0.899990495974366 0.924086595250703"

Jstat( tm )
# [1] "J-stat: 0.985974773600532"
# [1] "ACC: 0.993493136735775"
# print(paste(index, Metrics$Cutoff[index], Metrics$SENS[index], Metrics$SPEC[index]))
# [1] "24797 0.08085117 0.991975886929249 0.993998886671283"

#PLOT ROC
t_rocit_emp <- rocit( score = as.numeric(t$V2), class = as.numeric(ta$V2) , method = "emp")

pdf( "T_ROC.pdf", width=6, height=6)
	plot( t_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))
dev.off()

pdf( "T_RECOV.pdf", width=6, height=6 )
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
	grid()
dev.off()

#Precision Recall
pdf("T_PREC.pdf", width=6, height=6 )
	plot( tm$TP/(tm$TP+tm$FP), tm$TP/(tm$TP+tm$FN), type="l", ylim=c(0,1), xlim=c(0,1), col=3, ylab="Recall", xlab="Precision")
	grid()
dev.off()

##################################
#import validation data

v <- read.delim("val_predictions_split.tsv",header=F)
va <- read.delim("val_actuals_split.tsv",header=F)

#calculate accuracy metrics
vm <- measureit(score = as.numeric(v$V2), class = as.numeric(va$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))

for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( vm, s ) }
#	print(paste( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 
# [1] "10235 1 0.416520711987292 0.999891382564186 0.854048714919962"
# [1] "20394 0.9991339 0.827624129363366 0.998995288718722 0.956152498879883"
# [1] "25105 0.015241252 0.992505396928842 0.989993618725646 0.990621563276445"
# [1] "31807 0.00037272074 0.995478799234247 0.899990495974366 0.923862571789337"

Jstat(vm)
# [1] "J-stat: 0.984522015396522"
# [1] "ACC: 0.994114292696835"
# print(paste(index, Metrics$Cutoff[index], Metrics$SENS[index], Metrics$SPEC[index]))
# [1] "24568 0.43959102 0.988554437701112 0.99596757769541"

Maxcuracy(vm)
# [1] "Macc Accuracy: 0.994368864812024"
# [1] "24423 0.8163857 0.9861105453953 0.997121637950932"

#PLOT ROC
v_rocit_emp <- rocit( score = as.numeric(v$V2), class = as.numeric(va$V2) , method = "emp")
pdf( "V_ROC.pdf", width=6, height=6)
	plot( v_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))
dev.off()

#Ploc accuracy and recovery = f(cutoff)
pdf( "V_RECOV.pdf", width=6, height=6)
	par(mar = c(5, 4, 4, 4) + 0.3)
	plot( vm$Cutoff, vm$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
	axis(2, las=1)
	box()
	par(new=TRUE)
	lines( vm$Cutoff, vm$SENS, type="l", col=4) 
	mtext("% Recovery", side=4, col=4, line=3)
	axis(4, col=4, col.axis=4,las=1)
	axis(1)
	grid()
dev.off()

#Precision Recall
pdf("V_PREC.pdf", width=6, height=6)
	plot( vm$TP/(vm$TP+vm$FP), vm$TP/(vm$TP+vm$FN), type="l", ylim=c(0,1), xlim=c(0,1), col=3, ylab="Recall", xlab="Precision")
	grid()
dev.off()

##################################
#import rep5 data

r <- read.delim("rep5-test_predictions-on-rep2-4_split.tsv",header=F)
ra <- read.delim("rep5-test_actuals_split.tsv",header=F)

r2 <- read.delim("rep5-test_predictions-on-rep2-4_split_sub.tsv", header=F)
ra2 <- read.delim("rep5-test_actuals_split_sub.tsv",header=F)

#calculate accuracy metrics
rm <- measureit(score = as.numeric(r$V2), class = as.numeric(ra$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))
rm2 <- measureit(score = as.numeric(r2$V2), class = as.numeric(ra2$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))


for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( rm, s ) }
# ( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 

Jstat(rm)
# print(paste(index, Metrics$Cutoff[index], Metrics$SENS[index], Metrics$SPEC[index]))


for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( rm2, s ) }
# ( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 
[1] "1988 0.9999919 0.1983 0.999866666666667 0.799475"
[1] "5719 0.99891186 0.5687 0.998966666666667 0.8914"
[1] "8574 0.8951255 0.8272 0.989966666666667 0.949275"
[1] "12592 0.042037927 0.959 0.899966666666667 0.914725"

Jstat(rm2)
[1] "J-stat: 0.880333333333333"
[1] "ACC: 0.9458"
(index, Metrics$Cutoff[index], Metrics$SENS[index], Metrics$SPEC[index]))
[1] "10747 0.22783595 0.9289 0.951433333333333"

 for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( rm, s ) }
[1] "24250 0.99999523 0.17379348965438 0.999899487385667 0.793372987952845"
[1] "84575 0.99834526 0.604195683701162 0.998999660171637 0.900298666054018"
[1] "120947 0.880043 0.838325459845211 0.989998994873857 0.952080611116695"
[1] "176031 0.04012807 0.96380109989518 0.899999521368503 0.915949916000172"

Jstat(rm)
[1] "J-stat: 0.882390668600338"
[1] "ACC: 0.945877546917853"
[1] "150451 0.21432284 0.931830909064802 0.950559759535536"

Maxcuracy(rm)
[1] "Macc Accuracy: 0.955781629165889"
[1] "131883 0.64243907 0.884984851313126 0.97938055511681"


#PLOT ROC
r_rocit_emp <- rocit( score = as.numeric(r$V2), class = as.numeric(ra$V2) , method = "emp")
pdf( "fullROC.pdf", width=6, height=6)
	plot( r_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))
dev.off()

pdf( "fullRECOV.pdf", width=6, height=6)
	par(mar = c(5, 4, 4, 4) + 0.3)
	plot( rm$Cutoff, rm$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
	axis(2, las=1)
	box()
	par(new=TRUE)
	lines( rm$Cutoff, rm$SENS, type="l", col=4) 
	mtext("% Recovery", side=4, col=4, line=3)
	axis(4, col=4, col.axis=4,las=1)
	axis(1)
	grid()
dev.off()

#Precision Recall
pdf("fullPREC.pdf", width=6, height=6)
	plot( rm$TP/(rm$TP+rm$FN), rm$TP/(rm$TP+rm$FP), type="l", ylim=c(0,1), xlim=c(0,1), col=3, xlab="Recall", ylab="Precision")
	grid()
dev.off()



r2_rocit_emp <- rocit( score = as.numeric(r2$V2), class = as.numeric(ra2$V2) , method = "emp")
pdf( "ROC.pdf", width=6, height=6)
	plot( r2_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))
dev.off()

pdf( "RECOV.pdf", width=6, height=6)
	par(mar = c(5, 4, 4, 4) + 0.3)
	plot( rm2$Cutoff, rm2$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
	axis(2, las=1)
	box()
	par(new=TRUE)
	lines( rm2$Cutoff, rm2$SENS, type="l", col=4) 
	mtext("% Recovery", side=4, col=4, line=3)
	axis(4, col=4, col.axis=4,las=1)
	axis(1)
	grid()
dev.off()

#Precision Recall
plot( rm$TP/(rm$TP+rm$FN), rm$TP/(rm$TP+rm$FP), type="l", ylim=c(0,1), xlim=c(0,1), col=3, xlab="Recall", ylab="Precision")

pdf("PREC.pdf", width=6, height=6)
	plot( rm2$TP/(rm2$TP+rm2$FN), rm2$TP/(rm2$TP+rm2$FP), type="l", ylim=c(0,1), xlim=c(0,1), col=3, xlab="Recall", ylab="Precision")
	grid()
dev.off()


##################################
#import REP1 data


r1 <- read.delim("rep1-test_predictions_split.tsv",header=F)
r1a <- read.delim("rep1-test_actuals_split.tsv",header=F)


#calculate accuracy metrics
r1m <- measureit(score = as.numeric(r1$V2), class = as.numeric(r1a$V2) , measure = c("ACC", "SENS", "SPEC", "FSCR"))

for ( s in c(0.9999,0.999,0.99,0.90)) { Sn4Sp( r1m, s ) }
# ( i, Metrics$Cutoff[i], Metrics$SENS[i], Metrics$SPEC[i], Metrics$ACC[i])) 
[1] "831 1 0.0248839255909932 0.999898845830931 0.756145115770946"
[1] "4692 0.9999933 0.139349983309562 0.998998573726216 0.784086426122053"
[1] "20914 0.98338217 0.604618699359694 0.989995852679068 0.893651564349225"
[1] "38807 0.16259591 0.877613570843322 0.899998988458309 0.894402634054563"

Jstat(r1m)
[1] "J-stat: 0.778705024327578"
[1] "ACC: 0.898211088520013"
[1] "37911 0.19139561 0.87163535945134 0.907069664876238"
Maxcuracy(r1m)
[1] "Macc Accuracy: 0.917056110217583"
[1] "28381 0.75454766 0.764725518162231 0.967832974236033"

r1_rocit_emp <- rocit( score = as.numeric(r1$V2), class = as.numeric(r1a$V2) , method = "emp")

pdf( "ROC.pdf", width=6, height=6)
	plot( r1_rocit_emp , legend=F, YIndex=T, col=c(2,"grey90"))
dev.off()

pdf( "RECOV.pdf", width=6, height=6)
	par(mar = c(5, 4, 4, 4) + 0.3)
	plot( r1m$Cutoff, r1m$ACC, type="l" , ylim=c(0,1), xlab="Cutoff", ylab="Accuracy", axes=FALSE)
	axis(2, las=1)
	box()
	par(new=TRUE)
	lines( r1m$Cutoff, r1m$SENS, type="l", col=4) 
	mtext("% Recovery", side=4, col=4, line=3)
	axis(4, col=4, col.axis=4,las=1)
	axis(1)
	grid()
dev.off()

#Precision Recall
pdf("PREC.pdf", width=6, height=6)
	plot( r1m$TP/(r1m$TP+r1m$FN), r1m$TP/(r1m$TP+r1m$FP), type="l", ylim=c(0,1), xlim=c(0,1), col=3, xlab="Recall", ylab="Precision")
	grid()
dev.off()
