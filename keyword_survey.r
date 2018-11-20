data<-read.csv('infection_ma.csv')

keyword='OR';window_prev=5;window_next=40
data.sub<-data[apply(matrix(data$abstract),1,function(x)grepl(keyword,x)),]
abstract.segments<-apply(matrix(data.sub$abstract),1,function(x){
	indexs<-gregexpr(pattern = keyword,x)[[1]]
	keystr<-apply(matrix(indexs),1,function(y)substr(x,y-window_prev,y+window_next))
	return(keystr)
})
abstract.segments<-cbind(
	unlist(lapply(seq_along(abstract.segments),function(i){return(rep(i,length(abstract.segments[[i]])))})),
	unlist(abstract.segments)
)
colnames(abstract.segments)<-c('studyID','text')
write.csv(abstract.segments,'abstract_segs_or.csv')

sum(apply(matrix(data$abstract),1,function(x)grepl('OR', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('RR', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('SMD', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('sensitivity', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('specificity', x)))

sum(apply(matrix(data$abstract),1,function(x)grepl('CI', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('C.I.', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('CrI', x)))
sum(apply(matrix(data$abstract),1,function(x)grepl('95%', x)))

