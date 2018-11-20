library(stringr)
data<-read.csv('infection_ma.csv')

# Select studies containing keyword: 'OR'
keyword='OR'
data.sub<-data[apply(matrix(data$abstract),1,function(x)grepl(keyword,x)),]

# Find all index containing keyword -> judge containing 3 sets of numbers? -> 
# if yes, extract window from keyword to the third number
window_prev=2;window_next=40
float_pattern <- "[+-]?([0-9]*[.])?[0-9]+"
float_pattern_4range <- "([^0-9][-])?([0-9]*[.])?[0-9]+"
range_pattern <- "[0-9]\\s*[-]\\s*[0-9]|[0-9]\\s*[-]\\s*[\\.]"
abst.seg<-apply(matrix(data.sub$abstract),1,function(x){
	text<-''
	indexs<-gregexpr(pattern = keyword,x)[[1]]
	keystr<-lapply(seq_along(indexs),function(i){
		desc<-substr(x,indexs[i]-window_prev,indexs[i]+window_next)
		
		# replace '95%', '95 CI', '95 %'
		desc<-gsub('95%','___',desc); desc<-gsub('95 %','____',desc); desc<-gsub('95 C','____',desc)
		values<-str_extract_all(desc,float_pattern)[[1]]
		
		# if more than three floats matched
		if(length(values)>=3){
			i_find<-str_locate_all(desc,float_pattern)[[1]]
			
			# if the '-' symbol is for range
			if(dim(str_locate_all(desc,range_pattern)[[1]])[1]>0){
				i_find<-str_locate_all(desc,float_pattern_4range)[[1]]
			}
			
			# prevent if there is other 'OR' within segmented text
			i_first_start<-i_find[1,1]; i_first_end<-i_find[1,2]
			i_second_start<-i_find[2,1]; i_second_end<-i_find[2,2]
			i_third_start<-i_find[3,1]; i_third_end<-i_find[3,2]
			if((indexs[i+1]>i_third_end)|(indexs[i]==max(indexs))){
				return(c(substr(desc,i_first_start,i_third_end),     # text
						substr(desc,i_first_start,i_first_end),      # mean
						substr(desc,i_second_start,i_second_end),    # lb
						substr(desc,i_third_start,i_third_end)))     # ub
			} else {
				return(NA)
			}
		} else {
			return(NA)
		}
	})
	keystr<-unlist(keystr[!is.na(keystr)])
	return(keystr)
})

abst.seg<-lapply(abst.seg,function(x){
	if(!is.null(x)){
		df<-data.frame(matrix(x,ncol=4,byrow=TRUE))
		out.str<-apply(df,1,function(y){
			# Fixed-length output
			return(paste(sprintf("% 6s", y[2]),'[',sprintf("% 6s", y[3]),',',sprintf("% 6s", y[4]),']',sep=''))
		})
		df.with.out<-cbind(as.character(df$X1),out.str)
		colnames(df.with.out)<-c('input','output')
		return(df.with.out)
	} else {
		return(NULL)
	}
})
abst.seg<-abst.seg[unlist(lapply(abst.seg,function(x){!is.null(x)}))]

abst.df<-NULL
for(seg in abst.seg){
	abst.df<-rbind(abst.df,seg)
}

write.csv(abst.df,'abstsegs_regex_or.csv')

