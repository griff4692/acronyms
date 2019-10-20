library("easyPubMed")

lf <- list.files('./data', pattern='*.xml.gz$')

for (fname in lf)
{
  lt <- articles_to_list(paste('./data/', fname, sep=""))
  for (i in 1:length(lt))
  {
    x <- article_to_df(lt[i], max_chars=-1, getAuthors=FALSE)$abstract[1]
    write(x, file=paste('./abstracts/', fname, ".txt", sep=""), append=TRUE)
  }
}