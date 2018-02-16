#@Author: Ryan Cloke

library(Peptides)
apath = paste(getwd(),"/9mer_HLA_A_0201.csv",sep='')
fin = scan(apath,what="character",skip=1)
df <- data.frame()
for (i in 1:length(fin)) {
  aline = strsplit(fin[i],",")
  aseq = aline[[1]][4]
  binding = aline[[1]][6]
  
  pepCharge = Peptides::charge(aseq,pH = 7, pKscale = "EMBOSS")
  pepBoman = Peptides::boman(aseq)
  crucPolarity = Peptides::crucianiProperties(aseq)[[1]][1]
  crucHydroPhob = Peptides::crucianiProperties(aseq)[[1]][2]
  crucHBond = Peptides::crucianiProperties(aseq)[[1]][3]
  pepHMoment = Peptides::hmoment(aseq)
  pepHydroPhob = Peptides::hydrophobicity(aseq,scale = "KyteDoolittle")
  pepInsta = Peptides::instaIndex(aseq)
  #pepKidera = Peptides::kideraFactors(aseq)
  #pepMem = Peptides::membpos(aseq)
  pepWhim = Peptides::mswhimScores(aseq)[[1]][1]
  pepPI = Peptides::pI(aseq)
  #pepST =Peptides::stScales(aseq) - 8 results
  #pepTS = Peptides::tScales(aseq) - 5 results
  #pepVH = Peptides::vhseScales(aseq) - 8 results
  pepZ1 = Peptides::zScales(aseq)[[1]][1]
  pepZ2 = Peptides::zScales(aseq)[[1]][2]
  pepZ3 = Peptides::zScales(aseq)[[1]][3]
  pepZ4 = Peptides::zScales(aseq)[[1]][4]
  pepZ5 = Peptides::zScales(aseq)[[1]][5]
  
  newline = c(pepCharge,pepBoman,crucPolarity,crucHydroPhob,crucHBond,pepHMoment,pepHydroPhob,pepInsta,pepWhim,pepPI,pepZ1,pepZ2,pepZ3,pepZ4,pepZ5)
  df <-rbind(df,newline)
 
}
apath = paste(getwd(),"/seq_binding_phys.csv",sep='')
write.table(df, file = apath,row.names=FALSE,col.names=FALSE, sep = ",")
