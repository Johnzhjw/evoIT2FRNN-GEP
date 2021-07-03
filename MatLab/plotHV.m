%%

legendName = cell(length(algs),length(models),2);

for iAlg = 1:lenINDs
    iAlg = indAlg(iiii);
    for i = 1:length(models)
        legendName{iiii,i,1} = sprintf('%s-TRAIN', algs{iAlg});%, models{i});
        legendName{iiii,i,2} = sprintf('%s-TEST',  algs{iAlg});%, models{i});
        legendName{iiii,i,3} = sprintf('%s-FINAL',  algs{iAlg});%, models{i});
        legendName{iiii,i,4} = sprintf('%s-VALIDATION',  algs{iAlg});%, models{i});
    end
end

%%

for iInst = 1:length(allprobnames)
    strInst=sprintf('%s',allprobnames{iInst});
    tmpMean = [];
    tmpStd  = [];
    for iAlg = tarINDs
        for a = 1:length(models)
            for b = 1:3
                tmpMean = [tmpMean; all_meanHVs{iAlg,a,iInst,b}];
                tmpStd = [tmpStd; all_stdHVs{iAlg,a,iInst,b}];
            end
        end
    end
    maxFEs=10000;
%    FEs4grp=0;
    iters=zeros(1,26);
    for I = 1
        for i = 1:26
            iters(I,i)   = (i-1)*maxFEs/25;
        end
    end
%     for I = 1:9
%         for i = 1:26
%             iters(I,i) = FEs4grp + (i-1)*(maxFEs-FEs4grp)/25;
%         end
%     end
%             iters(4,:)=[];
%             tmpMean(4,:)=[];
%             tmpStd(4,:)=[];
    genfigHV(iters,tmpMean, tmpStd, strInst, legendName);
    fname=sprintf('FIGUREs/HV_%s.eps', allprobnames{iInst});
    print(fname,'-depsc2','-r300');
    fname=sprintf('FIGUREs/HV_%s.fig', allprobnames{iInst});
    savefig(fname);
%     close;
end
