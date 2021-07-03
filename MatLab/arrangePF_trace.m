%% 
%all_PFs_4_run_trace    = cell(length(algs),length(modelName),3,length(probs),nRun,NTRACE+1);

nObjMax = max(cell2mat(nobjs));

all_mean_over_trace = zeros(length(algs), length(modelName), 4, length(probs), nObjMax+1, NTRACE+1);
all_ind_mean_over_trace = cell(length(algs), length(modelName), 4, length(probs), nObjMax+1, NTRACE+1);
all_cor_mean_over_trace = zeros(length(algs), length(modelName), 4, length(probs), nObjMax+1, NTRACE+1);
all_min_over_trace = zeros(length(algs), length(modelName), 4, length(probs), nObjMax+1, NTRACE+1);

for iAlg = 1 : length(algs)
    nObj = nobjs{iAlg};
    for iModel = 1 : length(modelName)
        for iTrTs = 1 : 3
            for iProb = 1 : length(probs)
                for iFun = 1:(nObj+1)
                    for iTrace = 1 : (NTRACE + 1)
                        tmp1 = 0;
                        tmpCor1 = 0;
                        tmp2 = 1;
                        tmp_ind = [];
                        for iRun = 1 : nRun
                            if iTrTs == 3
                                tmpPop1 = all_PFs_4_run_trace{iAlg,iModel,1,iProb,iRun,iTrace};
                                tmpPop2 = all_PFs_4_run_trace{iAlg,iModel,2,iProb,iRun,iTrace};
                                tmpPop = (tmpPop1+tmpPop2)./2;
                            else
                                tmpPop = all_PFs_4_run_trace{iAlg,iModel,iTrTs,iProb,iRun,iTrace};
                            end
                            tmp_row = 0;
                            if iFun < nObj+1
                                [tmp,III] = min(tmpPop(:,iFun));
                                tmp_row = III;
                                tmp1 = tmp1 + tmp;
                                if tmp < tmp2
                                    tmp2 = tmp;
                                end
                                if iFun == 1
                                    iFunCor = 2;
                                elseif iFun == 2
                                    iFunCor = 1;
                                else
                                    iFunCor = iFun;
                                end
                                tmp = tmpPop(III,iFunCor);
                                tmpCor1 = tmpCor1 + tmp;
                            else
                                tmpObj1 = 1.-tmpPop(:,1);
                                tmpObj2 = 1.-tmpPop(:,2);
                                tmpObj  = 2.*tmpObj1.*tmpObj2./(tmpObj1+tmpObj2);
                                [tmp,III] = max(tmpObj);
                                tmp_row = III;
                                tmp1 = tmp1 + tmp;
                                if 1-tmp < tmp2
                                    tmp2 = 1-tmp;
                                end
                            end
                            tmp_ind = [tmp_ind; [iRun tmp_row]];
                        end
                        all_ind_mean_over_trace{iAlg,iModel,iTrTs,iProb,iFun,iTrace} = tmp_ind;
                        all_cor_mean_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmpCor1/nRun;
                        if iFun < nObj
                            all_mean_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmp1/nRun;
                            all_min_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmp2;
                        elseif iFun == nObj
                            all_mean_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmp1/nRun;
                            all_min_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmp2;
                        else
                            all_mean_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmp1/nRun;
                            all_min_over_trace(iAlg,iModel,iTrTs,iProb,iFun,iTrace) = tmp2;
                        end
                    end
                end
            end
        end
    end
end

%%
all_min_over_run = zeros(length(algs), length(modelName), 4, length(probs), nObjMax+1, nRun);
all_minInd_over_run = zeros(length(algs), length(modelName), 4, length(probs), nObjMax+1, nRun);
all_minInv_over_run = cell(length(algs), length(modelName), 4, length(probs), nObjMax+1, nRun);

for iAlg = 1 : length(algs)
    nObj = nobjs{iAlg};
    for iModel = 1 : length(modelName)
        for iTrTs = 1 : 3
            for iProb = 1 : length(probs)
                for iFun = 1:(nObj+1)
                    for iRun = 1 : nRun
                        tmp1 = 0;
                        tmpCor1 = 0;
                        tmp2 = 100;
                        tmp_ind = [];
                        tmp_III = [];
                        tmp_smp = 100;
                        for iTrace = 1 : (NTRACE + 1)
                            if iTrTs == 3
                                tmpPop1 = all_PFs_4_run_trace{iAlg,iModel,1,iProb,iRun,iTrace};
                                tmpPop2 = all_PFs_4_run_trace{iAlg,iModel,2,iProb,iRun,iTrace};
                                tmpPop = (tmpPop1+tmpPop2)./2;
                            else
                                tmpPop = all_PFs_4_run_trace{iAlg,iModel,iTrTs,iProb,iRun,iTrace};
                            end
                            tmp_row = 0;
                            if iFun < nObj + 1
                                [tmp,III] = min(tmpPop(:,iFun));
                                tmp_row = III;
                                smp = tmpPop(III,2);
                                tmp1 = tmp1 + tmp;
                                if tmp < tmp2 || ( tmp == tmp2 && iFun == 1 && smp < tmp_smp )
                                    tmp2 = tmp;
                                    tmp_ind = iTrace;
                                    tmp_III = III;
                                    tmp_smp = smp;
                                end
                                if iFun == 1
                                    iFunCor = 2;
                                elseif iFun == 2
                                    iFunCor = 1;
                                else
                                    iFunCor = iFun;
                                end
                                tmp = tmpPop(III,iFunCor);
                                tmpCor1 = tmpCor1 + tmp;
                            else
                                tmpObj1 = 1.-tmpPop(:,1);
                                tmpObj2 = 1.-tmpPop(:,2);
                                tmpObj  = 2.*tmpObj1.*tmpObj2./(tmpObj1+tmpObj2);
                                [tmp,III] = max(tmpObj);
                                tmp_row = III;
                                tmp1 = tmp1 + tmp;
                                if 1-tmp < tmp2
                                    tmp2 = 1-tmp;
                                    tmp_ind = iTrace;
                                    tmp_III = III;
                                end
                            end
                        end
                        if iFun < nObj+1
                            all_min_over_run(iAlg,iModel,iTrTs,iProb,iFun,iRun) = tmp2;
                            all_minInd_over_run(iAlg,iModel,iTrTs,iProb,iFun,iRun) = tmp_ind;
                        else
                            all_min_over_run(iAlg,iModel,iTrTs,iProb,iFun,iRun) = 1-tmp2;
                            all_minInd_over_run(iAlg,iModel,iTrTs,iProb,iFun,iRun) = tmp_ind;
                        end
                        all_minInv_over_run{iAlg,iModel,iTrTs,iProb,iFun,iRun} = [tmp_ind; tmp_III];
                    end
                end
            end
        end
    end
end

%% 
all_minF_4_run          = cell(length(algs),length(modelName),4,length(probs),nRun);
all_minF_4_run_mean     = cell(length(algs),length(modelName),4,length(probs));
all_minF_4_run_min      = cell(length(algs),length(modelName),4,length(probs));
all_minF_4_run_max      = cell(length(algs),length(modelName),4,length(probs));
all_minF_4_run_std      = cell(length(algs),length(modelName),4,length(probs));

for iAlg = 1:length(algs)
    for i=1:length(modelName)
        for j=1:length(probs)
            for k=1:4
                for n=1:nRun
                    tmp_min_run = ones(1,nobjs{iAlg});
                    for iTrace = 1 : (NTRACE+1)
                        tmp_PF = all_PFs_4_run_trace{iAlg,i,k,j,n,iTrace};
                        tmp_min = min(tmp_PF);
                        tmp_min_run = min([tmp_min; tmp_min_run]);
                    end
                    all_minF_4_run{iAlg,i,k,j,n} = tmp_min_run(1);
                end
                all_minF_4_run_mean{iAlg,i,k,j} = mean([all_minF_4_run{iAlg,i,k,j,:}]);
                all_minF_4_run_min{iAlg,i,k,j}  = min([all_minF_4_run{iAlg,i,k,j,:}]);
                all_minF_4_run_max{iAlg,i,k,j}  = max([all_minF_4_run{iAlg,i,k,j,:}]);
                all_minF_4_run_std{iAlg,i,k,j}  = std([all_minF_4_run{iAlg,i,k,j,:}]);
            end
        end
    end
end


