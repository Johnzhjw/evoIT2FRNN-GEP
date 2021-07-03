%%
all_nd_prc = cell(size(all_PFs,1),length(modelName),length(probs));
all_nd_prc_X = cell(size(all_PFs,1),length(modelName),length(probs));

for iAlg = 1:size(all_PFs,1)
    nObj = nobjs{iAlg};
    for i=1:length(modelName)
        for j=1:length(probs)
            if iAlg > length(algs)
                fprintf('PF_%s_%s_%s\n',...
                    'BP', modelName{i}, probName{iAlg}{j});
            else
                fprintf('PF_%s_%s_%s\n',...
                    algs{iAlg}, modelName{i}, probName{iAlg}{j});
            end
            tmp_nonDom = [all_PFs{iAlg,i,1,j}(:,1) all_PFs{iAlg,i,2,j}(:,1) all_PFs{iAlg,i,1,j}(:,(nObj+1):end)];
            tmp = NDSelection_with_nObj(tmp_nonDom,2);
            all_nd_prc{iAlg,i,j} = tmp;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for iAlg = 1:length(algs)
%     nObj = 2;%nobjs{iAlg};
%     for i=1:length(modelName)
%         for j=1:length(probs)
%             tmp = all_nd_prc{iAlg,i,j};
%             tmpSol = zeros(size(tmp,1),dims{iAlg}{1});
%             for a = 1 : size(tmp,1)
%                 iRun = tmp(a,nObj+1);
%                 iInd = tmp(a,nObj+2);
%                 switch iAlg
% %                         case {1 2 3}
% %                             switch k
% %                                 case 1
% %                                     fname = sprintf('%s_%s_EXP0_OBJ3_VAR%d_RUN%d_TRAIN',...
% %                                         PFstrs{iAlg}, probName{iAlg}{j}, dims{i}, n);
% %                                 case 2
% %                                     fname = sprintf('%s_%s_EXP0_OBJ3_VAR%d_RUN%d_TEST',...
% %                                         PFstrs{iAlg}, probName{iAlg}{j}, dims{i}, n);
% %                                 case 3
% %                                     fname = sprintf('%s_%s_EXP0_OBJ3_VAR%d_RUN%d',...
% %                                         PFstrs{iAlg}, probName{iAlg}{j}, dims{i}, n);
% %                             end
% %                         case 4
% %                             switch k
% %                                 case 1
% %                                     fname = sprintf('%s_%s_EXP0_OBJ3_VAR%d_RUN%d_TRAIN',...
% %                                         PFstrs{iAlg}, probName{iAlg}{j}, dims{i}, n-1);
% %                                 case 2
% %                                     fname = sprintf('%s_%s_EXP0_OBJ3_VAR%d_RUN%d_TEST',...
% %                                         PFstrs{iAlg}, probName{iAlg}{j}, dims{i}, n-1);
% %                                 case 3
% %                                     fname = sprintf('%s_%s_EXP0_OBJ3_VAR%d_RUN%d',...
% %                                         PFstrs{iAlg}, probName{iAlg}{j}, dims{i}, n-1);
% %                             end
%                     case {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27}
%                         fname = sprintf('%s_%s_OBJ%d_VAR%d_MPI%d_RUN%d',...
%                             PSstrs{iAlg}, probName{iAlg}{j}, nobjs{iAlg}, dims{iAlg}{iRun}, nMPI{iAlg}, iRun);
%                 end
%                 ffile = sprintf('%s/%s/%s/', mainfils{iAlg}, fnms{iAlg}, PSfilestrs{iAlg});
%                 fname = dir([ffile fname]);
%                 fname2 = sprintf('%s%s',ffile,fname.name);
%                 tmpdata = importdata(fname2);
%                 cur_dim = size(tmpdata,2);
%                 tmpSol(a,1:cur_dim) = tmpdata(iInd,:);
%             end
%             all_nd_prc_X{iAlg,i,j} = tmpSol;
%             savefilename = sprintf('PS_ND_%02d_%s', iAlg,algs{iAlg});
%             save(savefilename,'tmpSol','-ascii');
%         end
%     end
% end

%%
all_prc = zeros(length(algs),length(modelName),4,length(probs));

for iAlg = 1:length(algs)
    for i=1:length(modelName)
        for j=1:length(probs)
            for k=1:3
                tmp_nonDom = all_ND_PFs{iAlg,i,k,j};
                all_prc(iAlg,i,k,j) = min(tmp_nonDom(:,1));
            end
        end
    end
end

%%
%all_prc_x_algs = zeros(length(modelName)*3*length(probs),length(algs));

all_prc_x_algs = [];
iRow = 0;
indAlgs = tarINDs;
for j=1:length(probs)
    for i = 1:length(modelName)
        for k=1:2
            iRow = iRow + 1;
            for iiii = 1:lenINDs %length(algs)
                iAlg = indAlgs(iiii);
                all_prc_x_algs(iRow,iiii) = all_prc(iAlg,i,k,j);
            end
        end
    end
end

%%
%all_prc_x_models = zeros(length(algs)*3*length(probs),length(modelName));

all_prc_x_models = [];
iRow = 0;
for iAlg = tarINDs % 1:length(algs)
    for j=1:length(probs)
        for k=1:2
            iRow = iRow + 1;
            for i = 1:length(modelName)
                all_prc_x_models(iRow,i) = all_prc(iAlg,i,k,j);
            end
        end
    end
end

%%
all_minPrc_store = cell(length(algs),length(modelName),4,length(probs));
all_minPrc_mean = zeros(length(algs),length(modelName),4,length(probs));

for iAlg = 1:length(algs)
    for i=1:length(modelName)
        for j=1:length(probs)
            for k=1:3
                tmp_prc=[];
                switch k
                    case 1
                        fprintf('PF_%s_%s_%s_TRAIN\n',...
                            algs{iAlg}, modelName{i}, probs{j});
                    case 2
                        fprintf('PF_%s_%s_%s_TEST\n',...
                            algs{iAlg}, modelName{i}, probs{j});
                    case 3
                        fprintf('PF_%s_%s_%s_FINAL\n',...
                            algs{iAlg}, modelName{i}, probs{j});
                    case 4
                        fprintf('PF_%s_%s_%s_VALIDATION\n',...
                            algs{iAlg}, modelName{i}, probs{j});
                end
                fprintf('Run - ');
                for n=1:nRun
                    fprintf('%d ',n);
                    ind_s = (n - 1) * nPop_all{iAlg} + 1;
                    ind_f = n * nPop_all{iAlg};
                    cur_minPrc = all_PFs{iAlg,i,k,j}(ind_s:ind_f,1);
                    tmp_prc = [tmp_prc; min(cur_minPrc)];
                end
                all_minPrc_store{iAlg,i,k,j} = tmp_prc;
                all_minPrc_mean(iAlg,i,k,j) = mean(tmp_prc);
                fprintf('\n');
            end
        end
    end
end

%%
all_minPrc_avg_algs = [];

iRow = 0;
indAlgs = tarINDs;
for j=1:length(probs)
    for i = 1:length(modelName)
        for k=1:2
            iRow = iRow + 1;
            for iiii = 1:lenINDs %length(algs)
                iAlg = indAlgs(iiii);
                all_minPrc_avg_algs(iRow,iiii) = all_minPrc_mean(iAlg,i,k,j);
            end
        end
    end
end

%%
%all_prc_x_models = zeros(length(algs)*3*length(probs),length(modelName));

all_minPrc_avg_models = [];
iRow = 0;
for iAlg = tarINDs % 1:length(algs)
    for j=1:length(probs)
        for k=1:2
            iRow = iRow + 1;
            for i = 1:length(modelName)
                all_minPrc_avg_models(iRow,i) = all_minPrc_mean(iAlg,i,k,j);
            end
        end
    end
end

[tmp_min, tmp_ind] = min(all_minPrc_avg_models, [], 2);
all_minPrc_ANOVA_p_models = [];
all_minPrc_Friedman_p_models = [];
iRow = 0;
for iAlg = tarINDs % 1:length(algs)
    for j=1:length(probs)
        for k=1:2
            iRow = iRow + 1;
            for i = 1:length(modelName)
                tmp_set1 = all_minPrc_store{iAlg,i,k,j};
                tmp_set2 = all_minPrc_store{iAlg,tmp_ind(iRow),k,j};
                all_minPrc_ANOVA_p_models(iRow,i) = anova1([tmp_set1 tmp_set2],[],'off');
                all_minPrc_Friedman_p_models(iRow,i) = friedman([tmp_set1 tmp_set2],1,'off');
            end
        end
    end
end

all_minPrc_Friedman_sig_models = all_minPrc_Friedman_p_models<0.05;

all_minPrc_avg_models2 = [];
iRow = 0;
for j=1:length(probs)
    for k=1:2
        for iAlg = tarINDs % 1:length(algs)
            iRow = iRow + 1;
            for i = 1:length(modelName)
                all_minPrc_avg_models2(iRow,i) = all_minPrc_mean(iAlg,i,k,j);
            end
        end
    end
end
