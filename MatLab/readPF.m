%% 
all_ND_PFs = cell(length(algs),length(modelName),4,length(probs));
all_PFs    = cell(length(algs),length(modelName),4,length(probs));

for iAlg = 1:length(algs)
    for i=1:length(modelName)
        for j=1:length(probs)
            for k=1:3
                tmp_nonDom=[];
                tmp_PF=[];
                switch k
                    case 1
                        fprintf('PF_%s_%s_%s_TRAIN\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                    case 2
                        fprintf('PF_%s_%s_%s_TEST\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                    case 3
                        fprintf('PF_%s_%s_%s_FINAL\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                    case 4
                        fprintf('PF_%s_%s_%s_VALIDATION\n',...
                            algs{iAlg}, modelName{i}, probName{iAlg}{j});
                end
                fprintf('Run - ');
                for n=1:nRun
                    switch k
                        case 1
                            fname = sprintf('%s_FUN_%s_TRAIN_OBJ%d_VAR%d_MPI%d_RUN%d',...
                                filestrs{iAlg}, probName{iAlg}{j}, nobjs{iAlg}, dims{iAlg}{j}{n}, nMPI{iAlg}, n);
                        case 2
                            fname = sprintf('%s_FUN_%s_TEST_OBJ%d_VAR%d_MPI%d_RUN%d',...
                                filestrs{iAlg}, probName{iAlg}{j}, nobjs{iAlg}, dims{iAlg}{j}{n}, nMPI{iAlg}, n);
                        case 3
                            fname = sprintf('%s_FUN_%s_OBJ%d_VAR%d_MPI%d_RUN%d',...
                                filestrs{iAlg}, probName{iAlg}{j}, nobjs{iAlg}, dims{iAlg}{j}{n}, nMPI{iAlg}, n);
                        case 4
                            fname = sprintf('%s_FUN_%s_VALIDATION_OBJ%d_VAR%d_MPI%d_RUN%d',...
                                filestrs{iAlg}, probName{iAlg}{j}, nobjs{iAlg}, dims{iAlg}{j}{n}, nMPI{iAlg}, n);
                    end
                    ffile = sprintf('%s/%s/PF/', mainfils{iAlg}, fnms{iAlg});
                    fname1 = dir([ffile fname]);
                    fprintf('%d ',n);
                    fname2 = sprintf('%s%s',ffile,fname1.name);
                    tmp   = importdata(fname2);
                    nObj  = size(tmp,2);
                    tmp = [tmp ones(size(tmp,1),1)*n (1:size(tmp,1))'];
                    tmp_nonDom = [tmp_nonDom; tmp];
                    tmp_nonDom = NDSelection_with_nObj(tmp_nonDom,nObj);
                    tmp_PF = [tmp_PF; tmp];
                end
                all_ND_PFs{iAlg,i,k,j} = tmp_nonDom;
                all_PFs{iAlg,i,k,j} = tmp_PF;
                fprintf('\n');
            end
        end
    end
end

%%
% tmp_head = 'FUN_Classify_CNN_Indus_BP_RUN_';
% 
% tmp_cell = cell(1,length(modelName),4,length(probs));
% 
% tmp_PF = [];
% 
% for iRun = 1:5
%     tfn = sprintf('%s%d', tmp_head, iRun);
%     tmp = importdata(tfn);
%     tmp_PF = [tmp_PF; tmp];
% end
% 
% tmp_ind = [(1:size(tmp_PF,1))' ones(size(tmp_PF,1),1)];
% 
% tmp_train = [tmp_PF(:,1:2) tmp_ind];
% tmp_test  = [tmp_PF(:,3:4) tmp_ind];
% 
% tmp_cell{1,1,1,1} = tmp_train;
% tmp_cell{1,1,2,1} = tmp_test;
% tmp_cell{1,1,3,1} = tmp_test;
% 
% all_ND_PFs = [all_ND_PFs; tmp_cell];
% all_PFs = [all_PFs; tmp_cell];

%%

% function parsave(fname, x)
% save(fname, 'x')
% 
% function parsavefig(fname)
% savefig(fname)

