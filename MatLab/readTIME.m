%% 
all_TIMEs = zeros(length(algs),length(modelName),length(probs));

for iAlg = 1:length(algs)
    for i = 1:length(modelName)
        for j = 1:length(probs)
            tmp_TIME=[];
            fname0 = sprintf('%s_MPI%d_*', TIMEstrs{iAlg}, nMPI{iAlg});
            ffile = sprintf('%s/%s/OUTPUT/', mainfils{iAlg}, fnms{iAlg});
            fname = dir([ffile fname0]);
            fname2 = sprintf('%s%s',ffile,fname.name);
            tmp    = importdata(fname2);
            if j <= 3
                tmp_TIME = mean(tmp.data(mod(iAlg,8)*3+j,1:nRun));
                all_TIMEs(iAlg,i,j) = tmp_TIME;
            elseif j <= 11
                tmp_TIME = mean(tmp.data(24+mod(iAlg,8)*8+j-3,1:nRun));
                all_TIMEs(iAlg,i,j) = tmp_TIME;
            else
                disp('Error obtaining time.');
            end
        end
    end
end
